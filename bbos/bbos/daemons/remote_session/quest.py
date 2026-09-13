"""Quest packet decode for the remote_session daemon.

Turns a raw 140-byte `quest_state` packet from a Quest client into the same
fields the quest daemon publishes on quest.controllers: head-relative poses,
triggers, squeeze, thumbsticks and buttons.

Wire format must match bb-quest-app/app/scripts/protocol.gd.
"""

import queue
import struct
import time

import numpy as np

from bbos import Config

CFG_Q = Config("quest")


# ============================================================================
# Wire format
# ============================================================================
MAGIC_STATE = b"BBQ1"
# magic, seq, t_usec, head 7f, 2x (11f + buttons).
_STATE_FMT = "<4sIQ7f11fI11fI"
_STATE_SIZE = struct.calcsize(_STATE_FMT)   # 140 bytes.
BIT_AX = 1 << 0                 # A (right) / X (left).
BIT_BY = 1 << 1                 # B (right) / Y (left).
BIT_THUMB_CLICK = 1 << 2
BIT_POSE_VALID = 1 << 8
SEQ_REORDER_WINDOW = 64
QUAT_EPS = 1e-9                 # Below this a quaternion norm is unusable.
FWD_EPS = 1e-6                  # Below this the head has no ground heading.

# --- Frame transforms (verbatim from bbos/daemons/quest/daemon.py) ----------
T_robot_openxr = np.array([[0, 0, -1, 0], [-1, 0, 0, 0],
                           [0, 1, 0, 0], [0, 0, 0, 1]])
T_headlocal_from_world = np.array([[0, -1, 0, 0], [0, 0, 1, 0],
                                   [1, 0, 0, 0], [0, 0, 0, 1]])
_a = np.radians(CFG_Q.controller_angle_deg)
T_controller_frame = np.array([
    [np.sin(_a), 0, np.cos(_a), 0],
    [0, -1, 0, 0],
    [np.cos(_a), 0, -np.sin(_a), 0],
    [0, 0, 0, 1],
])
const_head_vuer_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 1.5],
                                [0, 0, 1, -0.2], [0, 0, 0, 1]])


# ============================================================================
# Frame math
# ============================================================================
def fast_mat_inv(mat):
    """Invert a rigid transform by transposing R and re-solving t."""
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def rotation_to_quaternion(R):
    """Rotation matrix -> xyzw quaternion, via the largest-trace branch."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def controller_rel_ground(T_head, T_controller):
    """Express a controller in the head's ground frame, as xyz + xyzw."""
    # Column 2 is the head FORWARD axis, not the up axis.
    fwd_head_xy = T_head[:2, 2]
    norm = np.linalg.norm(fwd_head_xy)
    fwd_head_xy = (fwd_head_xy / norm if norm > FWD_EPS
                   else np.array([0.0, 1.0]))
    T_ground = np.eye(4)
    T_ground[0, 0] = fwd_head_xy[0]
    T_ground[1, 0] = fwd_head_xy[1]
    T_ground[0, 1] = -fwd_head_xy[1]
    T_ground[1, 1] = fwd_head_xy[0]
    T_ground[:2, 3] = T_head[:2, 3]
    T_world = T_head @ T_controller
    T_gc = fast_mat_inv(T_ground) @ T_world
    return np.concatenate([T_gc[:3, 3], rotation_to_quaternion(T_gc[:3, :3])])


def pose_to_mat(px, py, pz, qx, qy, qz, qw):
    """Position + xyzw quaternion -> 4x4. Zeros if the quaternion is null."""
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n < QUAT_EPS:
        return np.zeros((4, 4))
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    M = np.eye(4)
    M[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    M[0, 1] = 2 * (qx * qy - qz * qw)
    M[0, 2] = 2 * (qx * qz + qy * qw)
    M[1, 0] = 2 * (qx * qy + qz * qw)
    M[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    M[1, 2] = 2 * (qy * qz - qx * qw)
    M[2, 0] = 2 * (qx * qz - qy * qw)
    M[2, 1] = 2 * (qy * qz + qx * qw)
    M[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    M[0, 3], M[1, 3], M[2, 3] = px, py, pz
    return M


# ============================================================================
# Decoder
# ============================================================================
class QuestDecoder:
    """Decode a raw Quest packet into what quest.controllers publishes.

    Stateful: drops reordered packets and holds the last valid pose per hand.
    """

    def __init__(self):
        """Start with identity poses and no sequence history."""
        self.last_seq = -1
        self.const_left = np.eye(4)
        self.const_right = np.eye(4)
        self.const_head = const_head_vuer_mat

    def _controller(self, v, i, is_left):
        """One hand's pose matrix and button/axis dict from the tuple at i."""
        mat = pose_to_mat(v[i], v[i + 1], v[i + 2], v[i + 3], v[i + 4],
                          v[i + 5], v[i + 6])
        if np.linalg.det(mat) == 0:
            mat = self.const_left if is_left else self.const_right
        elif is_left:
            self.const_left = mat
        else:
            self.const_right = mat
        btn = int(v[i + 11])
        st = {
            "thumb_click": 1.0 if (btn & BIT_THUMB_CLICK) else 0.0,
            "a": 1.0 if (btn & BIT_AX) else 0.0,
            "b": 1.0 if (btn & BIT_BY) else 0.0,
            "trigger": float(v[i + 9]), "squeeze": float(v[i + 10]),
            "thumb": np.array([float(v[i + 7]), float(v[i + 8])]),
        }
        return mat, st

    def decode(self, data):
        """Return the quest.controllers-equivalent dict, or None if stale."""
        if len(data) != _STATE_SIZE:
            return None
        v = struct.unpack(_STATE_FMT, data)
        if v[0] != MAGIC_STATE:
            return None
        seq = v[1]
        if 0 <= self.last_seq - seq < SEQ_REORDER_WINDOW:
            return None
        self.last_seq = seq
        head_mat = pose_to_mat(v[3], v[4], v[5], v[6], v[7], v[8], v[9])
        if np.linalg.det(head_mat) == 0:
            head_mat = self.const_head
        else:
            self.const_head = head_mat
        left_mat, ls = self._controller(v, 10, True)
        right_mat, rs = self._controller(v, 22, False)
        head_world = T_robot_openxr @ head_mat @ fast_mat_inv(T_robot_openxr)
        T_head = head_world @ fast_mat_inv(T_headlocal_from_world)
        left_world = T_robot_openxr @ left_mat @ fast_mat_inv(T_robot_openxr)
        right_world = (T_robot_openxr @ right_mat
                       @ fast_mat_inv(T_robot_openxr))
        T_left = (T_headlocal_from_world
                  @ (fast_mat_inv(head_world) @ left_world)
                  @ T_controller_frame)
        T_right = (T_headlocal_from_world
                   @ (fast_mat_inv(head_world) @ right_world)
                   @ T_controller_frame)
        return {
            "T_head": T_head,
            "left_pose": controller_rel_ground(T_head, T_left),
            "right_pose": controller_rel_ground(T_head, T_right),
            "left_trigger": ls["trigger"], "left_squeeze": ls["squeeze"],
            "left_thumbstick": ls["thumb"],
            "left_thumbstick_click": ls["thumb_click"],
            "left_a": ls["a"], "left_b": ls["b"],
            "right_trigger": rs["trigger"], "right_squeeze": rs["squeeze"],
            "right_thumbstick": rs["thumb"],
            "right_thumbstick_click": rs["thumb_click"],
            "right_a": rs["a"], "right_b": rs["b"],
        }


# ============================================================================
# Controllers
# ============================================================================
MENU_NORMAL = 0
MENU_SELECT = 1

# Only X and B change meaning between menus.
MENU_EVENTS = {
    MENU_NORMAL: {"x": "teleop", "b": "home"},
    MENU_SELECT: {"x": "no_headset", "b": "slow"},
}

# Haptic patterns, forwarded to the headset by session.py. Queued with
# timestamps and sent one per tick so the gaps survive.
HAPTIC_SHORT = {"freq": 160.0, "amp": 0.85, "dur": 0.15}
HAPTIC_LONG = {"freq": 160.0, "amp": 0.85, "dur": 0.45}
HAPTIC_GAP = 0.2
HAPTIC_GAP_LONG = 0.45
HAPTIC_LEFT = 0                 # Hand code: 0 = left, 1 = right, 2 = both.
HAPTIC_RIGHT = 1
HAPTIC_BOTH = 2

# Purr: a rumble that grows with how far you've drifted from where you
# engaged -> a felt cue to come back and use the grip as a clutch.
PURR_FREQ = 160.0
PURR_DUR = 0.12
PURR_AMP_MIN = 0.0
PURR_AMP_MAX = 0.18
PURR_FULL_DRIFT = 0.15          # Drift (m) at PURR_AMP_MAX; smoothstep under.
PURR_THRESH = 0.15              # Squeeze above which precision/purr engages.
PURR_INTERVAL = 0.1             # < dur, so consecutive pulses overlap.

TRIG_TH = 0.5                   # Trigger press threshold, 0..1.
TOGGLE_HOLD = 2.0               # Both triggers held this long -> MENU_SELECT.
HOLD_SEC = 1.0                  # Left-stick hold to lock the height plane.
EPISODE_DROP_HOLD_S = 2.0       # A held this long drops instead of toggles.


class Quest:
    """The controllers as one object: the latest frame and its button events.

    Same role as bbapps/quest_teleop/quest.py, but the frame arrives as a
    LiveKit packet rather than on the quest.controllers topic, so feed()
    replaces poll(). teleop.py decides what each event does.
    """

    def __init__(self, haptic_q=None, pack=None):
        """Start with no frame, the normal menu, and no button history.

        haptic_q/pack are session.py's outbound queue and packer; without them
        buzzes are dropped and everything else still works.
        """
        self._haptic_q = haptic_q
        self._pack = pack
        self._buzzes = []
        self._next_purr_t = 0.0
        self._purr_toggle = 0       # Alternates hands if both grips held.
        self._decoder = QuestDecoder()
        self.state = None
        self.menu = MENU_NORMAL
        self.prev_lx = False
        self.prev_ly = False
        self.prev_rb = False
        self.prev_ra = False
        self.a_press_t = 0.0
        self.a_consumed = False
        self.b_consumed = False
        self.combo_start = None
        self.stick_hold_start = None
        self.stick_hold_captured = False

    def feed(self, data):
        """Decode a packet and latch it. False if stale or malformed."""
        st = self._decoder.decode(data)
        if st is None:
            return False
        self.state = st
        return True

    def events(self, no_headset):
        """This frame's presses, in evaluation order: the menu resolves before
        B and X read it, so a same-frame arm-and-press lands on the right one.
        """
        st = self.state
        out = []

        # Left Y: teleop.py picks descend-to-park vs a second-press e-stop.
        ly = bool(st["left_b"])
        if ly and not self.prev_ly:
            out.append("dehome")
        self.prev_ly = ly

        # The anchor owns the height in no-headset mode, so the stick is dead.
        if bool(st["left_thumbstick_click"]) and not no_headset:
            if self.stick_hold_start is None:
                self.stick_hold_start = time.monotonic()
                self.stick_hold_captured = False
            if (not self.stick_hold_captured
                    and time.monotonic() - self.stick_hold_start >= HOLD_SEC):
                self.stick_hold_captured = True
                out.append("height_lock")
        else:
            self.stick_hold_start = None
            self.stick_hold_captured = False

        self._update_menu(float(st["left_trigger"]),
                          float(st["right_trigger"]))

        # B selects on the press, homes on the release; a press that selected
        # is consumed, so its release cannot also home.
        rb = bool(st["right_b"])
        if rb and not self.prev_rb:
            self.b_consumed = False                 # Fresh B press.
            if self.menu == MENU_SELECT:
                self.b_consumed = True
                out.append(MENU_EVENTS[MENU_SELECT]["b"])
        if self.prev_rb and not rb:                 # B released.
            if not self.b_consumed:
                out.append(MENU_EVENTS[MENU_NORMAL]["b"])
            self.b_consumed = False
        self.prev_rb = rb

        # Right A: tap toggles an episode, a 2 s hold drops it. The hold fires
        # once and consumes the press, so the release cannot also toggle.
        ra = bool(st["right_a"])
        if ra and not self.prev_ra:
            self.a_press_t = time.monotonic()
            self.a_consumed = False
        if (ra and not self.a_consumed
                and time.monotonic() - self.a_press_t >= EPISODE_DROP_HOLD_S):
            self.a_consumed = True
            out.append("episode_drop")
        if self.prev_ra and not ra and not self.a_consumed:
            out.append("episode_toggle")
        self.prev_ra = ra

        # Left X: whichever the current menu maps it to.
        lx = bool(st["left_a"])
        if lx and not self.prev_lx:
            out.append(MENU_EVENTS[self.menu]["x"])
        self.prev_lx = lx
        return out

    def buzz(self, pattern, delay=0.0):
        """Queue a pulse ``delay`` seconds out, so a pattern keeps its gaps."""
        self._buzzes.append((time.monotonic() + delay, pattern))

    def _emit(self, hand, freq, amp, dur):
        """One haptic packet out to session.py's queue."""
        try:
            self._haptic_q.put_nowait(self._pack(hand, freq, amp, dur))
        except queue.Full:
            pass

    def _purr_amp(self, drift):
        """Purr amplitude for a hand's drift (m); smoothstep to the cap."""
        d = min(drift / PURR_FULL_DRIFT, 1.0)
        return (PURR_AMP_MIN
                + (PURR_AMP_MAX - PURR_AMP_MIN) * d * d * (3.0 - 2.0 * d))

    def send_haptics(self, teleop_active=False, drift_left=0.0,
                     drift_right=0.0):
        """One command per tick: a due buzz, else the purr, scaled by each
        hand's drift (m). The app coalesces same-tick writes, so sending more
        would merge them into one."""
        if self._haptic_q is None:
            return
        now = time.monotonic()
        if self._buzzes:
            self._buzzes.sort(key=lambda b: b[0])
            if self._buzzes[0][0] <= now:
                _, p = self._buzzes.pop(0)
                self._emit(HAPTIC_BOTH, p["freq"], p["amp"], p["dur"])
                return
        if not teleop_active or self.state is None or now < self._next_purr_t:
            return
        cand = []
        if float(self.state["left_squeeze"]) > PURR_THRESH:
            cand.append((HAPTIC_LEFT, self._purr_amp(drift_left)))
        if float(self.state["right_squeeze"]) > PURR_THRESH:
            cand.append((HAPTIC_RIGHT, self._purr_amp(drift_right)))
        if cand:
            hand, amp = cand[self._purr_toggle % len(cand)]
            self._purr_toggle += 1
            self._emit(hand, PURR_FREQ, amp, PURR_DUR)
            self._next_purr_t = now + PURR_INTERVAL

    def _update_menu(self, lt, rt):
        """Both triggers held TOGGLE_HOLD s arm MENU_SELECT; a release
        drops it."""
        if lt > TRIG_TH and rt > TRIG_TH:
            if self.combo_start is None:
                self.combo_start = time.monotonic()
            if (self.menu == MENU_NORMAL
                    and time.monotonic() - self.combo_start >= TOGGLE_HOLD):
                self.menu = MENU_SELECT
                self.buzz(HAPTIC_LONG)
                print("  [quest] MODE SELECT - B = slow, X = no-headset",
                      flush=True)
        else:
            self.combo_start = None
            self.menu = MENU_NORMAL
