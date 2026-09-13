"""quest daemon: decode headset input into quest.controllers.

Two input paths feed the same shared state. The native bb-quest-app sends
140-byte OpenXR state packets over UDP; the webapp this daemon hosts (see
webxr.py) sends raw WebXR data over a WebSocket from the headset browser.
Either way the daemon builds the head and controller frames and publishes
quest.controllers, quest.link and haptics back to the headset.

PROVENANCE: reference copy, NOT the running file. The live daemon is
uncommitted local modifications on top of bbos `develop`. Wire-format changes
must stay in sync across the README, app/scripts/protocol.gd,
robot/quest_receiver.py and this file.
"""

import socket
import struct
import subprocess
import threading
import time
from multiprocessing import Array

import numpy as np

import webxr
from bbos import Writer, Reader, Config, Type

CFG = Config("quest")


# ============================================================================
# Constants
# ============================================================================
# --- Wire format (must match app/scripts/protocol.gd) -----------------------
STATE_PORT = CFG.port
MAGIC_STATE = b"BBQ1"
# magic, seq, t_usec, head 7f, 2x (11f + buttons).
_STATE_FMT = "<4sIQ7f11fI11fI"
_STATE_SIZE = struct.calcsize(_STATE_FMT)   # 140 bytes.
BIT_AX = 1 << 0                 # A (right) / X (left).
BIT_BY = 1 << 1                 # B (right) / Y (left).
BIT_THUMB_CLICK = 1 << 2
BIT_POSE_VALID = 1 << 8
RECV_MAX = 512                  # Datagram read size.
QUAT_EPS = 1e-9                 # Below this a quaternion norm is unusable.
FWD_EPS = 1e-6                  # Below this the head has no ground heading.
PRESS_TH = 0.5                  # Analog trigger/squeeze -> boolean.

# The Quest broadcasts DISC_PROBE; we reply unicast with our hostname so it
# can list robots. Unicast dodges the headset's multicast-lock problem.
DISC_PROBE = b"BBQDISC"
HOSTNAME = socket.gethostname()
DISC_REPLY = b"BBQROBOT:" + HOSTNAME.encode()

# avahi publishes this, so the headset browser resolves it without knowing
# the robot's (DHCP-assigned) address.
WEBAPP_HOST = f"{HOSTNAME}.local"

# Drop a packet this far behind the last accepted seq (a UDP reorder). A
# bigger jump back means the app restarted, so we resync and accept it.
SEQ_REORDER_WINDOW = 64

# --- Haptics ----------------------------------------------------------------
# Apps command haptics via quest.haptic; the loop forwards them over the SAME
# socket, to the source address of the state packets.
MAGIC_HAPTIC = b"BBQH"
# magic, hand (0=left, 1=right, 2=both), freq, amp, dur.
_HAPTIC_FMT = "<4sBfff"
HAPTIC_FREQ = 160.0
HAPTIC_AMP = 0.6
HAPTIC_DUR = 0.08

# --- Frame transforms -------------------------------------------------------
T_robot_openxr = np.array([
    [0, 0, -1, 0],
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

T_headlocal_from_world = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])

_a = np.radians(CFG.controller_angle_deg)
T_controller_frame = np.array([
    [np.sin(_a), 0, np.cos(_a), 0],
    [0, -1, 0, 0],
    [np.cos(_a), 0, -np.sin(_a), 0],
    [0, 0, 0, 1],
])

const_head_vuer_mat = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 1.5],
    [0, 0, 1, -0.2],
    [0, 0, 0, 1],
])

LOG_FIRST_FRAMES = 5            # Log every frame up to this many, then...
LOG_EVERY = 500                 # ...one in this many.


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
    """Express a controller in the head's ground frame, as xyz + xyzw.

    Yaw comes from the head FORWARD axis (column 2), not column 1: after the
    OpenXR->robot remap that one is UP, so its XY projection is near zero and
    flips sign on pitch/roll noise, corrupting the recenter.
    """
    fwd_head_xy = T_head[:2, 2]
    norm = np.linalg.norm(fwd_head_xy)
    if norm > FWD_EPS:
        fwd_head_xy = fwd_head_xy / norm
    else:
        fwd_head_xy = np.array([0.0, 1.0])
    T_ground = np.eye(4)
    T_ground[0, 0] = fwd_head_xy[0]
    T_ground[1, 0] = fwd_head_xy[1]
    T_ground[0, 1] = -fwd_head_xy[1]
    T_ground[1, 1] = fwd_head_xy[0]
    T_ground[:2, 3] = T_head[:2, 3]
    T_world = T_head @ T_controller
    T_ground_controller = fast_mat_inv(T_ground) @ T_world
    pos = T_ground_controller[:3, 3]
    quat = rotation_to_quaternion(T_ground_controller[:3, :3])
    return np.concatenate([pos, quat])


def mat_update(prev_mat, mat):
    """Keep prev_mat when mat is degenerate. Returns (matrix, was_valid)."""
    if np.linalg.det(mat) == 0:
        return prev_mat, False
    else:
        return mat, True


def pose_to_mat(px, py, pz, qx, qy, qz, qw):
    """Position + xyzw quaternion -> 4x4.

    Zeros when the quaternion is null, so det is 0 and mat_update keeps the
    previous matrix.
    """
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
# Shared state
# ============================================================================
# Written by the receiver thread, read by the publish loop.
left_controller_shared = Array('d', 16, lock=True)
right_controller_shared = Array('d', 16, lock=True)
left_controller_state_shared = Array('d', 12, lock=True)
right_controller_state_shared = Array('d', 12, lock=True)
head_matrix_shared = Array('d', 16, lock=True)
body_head_matrix_shared = Array('d', 16, lock=True)

const_left_controller = np.eye(4)
const_right_controller = np.eye(4)

quest_socket = None     # The bound UDP socket, set by the receiver thread.
quest_addr = None       # (ip, port) of the Quest, learned from its packets.
last_packet_t = -1e9    # Monotonic time of the last valid state packet.


# ============================================================================
# Packet receive
# ============================================================================
def _store_controller(v, i, mat_shared, state_shared):
    """Unpack one hand from the tuple at i into the shared arrays."""
    px, py, pz = v[i], v[i + 1], v[i + 2]
    qx, qy, qz, qw = v[i + 3], v[i + 4], v[i + 5], v[i + 6]
    sx, sy = v[i + 7], v[i + 8]
    trig, grip, btn = v[i + 9], v[i + 10], v[i + 11]
    if btn & BIT_POSE_VALID:
        mat_shared[:] = pose_to_mat(
            px, py, pz, qx, qy, qz, qw).flatten(order="F")
    else:
        mat_shared[:] = [0.0] * 16  # det 0 -> keep previous matrix.
    state_shared[0] = 1.0 if trig > PRESS_TH else 0.0   # Trigger (bool).
    state_shared[1] = 1.0 if grip > PRESS_TH else 0.0   # Squeeze (bool).
    state_shared[2] = 0.0                           # Touchpad (n/a on Touch).
    state_shared[3] = 1.0 if (btn & BIT_THUMB_CLICK) else 0.0
    state_shared[4] = 1.0 if (btn & BIT_AX) else 0.0    # A / X.
    state_shared[5] = 1.0 if (btn & BIT_BY) else 0.0    # B / Y.
    state_shared[6] = float(trig)                       # triggerValue.
    state_shared[7] = float(grip)                       # squeezeValue.
    state_shared[8] = 0.0                               # touchpadValue x.
    state_shared[9] = 0.0                               # touchpadValue y.
    state_shared[10] = float(sx)                        # thumbstickValue x.
    state_shared[11] = float(sy)                        # thumbstickValue y.


def haptic_packet(hand, freq=HAPTIC_FREQ, amp=HAPTIC_AMP, dur=HAPTIC_DUR):
    """Pack one haptic command for the headset."""
    return struct.pack(_HAPTIC_FMT, MAGIC_HAPTIC, hand, freq, amp, dur)


def run_udp_receiver():
    """Bind the state port and fold every valid packet into shared state."""
    global quest_socket, quest_addr, last_packet_t
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", STATE_PORT))
    quest_socket = sock
    last_seq = -1
    while True:
        try:
            data, addr = sock.recvfrom(RECV_MAX)
        except OSError:
            continue
        if data == DISC_PROBE:
            try:
                sock.sendto(DISC_REPLY, addr)
            except OSError:
                pass
            continue
        if len(data) != _STATE_SIZE:
            continue
        v = struct.unpack(_STATE_FMT, data)
        if v[0] != MAGIC_STATE:
            continue
        seq = v[1]
        if 0 <= last_seq - seq < SEQ_REORDER_WINDOW:
            continue   # Stale or duplicate; a newer one already arrived.
        last_seq = seq
        last_packet_t = time.monotonic()    # Link liveness -> quest.link.
        quest_addr = addr   # Where to send haptics back.
        head_matrix_shared[:] = pose_to_mat(
            v[3], v[4], v[5], v[6], v[7], v[8], v[9]
        ).flatten(order="F")
        _store_controller(v, 10, left_controller_shared,
                          left_controller_state_shared)
        _store_controller(v, 22, right_controller_shared,
                          right_controller_state_shared)


# ============================================================================
# WebXR packet receive
# ============================================================================
def _store_web_controller(hand, mat_shared, state_shared):
    """Store one decoded WebXR hand (see webxr.decode) into shared state."""
    if hand is None:
        mat_shared[:] = [0.0] * 16      # det 0 -> keep previous matrix.
        state_shared[:] = [0.0] * 12    # A hand that left holds nothing down.
        return
    pose, state = hand
    mat_shared[:] = (pose_to_mat(*pose).flatten(order="F") if pose
                     else [0.0] * 16)   # Tracking lost -> keep previous.
    state_shared[:] = state


def store_web_state(msg):
    """Fold one raw WebXR packet from the webapp into shared state."""
    global last_packet_t
    head, left, right = webxr.decode(msg, PRESS_TH)
    if head is not None:
        head_matrix_shared[:] = pose_to_mat(*head).flatten(order="F")
    _store_web_controller(left, left_controller_shared,
                          left_controller_state_shared)
    _store_web_controller(right, right_controller_shared,
                          right_controller_state_shared)
    last_packet_t = time.monotonic()    # Link liveness -> quest.link.


# ============================================================================
# Pose assembly
# ============================================================================
def get_head_pose():
    """Build the head frame in robot coords, falling back if degenerate."""
    head_vuer_mat = np.array(
        body_head_matrix_shared[:]).reshape(4, 4, order="F")
    if np.linalg.det(head_vuer_mat) == 0:
        head_vuer_mat = np.array(
            head_matrix_shared[:]).reshape(4, 4, order="F")
    if np.linalg.det(head_vuer_mat) == 0:
        head_vuer_mat = const_head_vuer_mat
    head_world = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
    head_mat = head_world @ fast_mat_inv(T_headlocal_from_world)
    return head_mat


def get_controller_data():
    """Head-local controller frames plus both raw state arrays."""
    global const_left_controller, const_right_controller
    left_controller_vuer = np.array(
        left_controller_shared[:]).reshape(4, 4, order="F")
    right_controller_vuer = np.array(
        right_controller_shared[:]).reshape(4, 4, order="F")

    head_vuer_mat = np.array(
        body_head_matrix_shared[:]).reshape(4, 4, order="F")
    if np.linalg.det(head_vuer_mat) == 0:
        head_vuer_mat = np.array(
            head_matrix_shared[:]).reshape(4, 4, order="F")
    if np.linalg.det(head_vuer_mat) == 0:
        head_vuer_mat = const_head_vuer_mat

    left_controller_vuer_mat, left_flag = mat_update(
        const_left_controller, left_controller_vuer)
    right_controller_vuer_mat, right_flag = mat_update(
        const_right_controller, right_controller_vuer)
    if left_flag:
        const_left_controller = left_controller_vuer_mat
    if right_flag:
        const_right_controller = right_controller_vuer_mat

    head_world = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
    left_world = (T_robot_openxr @ left_controller_vuer_mat
                  @ fast_mat_inv(T_robot_openxr))
    right_world = (T_robot_openxr @ right_controller_vuer_mat
                   @ fast_mat_inv(T_robot_openxr))

    left_rel = fast_mat_inv(head_world) @ left_world
    right_rel = fast_mat_inv(head_world) @ right_world

    left_headlocal = T_headlocal_from_world @ left_rel @ T_controller_frame
    right_headlocal = T_headlocal_from_world @ right_rel @ T_controller_frame

    left_state = np.array(left_controller_state_shared[:])
    right_state = np.array(right_controller_state_shared[:])

    return left_headlocal, right_headlocal, left_state, right_state


def get_ip_address():
    """Our LAN address, for the "point the app here" startup line."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


# ============================================================================
# Daemon
# ============================================================================
def main():
    """Receive on UDP, publish quest.controllers and quest.link forever."""
    threading.Thread(target=run_udp_receiver, daemon=True).start()
    webxr.start(CFG.web_port, CFG.cert_file, CFG.key_file, store_web_state)
    ip = get_ip_address()
    print(f"[quest] UDP receiver listening on :{STATE_PORT}", flush=True)
    print(f"[quest] Set the bb-quest-app robot_ip to {ip}", flush=True)
    print(f"[quest] WebXR webapp on https://{WEBAPP_HOST}:{CFG.web_port}",
          flush=True)

    # quest.link is @state, so it never paces this 20 ms loop (keeptime is
    # forced off for period-less types); the kwarg just documents the intent.
    with Writer("quest.controllers", Type("quest_controllers")) as w, \
         Writer("quest.link", Type("quest_link"), keeptime=False) as w_link, \
         Reader("quest.haptic", Type("quest_haptic")) as r_hap:
        frame = 0
        link_up = None          # Last published value.
        link_published_t = 0.0
        while True:
            # ---- headset link -> quest.link ----
            now = time.monotonic()
            up = (now - last_packet_t) <= CFG.link_timeout_s
            if up != link_up or now - link_published_t >= CFG.link_publish_s:
                with w_link.buf() as lb:
                    lb["connected"] = np.uint8(1 if up else 0)
                if up != link_up:
                    print(f"[quest] link {'up' if up else 'down'}", flush=True)
                link_up, link_published_t = up, now

            # Forward any haptic command from an app to the headset, over
            # whichever transport it arrived on (both, if both are up).
            try:
                if r_hap.ready():
                    h = r_hap.data
                    hand, freq = int(h["hand"]), float(h["frequency"])
                    amp, dur = float(h["amplitude"]), float(h["duration"])
                    if quest_addr is not None and quest_socket is not None:
                        quest_socket.sendto(
                            haptic_packet(hand, freq, amp, dur), quest_addr)
                    webxr.send_haptic(hand, freq, amp, dur)
            except Exception:
                pass

            T_left, T_right, left_state, right_state = get_controller_data()
            T_head = get_head_pose()
            left_pose = controller_rel_ground(T_head, T_left)
            right_pose = controller_rel_ground(T_head, T_right)

            # Emergency restart: both triggers + squeezes + B buttons.
            both_triggers = left_state[0] and right_state[0]
            both_squeezes = left_state[1] and right_state[1]
            both_b = left_state[5] and right_state[5]
            if both_triggers and both_squeezes and both_b:
                print("[quest] RESTART triggered "
                      "(both triggers + squeezes + B)", flush=True)
                subprocess.Popen(["restart"], start_new_session=True)

            with w.buf() as b:
                b["T_head"][:] = T_head
                b["left_pose"][:] = left_pose
                b["right_pose"][:] = right_pose
                b["left_trigger"] = left_state[6]
                b["left_squeeze"] = left_state[7]
                b["left_thumbstick"][:] = [left_state[10], left_state[11]]
                b["left_thumbstick_click"] = left_state[3]
                b["left_a"] = left_state[4]
                b["left_b"] = left_state[5]
                b["right_trigger"] = right_state[6]
                b["right_squeeze"] = right_state[7]
                b["right_thumbstick"][:] = [right_state[10], right_state[11]]
                b["right_thumbstick_click"] = right_state[3]
                b["right_a"] = right_state[4]
                b["right_b"] = right_state[5]

            frame += 1
            if frame <= LOG_FIRST_FRAMES or frame % LOG_EVERY == 0:
                print(f"[quest] Frame {frame}: "
                      f"L={left_pose[:3]} R={right_pose[:3]}", flush=True)


if __name__ == "__main__":
    main()
