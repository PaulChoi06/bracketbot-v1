"""IK target math for the remote_session daemon.

Turns decoded Quest poses (see quest.py) into IK targets, and clamps them in
slow mode. The dehome/park paths live in homing.py; daemon.py runs the control
loop.
"""

import numpy as np

from bbos import Config

from quat import quat_forward_z, quat_mul, slerp

CFG_Q = Config("quest")


# ============================================================================
# Constants
# ============================================================================
INTERP_DURATION = 2.5           # s to ramp from current joints to the target.
HANDOFF_OFFSET = 0.08           # m forward along the gripper axis, to cross.
ROBOT_REF = float(CFG_Q.robot_shoulder_height)   # Head plane maps onto this.

POSE_JUMP_MAX_M = 1.0           # A bigger one-tick move is a dropped controller.

# Min |XY| of the gripper axis at capture (~70 deg off vertical); below
# that its bearing is noise.
ANCHOR_MIN_LEVEL = 0.34

# --- Precision mode ---------------------------------------------------------
# Hold a grip to scale that hand down for fine work. Same numbers as
# quest_teleop; see precision_step().
PRECISION_SCALE = 0.3           # Goal motion per unit of hand motion.
PRECISION_SNAP_TAU = 0.35       # s, glide back to 1:1 on release.

# Gripper, position control; compliance/torque mode deferred.
GRIPPER_OPEN_POS = 0.8
GRIPPER_OPEN_POS_WIDE = 1.2     # Extra-open target held by the left stick.
GRIPPER_CLOSED_POS_L = -0.15
GRIPPER_CLOSED_POS_R = -0.15

# --- Left thumbstick modifier -----------------------------------------------
# The right stick drives, so the left is free as an analog modifier: pushed
# left or right it widens that side's open end, in proportion to the push
# past the deadzone.
STICK_MOD_DEADZONE = 0.35

# --- Slow mode --------------------------------------------------------------
# Hold both triggers + right B for TOGGLE_HOLD s. Targets are then clamped
# into an annular shell (half-split per arm) and speed-capped: a LIMIT.
SLOW_MODE_DEFAULT = False
CYL_CX = 0.0                    # Cylinder axis in robot XY, i.e. the base.
CYL_CY = 0.0
CYL_R_INNER = 0.1155
CYL_R_OUTER = 0.3751
CYL_HEIGHT = 1.50
CYL_TRIM_BOTTOM = 0.50          # Trims leave a z band of [0.50, 1.30].
CYL_TRIM_TOP = 0.20
SLOW_MAX_SPEED = 0.15           # End-effector linear cap (m/s).
SLOW_MAX_ANG_SPEED = 90.0       # End-effector angular cap (deg/s).

SNAP_EPS = 1e-9                 # Distances below this are already there.
ANG_EPS = 1e-6                  # Rotations below this are already there.


# ============================================================================
# IK targets
# ============================================================================
def height_remap(pose, ref_h):
    """Head-relative pose -> IK target: keep XY, map ref_h to the shoulder."""
    p = np.asarray(pose[:3], dtype=np.float64).copy()
    p[2] = p[2] - ref_h + ROBOT_REF
    return p


def home_ik(cfg_l, cfg_r):
    """Reset both solvers' warm start to home (URDF radians, first 7)."""
    cfg_l.ik.reset(list(cfg_l.q2urdf(
        np.asarray(cfg_l.home, dtype=np.float64))[:7]))
    cfg_r.ik.reset(list(cfg_r.q2urdf(
        np.asarray(cfg_r.home, dtype=np.float64))[:7]))


def stick_frac(v, deadzone=STICK_MOD_DEADZONE):
    """One thumbstick axis as a 0..1 fraction past ``deadzone`` (negatives -> 0)."""
    v = float(v)
    if v <= deadzone:
        return 0.0
    return min((v - deadzone) / (1.0 - deadzone), 1.0)


def gripper_open_pos(left_thumbstick):
    """(left, right) open ends, GRIPPER_OPEN_POS .. GRIPPER_OPEN_POS_WIDE."""
    x = float(left_thumbstick[0])          # Raw x is +right.
    span = GRIPPER_OPEN_POS_WIDE - GRIPPER_OPEN_POS
    wide = GRIPPER_OPEN_POS + stick_frac(abs(x)) * span
    return (wide, GRIPPER_OPEN_POS) if x < 0 else (GRIPPER_OPEN_POS, wide)


def gripper_command(cfg, q, trigger, closed_pos, engaged,
                    open_pos=GRIPPER_OPEN_POS):
    """Position-mode gripper: trigger 0 -> open, 1 -> closed.

    engaged=False forces it open. ``open_pos`` is the trigger-released end,
    widened by the left stick. urdf2q() applies gripper_sign downstream, so
    do not apply it here.
    """
    gidx = cfg.dof - 1
    trig = trigger if engaged else 0.0
    q[gidx] = open_pos + trig * (closed_pos - open_pos)


# ============================================================================
# Slow-mode target shaping
# ============================================================================


def cyl_bounds():
    """(cx, cy, r_inner, r_outer, z_min, z_max) for the slow-mode shell."""
    z_min = CYL_TRIM_BOTTOM
    z_max = max(CYL_HEIGHT - CYL_TRIM_TOP, z_min)
    return (CYL_CX, CYL_CY, CYL_R_INNER, CYL_R_OUTER, z_min, z_max)


def constrain_to_cylinders(pos, side=None):
    """Clamp a target into the annular shell: r -> [r_in, r_out], z -> band.

    A `side` of 'left'/'right' restricts to that half of the ring, split by
    the forward axis, so the two arms cannot cross.
    """
    cx, cy, r_in, r_out, z_min, z_max = cyl_bounds()
    dx, dy = float(pos[0]) - cx, float(pos[1]) - cy
    if side == "left":
        dy = max(dy, 0.0)
    elif side == "right":
        dy = min(dy, 0.0)
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    r_c = min(max(r, r_in), r_out)
    return np.array([cx + r_c * np.cos(theta), cy + r_c * np.sin(theta),
                     min(max(float(pos[2]), z_min), z_max)])


def rate_limit(cur, target, max_step):
    """Step cur toward target by at most max_step; snap when within reach."""
    target = np.asarray(target, dtype=np.float64)
    if cur is None:
        return target.copy()
    delta = target - cur
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < SNAP_EPS:
        return target.copy()
    return cur + delta * (max_step / dist)


def slerp_limit(cur, target, max_angle):
    """Rotate cur toward target (xyzw) by at most max_angle rad."""
    target = np.asarray(target, dtype=np.float64)
    target = target / np.linalg.norm(target)
    if cur is None:
        return target.copy()
    cur = cur / np.linalg.norm(cur)
    d = float(np.dot(cur, target))
    if d < 0.0:
        target, d = -target, -d
    full_rot = 2.0 * np.arccos(min(max(d, -1.0), 1.0))
    if full_rot <= max_angle or full_rot < ANG_EPS:
        return target.copy()
    return slerp(cur, target, max_angle / full_rot)


# ============================================================================
# No-headset anchor
# ============================================================================
# The decoder re-centres poses on the head, which is useless with the headset
# on a table. to_global undoes that; the anchor re-centres on the arms instead.


def ground_frame(T_head):
    """The daemon's head ground frame: yaw from the head's forward axis, head XY, z = 0."""
    T_head = np.asarray(T_head, dtype=np.float64)
    f = T_head[:2, 2]
    n = float(np.linalg.norm(f))
    f = f / n if n > 1e-6 else np.array([0.0, 1.0])
    T = np.eye(4)
    T[0, 0], T[1, 0], T[0, 1], T[1, 1] = f[0], f[1], -f[1], f[0]
    T[:2, 3] = T_head[:2, 3]
    return T


def to_global(pose, T_head):
    """Head-recentred pose -> global frame. T_head ships in the same buffer write as the
    poses, so head jitter cancels exactly."""
    T = ground_frame(T_head)
    pos = T[:3, :3] @ np.asarray(pose[:3], dtype=np.float64) + T[:3, 3]
    half = 0.5 * np.arctan2(T[1, 0], T[0, 0])
    q = quat_mul(np.array([0.0, 0.0, np.sin(half), np.cos(half)]),
                  np.asarray(pose[3:], dtype=np.float64))
    return np.concatenate([pos, q])


def ee_anchor(cfg, q_urdf):
    """EE for a pose in URDF radians, HANDOFF_OFFSET backed out: the loop re-adds it to every
    goal, so anchoring on the raw EE would walk the arms forward 8 cm per engage."""
    pos, quat = cfg.ik.fk(list(np.asarray(q_urdf, dtype=np.float64)[:7]))
    return (np.asarray(pos, dtype=np.float64)
            - HANDOFF_OFFSET * quat_forward_z(np.asarray(quat, dtype=np.float64)))


def ground_bearing(quat):
    """Ground bearing (rad) of an xyzw quaternion's gripper (body-Z) axis."""
    f = quat_forward_z(quat)
    return float(np.arctan2(f[1], f[0]))


def capture_anchor(g_left, g_right, ee_left, ee_right):
    """Freeze the user's frame: yaw = circular mean of both ground bearings, each hand pinned
    to its EE. None if a gripper axis is too near vertical, where its bearing is noise."""
    fl, fr = quat_forward_z(g_left[3:]), quat_forward_z(g_right[3:])
    if min(float(np.hypot(fl[0], fl[1])), float(np.hypot(fr[0], fr[1]))) < ANCHOR_MIN_LEVEL:
        return None
    yl, yr = ground_bearing(g_left[3:]), ground_bearing(g_right[3:])
    return {
        "yaw": float(np.arctan2(np.sin(yl) + np.sin(yr), np.cos(yl) + np.cos(yr))),
        "p0_L": np.asarray(g_left[:3], dtype=np.float64).copy(),
        "p0_R": np.asarray(g_right[:3], dtype=np.float64).copy(),
        "ee_L": np.asarray(ee_left, dtype=np.float64),
        "ee_R": np.asarray(ee_right, dtype=np.float64),
        "z": 0.5 * (float(g_left[2]) + float(g_right[2])),
    }


def anchor_apply(anchor, g, side):
    """Global pose -> goal: rotate the motion since the anchor out of the user's yaw, add it
    to the anchored EE."""
    c, s = np.cos(anchor["yaw"]), np.sin(anchor["yaw"])
    d = np.asarray(g[:3], dtype=np.float64) - anchor["p0_" + side]
    d = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])   # rotate by -yaw
    q_inv = np.array([0.0, 0.0, -np.sin(0.5 * anchor["yaw"]), np.cos(0.5 * anchor["yaw"])])
    return anchor["ee_" + side] + d, quat_mul(q_inv, np.asarray(g[3:], dtype=np.float64))


# ============================================================================
# Disconnect guard
# ============================================================================
def new_pose_gate():
    return {"good": None, "holding": False}


def pose_gate(st, pose, engaged, label):
    """Hold the last good pose when a sample jumps or goes non-finite. Only
    while ENGAGED; disengaged, every finite sample passes and refreshes the
    baseline, so the pre-tracking placeholder never latches."""
    if np.all(np.isfinite(pose)) and (not engaged or st["good"] is None
            or np.linalg.norm(pose[:3] - st["good"][:3]) < POSE_JUMP_MAX_M):
        if st["holding"]:
            print(f"[pose] {label} controller recovered - tracking resumes.",
                  flush=True)
        # copy: never alias the reader buffer
        st["good"], st["holding"] = pose.copy(), False
        return pose
    if st["good"] is None:
        return pose                       # nothing good yet
    if not st["holding"]:
        st["holding"] = True
        print(f"[pose] {label} controller jumped >{POSE_JUMP_MAX_M:.0f} m in "
              f"one tick (disconnect?) - holding last good pose.", flush=True)
    return st["good"]


# ============================================================================
# Precision clutch
# ============================================================================
def new_precision_state():
    # off: the target's offset from the raw 1:1 controller position -- the
    # "carried error" precision builds and release eases away.
    return {"off": np.zeros(3), "prev_raw": None, "prev_grip": False}


def precision_reset(st):
    st["off"][:] = 0.0; st["prev_raw"] = None; st["prev_grip"] = False


def precision_step(st, raw, grip_held, scale, ease):
    """Advance the per-hand clutch and return the goal position.

    Holding the grip moves the goal `scale` x the hand and builds an offset;
    releasing eases that offset to zero, so no error is ever carried outside
    precision mode.
    """
    if grip_held:
        if st["prev_grip"]:
            st["off"] += (scale - 1.0) * (raw - st["prev_raw"])
        st["prev_raw"] = raw.copy()
    else:
        st["off"] *= (1.0 - ease)        # glide back to zero-error 1:1
        if np.linalg.norm(st["off"]) < 0.002:
            st["off"][:] = 0.0
    st["prev_grip"] = grip_held
    return raw + st["off"]
