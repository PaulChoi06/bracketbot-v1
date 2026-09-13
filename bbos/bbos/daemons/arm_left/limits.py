import time
import json
import os
import numpy as np
from driver import write_motors, read_motors


def _log(msg):
    """Print a limit/safety event prefixed with a wall-clock timestamp."""
    t = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))}.{int(t % 1 * 1000):03d}] {msg}", flush=True)


def check_current(port, packet, CFG, abs_cs, now, torque_enable, torque_was_enabled,
                  current_limit_cooldown, sustained_current_time, sustained_threshold):
    # Trip: disable torque only on joints exceeding limit
    if np.any(abs_cs > CFG.hard_current_limit):
        tripped = np.where((abs_cs > CFG.hard_current_limit) & (current_limit_cooldown == 0))[0]
        if len(tripped) > 0:
            _log(f"HARD CURRENT LIMIT: motors {tripped.tolist()} {abs_cs[tripped].round(2).tolist()}A")
            tripped_mask = np.zeros(len(CFG.motors), dtype=np.bool_)
            tripped_mask[tripped] = True
            torque_enable[tripped] = False
            torque_was_enabled[tripped] = False
            write_motors(port, packet, "Torque_Enable", torque_enable.astype(np.uint8))
            current_limit_cooldown[tripped] = now + CFG.current_limit_cooldown_s
    # Sustained current: trip if above sustained_threshold for too long
    above_sustained = abs_cs > sustained_threshold
    sustained_current_time[above_sustained] += CFG.dt
    sustained_current_time[~above_sustained] = 0.0
    sustained_tripped = np.where((sustained_current_time > CFG.sustained_current_time_limit) & torque_enable & (current_limit_cooldown == 0))[0]
    if len(sustained_tripped) > 0:
        _log(f"SUSTAINED CURRENT LIMIT: motors {sustained_tripped.tolist()} {abs_cs[sustained_tripped].round(2).tolist()}A for >{CFG.sustained_current_time_limit}s")
        torque_enable[sustained_tripped] = False
        torque_was_enabled[sustained_tripped] = False
        write_motors(port, packet, "Torque_Enable", torque_enable.astype(np.uint8))
        current_limit_cooldown[sustained_tripped] = now + CFG.current_limit_cooldown_s
        sustained_current_time[sustained_tripped] = 0.0


def service_recovery(port, packet, CFG, now, ps_raw, ps_acc, torque_enable, torque_was_enabled,
                     pos_filtered, last_ctrl_pos, current_limit_cooldown,
                     current_limit_recovery_start, current_limit_recovery_start_pos):
    # Recovery: re-enable after cooldown, interpolate to last commanded pos over 3s
    cooled = np.where((current_limit_cooldown > 0) & (now >= current_limit_cooldown))[0]
    if len(cooled) > 0:
        _log(f"COOLDOWN DONE: re-enabling motors {cooled.tolist()}")
        read_motors(port, packet, "Present_Position", ps_raw)
        cooled_mask = np.zeros(len(CFG.motors), dtype=np.bool_)
        cooled_mask[cooled] = True
        _goal = ps_raw.copy()
        write_motors(port, packet, "Goal_Position", _goal, mask=cooled_mask)
        torque_enable[cooled] = True
        torque_was_enabled[cooled] = True
        write_motors(port, packet, "Torque_Enable", torque_enable.astype(np.uint8))
        pos_filtered[cooled] = ps_acc[cooled]
        current_limit_recovery_start_pos[cooled] = ps_acc[cooled]
        current_limit_recovery_start[cooled] = now
        current_limit_cooldown[cooled] = 0
    # Interpolate recovering joints back to last commanded position
    recovering = np.where(current_limit_recovery_start > 0)[0]
    if len(recovering) > 0:
        alpha = np.clip((now - current_limit_recovery_start[recovering]) / CFG.current_limit_interp_s, 0.0, 1.0)
        pos_filtered[recovering] = (1.0 - alpha) * current_limit_recovery_start_pos[recovering] + alpha * last_ctrl_pos[recovering]
        current_limit_recovery_start[recovering[alpha >= 1.0]] = 0


def check_temp(port, packet, CFG, temp_filtered, now, torque_enable, torque_was_enabled,
               current_limit_cooldown, sustained_temp_time):
    # Software temperature trip — instant at software_temp_limit
    overtemp = np.where((temp_filtered > CFG.software_temp_limit) & torque_enable & (current_limit_cooldown == 0))[0]
    if len(overtemp) > 0:
        _log(f"SOFTWARE OVERTEMP: motors {overtemp.tolist()} temps {temp_filtered[overtemp].round(1).tolist()}°C")
        torque_enable[overtemp] = False
        torque_was_enabled[overtemp] = False
        write_motors(port, packet, "Torque_Enable", torque_enable.astype(np.uint8))
        current_limit_cooldown[overtemp] = now + CFG.overtemp_cooldown_s
        sustained_temp_time[overtemp] = 0.0
    # Sustained temperature trip — lower threshold, longer window
    above_sustained_temp = temp_filtered > CFG.sustained_temp_limit
    sustained_temp_time[above_sustained_temp] += CFG.dt
    sustained_temp_time[~above_sustained_temp] = 0.0
    sustained_temp_tripped = np.where((sustained_temp_time > CFG.sustained_temp_time_limit) & torque_enable & (current_limit_cooldown == 0))[0]
    if len(sustained_temp_tripped) > 0:
        _log(f"SUSTAINED OVERTEMP: motors {sustained_temp_tripped.tolist()} temps {temp_filtered[sustained_temp_tripped].round(1).tolist()}°C for >{CFG.sustained_temp_time_limit}s")
        torque_enable[sustained_temp_tripped] = False
        torque_was_enabled[sustained_temp_tripped] = False
        write_motors(port, packet, "Torque_Enable", torque_enable.astype(np.uint8))
        current_limit_cooldown[sustained_temp_tripped] = now + CFG.overtemp_cooldown_s
        sustained_temp_time[sustained_temp_tripped] = 0.0


def load_joint_range(dof, path=None):
    """Per-joint software range limits (lo, hi) in MOTOR TURNS, zero-relative —
    the SAME frame as arm_state.pos / arm_ctrl.pos — from ranges.calibration.json.

    The file's cal_min/cal_max are NAMED extremes (ccw/left/up vs cw/right/down),
    NOT numerically ordered (J0 and several wrist joints are inverted), so we
    return lo=min, hi=max per joint. The gripper (last joint) is never
    range-limited: it seats grasps past its empty-closed stop and often runs in
    tau/compliance mode -> ±inf. Returns (None, None) when the arm is uncalibrated
    so the caller skips clamping (never silently freeze an uncalibrated arm)."""
    if path is None:
        path = os.path.join(os.getcwd(), "ranges.calibration.json")
    if not os.path.exists(path):
        _log(f"[RANGE] {path} not found -> joint-range clamp DISABLED (uncalibrated)")
        return None, None
    # Any problem with the file -> disable the clamp with a loud log rather than
    # crash-loop the daemon or (worse) let a bad value through. A NaN/inf bound
    # would make np.clip emit a NaN motor command that slips past every check
    # below, so non-finite bounds are treated as a corrupt calibration.
    try:
        with open(path) as f:
            cal = json.load(f)
        cal_min = np.asarray(cal["cal_min"], dtype=np.float32)
        cal_max = np.asarray(cal["cal_max"], dtype=np.float32)
        if cal_min.shape != (dof,) or cal_max.shape != (dof,):
            raise ValueError(f"expected {dof} entries, got {cal_min.shape}/{cal_max.shape}")
        lo = np.minimum(cal_min, cal_max)
        hi = np.maximum(cal_min, cal_max)
        if not (np.all(np.isfinite(lo[:dof - 1])) and np.all(np.isfinite(hi[:dof - 1]))):
            raise ValueError(f"non-finite bound(s): lo={lo.tolist()} hi={hi.tolist()}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        _log(f"[RANGE] failed to load {path} ({e}) -> joint-range clamp DISABLED")
        return None, None
    lo[dof - 1] = -np.inf   # gripper: no range clamp
    hi[dof - 1] = np.inf
    _log(f"[RANGE] clamp ENABLED lo={np.round(lo, 4).tolist()} hi={np.round(hi, 4).tolist()}")
    return lo, hi


def clip_range(target_pos, lo, hi):
    """Clamp each joint of the commanded position to its calibrated mechanical
    range. Per-joint: a saturated joint holds at its limit while the others keep
    tracking, so the arm degrades gracefully at a limit instead of driving a joint
    into a hardstop. No-op where lo/hi are ±inf (the gripper)."""
    clamped = np.clip(target_pos, lo, hi)
    hit = np.abs(clamped - target_pos) > 1e-6
    if np.any(hit):
        for j in np.where(hit)[0]:
            _log(f"[RANGE] j{j} cmd={target_pos[j]:.4f} clamped_to={clamped[j]:.4f} "
                 f"range=[{lo[j]:.4f},{hi[j]:.4f}]")
    return clamped


def warn_waypoints_out_of_range(name, waypoints, lo, hi):
    """Log any (dof,) waypoint outside [lo, hi] — a homing/startup pose the range
    clamp would silently truncate. Called once at startup so a too-tight cal that
    would break homing is visible instead of mysterious."""
    for wp in np.atleast_2d(np.asarray(waypoints, dtype=np.float32)):
        out = (wp < lo) | (wp > hi)
        for j in np.where(out)[0]:
            _log(f"[RANGE] WARNING {name} waypoint j{j}={wp[j]:.4f} outside "
                 f"[{lo[j]:.4f},{hi[j]:.4f}] -> homing will be clamped here")


def clip_target(raw_ctrl, ps_acc):
    target_pos = np.clip(raw_ctrl, ps_acc - 0.5, ps_acc + 0.5)
    clipped = np.abs(raw_ctrl - target_pos) > 1e-6
    if np.any(clipped):
        for j in range(len(raw_ctrl)):
            if clipped[j]:
                _log(f"[CLIP] j{j} cmd={raw_ctrl[j]:.4f} cur={ps_acc[j]:.4f} clipped_to={target_pos[j]:.4f} delta={raw_ctrl[j]-ps_acc[j]:.4f}")
    return target_pos
