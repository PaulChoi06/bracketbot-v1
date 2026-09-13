# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "yourdfpy",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""
Arm calibration (works for arm_left and arm_right via symlink):
1. USB detection and udev setup (ttyARMLEFT/ttyARMRIGHT)
2. Joint zeroing via daemon IPC (daemon must be running, handles j0 homing)
"""
import subprocess
import sys
import re
import os
import time
import glob
import json
import numpy as np

# --- Paths and daemon identity ---
DAEMON_DIR = os.getcwd()  # arm_left/ or arm_right/ (where the symlink lives)
DAEMON_NAME = os.path.basename(DAEMON_DIR)
SIDE = DAEMON_NAME.split("_")[-1].upper()    # "LEFT" or "RIGHT"
SYMLINK_NAME = "tty" + DAEMON_NAME.replace("_", "").upper()
UDEV_RULES_FILE = f"/etc/udev/rules.d/99-{DAEMON_NAME}-udev.rules"
CONSTANTS_FILE = os.path.join(DAEMON_DIR, "constants.py")
ZEROS_FILE = os.path.join(DAEMON_DIR, "zeros.txt")
CAL_RANGES_FILE = os.path.join(DAEMON_DIR, "ranges.calibration.json")

# --- J0 (lift) / J7 (gripper) torque sweeps ---
# J0 limit-find params (j0_cal_tau, j0_top_offset_turns, ...) live in constants.py.
GRIPPER_CAL_TAU = 1.0          # Nm peak torque at the stop (firm close)
GRIPPER_CAL_TAU_START = 0.5     # Nm torque at ramp start (above breakaway so it moves smoothly)
GRIPPER_CAL_RAMP_S = 1.5        # s to ramp torque start -> peak (higher = slower, gentler)
GRIPPER_CAL_VEL_STOP = 0.0001   # turns/s; gripper is "stopped" when |vel| stays below this
CAL_SETTLE_S = 0.37             # s of |vel| under vel_stop before a stall is accepted

# --- Span check (validation) ---
# Expected |cal_max - cal_min| per joint from a known-good cal; a fresh calibration that
# deviates by more than the tolerance is flagged (joint likely missed its extremes).
SPAN_HEURISTIC = [3.53, 0.50, 0.42, 0.51, 0.62, 0.44, 0.45, 0.28]
SPAN_TOLERANCE_PCT = 10.0

# --- Zero derivation (per side) ---
# Zero's fractional position in [cal_min, cal_max]; J1/J2/J3 zero = cal_min + rel*span.
# Wrist J4/J5/J6 is always 0.5; J0/J7 zeros come from homing/closed and are untouched.
ZERO_REL = {
    "LEFT":  {1: 0.437, 2: 0.397, 3: 0.206},
    "RIGHT": {1: 0.405, 2: 0.593, 3: 0.257},
}


def warn_on_span_outliers(cal_ranges):
    """Compare each captured span to SPAN_HEURISTIC; warn on >SPAN_TOLERANCE_PCT
    deviation. A span of exactly 0 means the joint was never swept -> hard ERROR."""
    print("\n--- Span check (vs heuristic) ---")
    dead = []
    for i, expected in enumerate(SPAN_HEURISTIC):
        span = abs(cal_ranges["cal_max"][i] - cal_ranges["cal_min"][i])
        if span == 0.0:
            dead.append(i)
            print(f"  ERROR   J{i}: span=0 (cal_min == cal_max) -> J{i} NOT calibrated")
            continue
        dev = 100 * abs(span - expected) / expected if expected else 0.0
        tag = f"WARNING J{i}" if dev > SPAN_TOLERANCE_PCT else f"  J{i}"
        print(f"  {tag}: span={span:.4f} expected~{expected:.2f} ({dev:.1f}% off)")
    if dead:
        print(f"\n  >>> J{dead} have no range. The arm daemon will refuse to start "
              f"until they are calibrated. <<<")

# ============================================================================
# PHASE 1: USB Detection and udev Setup
# ============================================================================

def get_tty_devices():
    devices = set()
    for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*"]:
        devices.update(glob.glob(pattern))
    return devices

def get_serial(device_path):
    result = subprocess.run(
        ["/usr/bin/udevadm", "info", f"--name={device_path}", "--attribute-walk"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        match = re.search(r'ATTRS\{serial\}=="([^"]+)"', line)
        if match:
            return match.group(1)
    return None

def update_udev_rules(serial):
    rule = (
        f'SUBSYSTEM=="tty", ATTRS{{serial}}=="{serial}", '
        f'ENV{{ID_MM_DEVICE_IGNORE}}="1", ATTR{{device/latency_timer}}="1", '
        f'SYMLINK+="{SYMLINK_NAME}"'
    )
    existing_rules = []
    if os.path.exists(UDEV_RULES_FILE):
        with open(UDEV_RULES_FILE, "r") as f:
            existing_rules = f.readlines()
    new_rules = [r for r in existing_rules if SYMLINK_NAME not in r]
    new_rules.append(rule + "\n")
    print(f"  Rule: {rule}")
    with open("/tmp/udev_rule.tmp", "w") as f:
        f.writelines(new_rules)
    subprocess.run(["/usr/bin/sudo", "cp", "/tmp/udev_rule.tmp", UDEV_RULES_FILE], check=True)
    subprocess.run(["/usr/bin/sudo", "chmod", "644", UDEV_RULES_FILE], check=True)
    subprocess.run(["/usr/bin/sudo", "/usr/bin/udevadm", "control", "--reload"], check=True)
    subprocess.run(["/usr/bin/sudo", "/usr/bin/udevadm", "trigger"], check=True)
    print(f"  Device available at /dev/{SYMLINK_NAME}")

def update_constants_port():
    with open(CONSTANTS_FILE, "r") as f:
        content = f.read()
    new_content = re.sub(
        r'port\s*=\s*"[^"]+"', f'port = "/dev/{SYMLINK_NAME}"', content
    )
    with open(CONSTANTS_FILE, "w") as f:
        f.write(new_content)
    print(f"  Updated {CONSTANTS_FILE} port=/dev/{SYMLINK_NAME}")

def setup_udev():
    print("\n" + "=" * 60)
    print("PHASE 1: USB Device Setup")
    print("=" * 60)

    devices_before = get_tty_devices()
    print(f"Current tty devices: {devices_before}")
    print(f"\n>>> Please UNPLUG the {SIDE.lower()} arm USB cable <<<")
    input("Press ENTER when unplugged...")

    devices_after_unplug = get_tty_devices()
    print(f"Devices after unplug: {devices_after_unplug}")

    print(f"\n>>> Now PLUG IN the {SIDE.lower()} arm USB cable <<<")
    input("Press ENTER when plugged in...")

    new_devices = set()
    for _ in range(50):
        time.sleep(0.2)
        devices_after_plug = get_tty_devices()
        new_devices = devices_after_plug - devices_after_unplug
        if new_devices:
            break

    if not new_devices:
        print("ERROR: No new device detected!")
        sys.exit(1)

    device = sorted(new_devices)[0]
    print(f"  Detected: {device}")

    serial = get_serial(device)
    if not serial:
        print(f"ERROR: Could not get serial for {device}")
        sys.exit(1)
    print(f"  Serial: {serial}")

    update_udev_rules(serial)
    update_constants_port()

    for _ in range(10):
        if os.path.exists(f"/dev/{SYMLINK_NAME}"):
            print(f"  Symlink /dev/{SYMLINK_NAME} ready")
            return True
        time.sleep(0.2)

    print(f"  WARNING: Symlink /dev/{SYMLINK_NAME} not yet available")
    return True

# ============================================================================
# PHASE 2
# Joint zeroing via daemon IPC
# ============================================================================
def require_daemon_running():
    # The daemon owns the serial port, so nothing here can read the motors without its .state writer.
    try:
        with open(f"/dev/shm/{DAEMON_NAME}.state", "rb") as f:
            pid = int.from_bytes(f.read(12)[8:12], "little")
        if pid <= 0:
            raise OSError
        os.kill(pid, 0)
    except OSError:
        print(f"ERROR: {DAEMON_NAME} is not running -- no {DAEMON_NAME}.state writer.")
        if os.path.exists(os.path.join(DAEMON_DIR, ".disabled")):
            print(f"It is disabled: rm {os.path.join(DAEMON_DIR, '.disabled')}")
        print(f"Start it with `restart {DAEMON_NAME}`, then check `logs {DAEMON_NAME}`.")
        sys.exit(1)


def calibrate_joints():
    from bbos import Reader, Writer, Config, Type

    CFG = Config(DAEMON_NAME)
    dof = CFG.dof

    print("\n" + "=" * 60)
    print("PHASE 2: Joint Zeroing (via daemon)")
    print("=" * 60)

    require_daemon_running()

    # zeros.txt is calibration output, not daemon output; seed it to match the daemon's zero fallback.
    if not os.path.exists(ZEROS_FILE):
        np.savetxt(ZEROS_FILE, np.zeros(dof, dtype=np.float32))
        print(f"Created {ZEROS_FILE} (zeros)")
    ps0 = np.loadtxt(ZEROS_FILE, dtype=np.float32)
    print(f"Current zeros: {ps0}")

    with Reader(f"{DAEMON_NAME}.state") as r:
        while not r.ready():
            pass
        init_pos = r.data['pos'].copy()
    print(f"Daemon position: {init_pos}")

    offsets = np.zeros(dof, dtype=np.float32)
    # Seed ranges from the existing file (if any) so a per-joint save preserves
    # joints we don't (re)calibrate this run; each joint we DO calibrate then
    # overwrites its own entry.
    if os.path.exists(CAL_RANGES_FILE):
        with open(CAL_RANGES_FILE) as _f:
            _prev = json.load(_f)
        cal_ranges = {"cal_min": list(_prev["cal_min"]), "cal_max": list(_prev["cal_max"])}
    else:
        cal_ranges = {"cal_min": [0.0] * dof, "cal_max": [0.0] * dof}

    def save_progress(note=""):
        """Persist zeros + ranges to disk NOW so a Ctrl-C keeps every joint finished
        so far. The daemon only reads these files at startup, so writing mid-run is
        safe and takes effect on the next daemon restart."""
        np.savetxt(ZEROS_FILE, ps0)
        with open(CAL_RANGES_FILE, "w") as _f:
            json.dump(cal_ranges, _f, indent=2)
        if note:
            print(note)

    with Writer(f"{DAEMON_NAME}.torque", Type("arm_torque")) as w_torque, \
         Writer(f"{DAEMON_NAME}.ctrl", Type("arm_ctrl")) as w_ctrl, \
         Reader(f"{DAEMON_NAME}.state", sync=True) as r_state:

        # Disable the daemon's gripper torque-limit clamp for the whole run: the gripper
        # sweep drives into its stops at full torque and reads current to find min/max.
        w_torque['calibrating'] = True
        try:

            # Read current position and write it as ctrl BEFORE enabling torque
            while not r_state.ready():
                pass
            pos_cmd = r_state.data['pos'].copy()
            pos = r_state.data['pos'].copy()
            # Flush any stale ctrl command left by a previous (e.g. cancelled) run: hold the
            # live position with torque still OFF for a moment so the daemon adopts the current
            # position before we enable torque. Without this the daemon briefly chases the old
            # goal and the joints (J0 especially) jerk to wherever the last run left off.
            flush_end = time.monotonic() + 0.3
            while time.monotonic() < flush_end:
                if r_state.ready():
                    pos = r_state.data['pos']
                    pos_cmd = pos.copy()
                if w_ctrl.ready():
                    w_ctrl['pos'] = pos_cmd

            # Enable lift + shoulder (J0-J2); J3 and wrist stay limp until calibrated.
            torque_mask = np.zeros(dof, dtype=np.bool_)
            for ji in range(3):
                if r_state.ready():
                    pos = r_state.data['pos']
                    pos_cmd = pos.copy()
                w_ctrl['pos'] = pos_cmd
                torque_mask[ji] = True
                w_torque['enable'] = torque_mask
                time.sleep(0.1)

            # ---------------------------------------------------------------------------
            # Torque-drive helper (used by the J0 lift and J7 gripper sweeps)
            # ---------------------------------------------------------------------------
            def drive_to_stop(joint_idx, signed_tau, vel_stop, ramp_s=0.0, tau_start=None,
                              min_drive_s=0.3, settle_s=CAL_SETTLE_S, max_s=8.0,
                              min_travel=0.0):
                """Drive joint_idx in torque mode at signed_tau until it stops moving (|vel| <
                vel_stop continuously for settle_s seconds), then return its resting position. If
                tau_start is given, ramp the torque from there up to full over ramp_s (start above
                breakaway, approach gently); the stall check waits for full torque so the joint
                still presses firmly. If min_travel > 0, a stall is only accepted after the joint
                has moved at least that far from its start (rejects a false stall when the drive
                can't move it -- e.g. J0 torque too low to lift). Caller sets tau_mode/enable for
                joint_idx; others hold via pos_cmd."""
                full = abs(signed_tau)
                start_frac = (abs(tau_start) / full) if (tau_start is not None and full) else 1.0
                tau_cmd = np.zeros(dof, dtype=np.float32)
                stall_since = None
                t0 = time.monotonic()
                jpos = float(pos_cmd[joint_idx])
                jstart = jpos
                while time.monotonic() - t0 < max_s:
                    now = time.monotonic()
                    elapsed = now - t0
                    ramp = min(1.0, elapsed / ramp_s) if ramp_s > 0 else 1.0
                    tau_cmd[joint_idx] = signed_tau * (start_frac + (1.0 - start_frac) * ramp)
                    if r_state.ready():
                        jpos = float(r_state.data['pos'][joint_idx])
                        # Judge "stalled" only after full torque, a minimum drive time, and (if
                        # required) a minimum travel from the start.
                        if (elapsed >= max(ramp_s, min_drive_s) and abs(jpos - jstart) >= min_travel
                                and abs(float(r_state.data['vel'][joint_idx])) < vel_stop):
                            if stall_since is None:
                                stall_since = now
                            elif now - stall_since >= settle_s:
                                break
                        else:
                            stall_since = None
                    if w_ctrl.ready():
                        w_ctrl['pos'] = pos_cmd
                        w_ctrl['tau'] = tau_cmd
                return jpos

            # ---------------------------------------------------------------------------
            # J0 (lift): constant-torque drive into the top hardstop, stall at zero velocity
            # ---------------------------------------------------------------------------
            # Drive J0 UP under constant torque until it stalls at the top stop (same scheme as the
            # gripper), then ease j0_top_offset_turns below it and hold. The daemon bakes that same
            # back-off into the persisted J0 zero at startup; here it just parks the lift clear of
            # the stop to hold the arm up for the rest of calibration.
            print("\n[J0] Driving to top limit (constant torque)...")
            j0_dir = float(np.sign(CFG.j0_increment))   # raw-position direction for the lift's "up"
            j0_tau_mask = np.zeros(dof, dtype=np.bool_); j0_tau_mask[0] = True
            w_torque['tau_mode'] = j0_tau_mask          # J0 -> torque mode before driving
            w_torque['enable'] = torque_mask            # J0-J2 stay enabled
            # Drive UP with -sign(j0_increment) torque: on these HLS servos positive current
            # DECREASES position, so the current-mode "up" sign is opposite the position increment.
            top_pos = drive_to_stop(0, -j0_dir * CFG.j0_cal_tau, CFG.j0_cal_vel_stop,
                                    ramp_s=CFG.j0_cal_ramp_s, tau_start=CFG.j0_cal_tau_start,
                                    settle_s=float(CFG.j0_cal_settle_s),
                                    max_s=CFG.j0_homing_timeout_s,
                                    min_travel=float(CFG.j0_cal_min_travel_turns))
            print(f"[J0] top limit at {top_pos:.4f}")

            # J0 -> back to position mode; ease off the stop to (top - offset) and hold there.
            w_ctrl['tau'] = np.zeros(dof, dtype=np.float32)
            j0_tau_mask[0] = False
            w_torque['tau_mode'] = j0_tau_mask
            j0_hold = top_pos - j0_dir * float(CFG.j0_top_offset_turns)   # one offset below the top stop
            while not r_state.ready():
                pass
            pos_cmd = r_state.data['pos'].copy()
            print("\nEasing J0 to top - offset...")
            settle_end = time.monotonic() + 2.0
            while time.monotonic() < settle_end:
                if r_state.ready():
                    pos = r_state.data['pos']
                if w_ctrl.ready():
                    diff = j0_hold - pos_cmd[0]
                    pos_cmd[0] += float(np.clip(diff, -CFG.j0_cal_step, CFG.j0_cal_step))
                    w_ctrl['pos'] = pos_cmd   # ramp down off the stop, then hold at top - offset

            # Re-enable J0 hold (J1/J2/J3 zeros are derived later from their swept min/max).
            while not r_state.ready():
                pass

            pos_cmd = r_state.data['pos'].copy()
            w_ctrl['pos'] = pos_cmd
            torque_mask = np.zeros(dof, dtype=np.bool_)
            torque_mask[0] = True
            w_torque['enable'] = torque_mask
            settle_end = time.monotonic() + 0.3

            while time.monotonic() < settle_end:
                if r_state.ready():
                    pos_cmd = r_state.data['pos'].copy()
                if w_ctrl.ready():
                    w_ctrl['pos'] = pos_cmd

            # J0 is calibrated: record its range and save immediately. J0 range comes from
            # the URDF (no sweep; homing lands at the top), and the J0 zero is derived by
            # the daemon at startup homing (ps0[0] untouched here).
            import yourdfpy as _yrdf
            _urdf = _yrdf.URDF.load(CFG.urdf_path, load_meshes=False, build_scene_graph=False)
            _j0_jt = _urdf.joint_map.get(CFG.joint_names[0])
            if _j0_jt is None:
                print(f"ERROR: URDF has no joint {CFG.joint_names[0]} -- cannot derive the J0 range")
                sys.exit(1)
            # Travel is the joint's FULL span: the URDF writes lower=-travel, upper=0
            # (0 = the top), so `upper` alone is 0 and would give a zero-span range.
            _travel_m = abs(float(_j0_jt.limit.upper) - float(_j0_jt.limit.lower))
            _j0_turns = _travel_m / (2 * np.pi * CFG.wheel_radius)
            if not _j0_turns > 0:
                print(f"ERROR: URDF {CFG.joint_names[0]} limit spans no travel "
                      f"({_j0_jt.limit.lower}..{_j0_jt.limit.upper}) -- cannot derive the J0 range")
                sys.exit(1)
            # cal_max = 0 (top, where homing lands); cal_min = bottom (signed).
            cal_ranges["cal_min"][0] = float(-np.sign(CFG.j0_increment) * _j0_turns)
            cal_ranges["cal_max"][0] = 0.0
            j0_min, j0_max = cal_ranges["cal_min"][0], cal_ranges["cal_max"][0]
            print(f"  J0: cal_min={j0_min:.4f} (bottom) cal_max={j0_max:.4f} (top)")
            save_progress("  saved J0 (range; zero set by daemon homing)")

            home = CFG.home.copy()
            home[1] += CFG.j1_cal_clearance  # extra J1 clearance for self-cal only

            # Move J1/J2 to home (J3 stays limp; J0 holds at top - clearance).
            print("\nMoving shoulder to home...")
            shoulder_target = pos_cmd.copy()
            while True:
                if r_state.ready():
                    pos = r_state.data['pos']
                diff = shoulder_target - pos_cmd
                if np.max(np.abs(diff[1:3])) < 0.0005:
                    pos_cmd[1:3] = shoulder_target[1:3]
                    break
                pos_cmd += np.clip(diff, -CFG.cal_step, CFG.cal_step)
                if w_ctrl.ready():
                    w_ctrl['pos'] = pos_cmd

            settle_end = time.monotonic() + 1.0
            while time.monotonic() < settle_end:
                if r_state.ready():
                    pos = r_state.data['pos']
                if w_ctrl.ready():
                    w_ctrl['pos'] = pos_cmd
            print(f"Shoulder/elbow at home: {pos[:4]}")

            # ---------------------------------------------------------------------------
            # J7 (gripper): constant-torque sweep, stop at zero velocity (closed = zero)
            # ---------------------------------------------------------------------------
            # Open first, then close; each extreme is where the gripper stalls.
            input("\nPress ENTER to calibrate the gripper (J7)...")
            gripper_flip = float(CFG.gripper_sign)
            gripper_tau_mask = np.zeros(dof, dtype=np.bool_)
            gripper_tau_mask[7] = True
            w_torque['tau_mode'] = gripper_tau_mask   # J7 -> torque mode before energizing
            torque_mask[7] = True
            w_torque['enable'] = torque_mask

            print("\n[J7] Opening gripper (constant torque)...")
            gripper_open = drive_to_stop(7, -gripper_flip * GRIPPER_CAL_TAU, GRIPPER_CAL_VEL_STOP,
                                         ramp_s=GRIPPER_CAL_RAMP_S, tau_start=GRIPPER_CAL_TAU_START)
            print(f"[J7] open: {gripper_open:.4f}")

            print("\n[J7] Closing gripper (constant torque)...")
            offsets[7] = drive_to_stop(7, gripper_flip * GRIPPER_CAL_TAU, GRIPPER_CAL_VEL_STOP,
                                       ramp_s=GRIPPER_CAL_RAMP_S, tau_start=GRIPPER_CAL_TAU_START)
            print(f"[J7] closed offset: {offsets[7]:.4f}")

            gripper_closed = float(offsets[7])
            gripper_open_distance = gripper_open - gripper_closed
            print(f"[J7] open distance (signed): {gripper_open_distance:.4f}")

            # Back to position mode, gripper off (recorded values already captured).
            w_ctrl['tau'] = np.zeros(dof, dtype=np.float32)
            gripper_tau_mask[7] = False
            w_torque['tau_mode'] = gripper_tau_mask
            torque_mask[7] = False
            w_torque['enable'] = torque_mask

            # J7 is calibrated: range (cal_min=closed=0, cal_max=signed open distance),
            # zero (closed position) -> bake into ps0 and save immediately.
            cal_ranges["cal_min"][7] = 0.0
            cal_ranges["cal_max"][7] = float(gripper_open_distance)
            ps0[7] = (ps0[7] + offsets[7]) % 1.0
            g_min, g_max = cal_ranges["cal_min"][7], cal_ranges["cal_max"][7]
            print(f"  Gripper: closed={g_min:.4f} open={g_max:.4f}")
            save_progress("  saved J7 (zero + range)")

            # ---------------------------------------------------------------------------
            # J6..J1 (wrist + shoulder/elbow): manual hand-sweep to named extremes
            # ---------------------------------------------------------------------------
            # Torque is OFF; operator moves each joint to its named world extreme and presses
            # ENTER. cal_max = the named-positive (up/left/ccw/forward) side. The zero sits at
            # fraction `rel` within [cal_min, cal_max]: wrist (J4/J5/J6) centers at 0.5;
            # shoulder/elbow (J1/J2/J3) use the per-side ZERO_REL fraction.
            import threading
            MANUAL_DIRS = {
                6: ("up", "down"), 5: ("left", "right"), 4: ("ccw", "cw"),
                3: ("forward", "back"), 2: ("left", "right"), 1: ("forward", "back"),
            }

            def _capture(joint_idx):
                """Hold position until the operator presses ENTER; return the live joint pos."""
                ev = threading.Event()
                threading.Thread(target=lambda: (input(), ev.set()), daemon=True).start()
                p = 0.0
                while not ev.is_set():
                    if r_state.ready():
                        p = float(r_state.data["pos"][joint_idx])
                    if w_ctrl.ready():
                        w_ctrl["pos"] = pos_cmd
                return p

            print("\n--- Manual range capture (J6, J5, J4, J3, J2, J1) ---")
            print("Torque is OFF. Move each joint to the named extreme, then press ENTER.")
            for joint_idx in [6, 5, 4, 3, 2, 1]:
                pos_dir, neg_dir = MANUAL_DIRS[joint_idx]
                print(f"\nMove J{joint_idx} to its {pos_dir.upper()} extreme, then press ENTER.")
                cap_max = _capture(joint_idx)
                print(f"Now move J{joint_idx} to its {neg_dir.upper()} extreme, then press ENTER.")
                cap_min = _capture(joint_idx)
                # Derive the zero from the swept range at the per-side fraction (wrist -> 0.5),
                # then store min/max relative to that zero.
                rel = ZERO_REL[SIDE].get(joint_idx, 0.5)
                offset = cap_min + rel * (cap_max - cap_min)
                offsets[joint_idx] = offset
                cal_ranges["cal_max"][joint_idx] = float(cap_max - offset)
                cal_ranges["cal_min"][joint_idx] = float(cap_min - offset)
                print(f"  J{joint_idx}: cal_max={cap_max - offset:.4f} cal_min={cap_min - offset:.4f} rel={rel:.3f} zero_off={offset:.4f}")
                # This joint is done: bake its zero and save immediately.
                ps0[joint_idx] = (ps0[joint_idx] + offset) % 1.0
                save_progress(f"  saved J{joint_idx} (zero + range)")

            warn_on_span_outliers(cal_ranges)
        finally:
            # Always restore the daemon's gripper torque-limit clamp on exit,
            # including Ctrl-C mid-calibration (the writer is still open here,
            # so this write reaches shared memory before the writer unlinks it).
            w_torque['calibrating'] = False

    print(f"\nFinal zeros: {ps0}")
    print(f"Saved to {ZEROS_FILE}")

    return offsets, cal_ranges

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print(f"{SIDE} ARM CALIBRATION")
    print("=" * 60)

    # PHASE 1: Setup udev
    if os.path.exists(f"/dev/{SYMLINK_NAME}"):
        print(f"/dev/{SYMLINK_NAME} already exists.")
        resp = input("Skip udev setup? [Y/n] ").strip().lower()
        if resp == "n":
            setup_udev()
        else:
            print("Skipping udev setup.")
    else:
        print(f"/dev/{SYMLINK_NAME} not found.")
        setup_udev()

    # PHASE 2: Joint zeroing and calibration ranges
    resp = input("\nRun joint zeroing? (daemon must be running) [Y/n] ").strip().lower()
    did_zeroing = False
    if resp == "n":
        print("Skipping joint zeroing.")
    else:
        calibrate_joints()
        did_zeroing = True

    print("\n" + "=" * 60)
    print(f"{SIDE} ARM CALIBRATION COMPLETE")
    print("=" * 60)

    if did_zeroing:
        print(f"Zeros: {ZEROS_FILE}")
    print(f">>> Restart daemon to apply: restart {DAEMON_NAME} <<<")

if __name__ == "__main__":
    main()
