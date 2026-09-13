from bbos import Reader, Writer, Config, Type
from driver import init_feetech, write_motors, read_motors, read_motors_block, configure_motor_phase, set_operating_mode
import limits
import scservo_sdk as scs
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import time
import os
import pinocchio as pin
import gc


def filter_temp(temp, temp_filtered, reject_count, max_delta, stuck_limit):
    """Update temp_filtered in place from a fresh raw temp read, rejecting
    single-tick sensor spikes (a jump > max_delta from the held value) while
    still recovering from a wrong/stuck value. A noise spike reverts within a
    tick so it never accumulates; but if a channel reads implausibly for
    stuck_limit consecutive ticks the shift is real (or the filter seeded wrong
    after a restart), so we snap to the raw reading instead of locking the
    published temperature forever.

    Failed per-motor reads arrive as NaN (driver.read_motors). NaN is ignored
    here: it is never accepted, counted, or re-seeded, so a dead sensor holds
    the last good value rather than poisoning temp_filtered with NaN (which
    would silently disable that motor's overtemp trips). The stuck re-seed only
    ever snaps to a finite reading. Mutates temp_filtered and reject_count."""
    finite = np.isfinite(temp)
    accept = finite & (np.abs(temp - temp_filtered) <= max_delta)
    temp_filtered[accept] = temp[accept]
    reject_count[accept] = 0
    rejected = finite & ~accept
    reject_count[rejected] += 1
    stuck = rejected & (reject_count >= stuck_limit)
    temp_filtered[stuck] = temp[stuck]
    reject_count[stuck] = 0


class ComplianceController:
    """Admittance controller: wrench estimation + spring-damper dynamics + IK.

    All parameters are read from the CFG object at construction time.
    Call ``step()`` each control tick; call ``set_target()`` when a new
    commanded pose arrives.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg.dt

        # Build reduced 7-DOF Pinocchio model
        full_model = pin.buildModelFromUrdf(cfg.urdf_path)
        arm_ids = [full_model.getJointId(n) for n in cfg.joint_names[:7]]
        assert all(jid < full_model.njoints for jid in arm_ids), "Missing arm joints"
        locks = [j for j in range(1, full_model.njoints) if j not in arm_ids]
        self.model = pin.buildReducedModel(full_model, locks, pin.neutral(full_model))
        assert self.model.nq == 7 and self.model.nv == 7
        self.data = self.model.createData()
        self._zero7 = np.zeros(7, dtype=np.float64)

        # EE frame
        self.ee_frame_id = self._find_frame(self.model, cfg.ee_frame)

        # Impedance gains (scalar, critically damped)
        mass = float(cfg.compliance_mass)
        kp_pos = float(cfg.compliance_kp_pos)
        kp_rot = float(cfg.compliance_kp_rot)
        self.mass = mass
        self.kp_pos = np.full(3, kp_pos, dtype=np.float64)
        self.kp_rot = np.full(3, kp_rot, dtype=np.float64)
        self.kd_pos = 2.0 * np.sqrt(self.kp_pos * mass)
        self.kd_rot = 2.0 * np.sqrt(self.kp_rot * 1.0)  # inertia = 1
        self.force_reg = float(cfg.compliance_force_reg)
        self.torque_reg = float(cfg.compliance_torque_reg)
        self.normal_axis = int(cfg.compliance_normal_axis)  # 0=x, 1=y, 2=z

        # Runtime flags (toggled via IPC)
        self.axis_aligned = False
        self.force_only = False

        # State
        self.x = np.zeros(6, dtype=np.float64)       # [pos, rotvec]
        self.v = np.zeros(6, dtype=np.float64)       # [lin_vel, ang_vel]
        self.x_des = np.zeros(6, dtype=np.float64)   # target pose
        self.wrench_bias = np.zeros(6, dtype=np.float64)
        self.initialized = False

        # IK solver (lazily initialized via cfg.ik.init())
        cfg.ik.init()

    # ------------------------------------------------------------------
    @staticmethod
    def _find_frame(model, name):
        for i, f in enumerate(model.frames):
            if f.name == name:
                return i
        raise ValueError(f"Frame '{name}' not found")

    # ------------------------------------------------------------------
    def reset(self):
        """Mark as uninitialized so next step() recalibrates bias."""
        self.initialized = False
        self.v[:] = 0.0

    # ------------------------------------------------------------------
    def set_target(self, cmd_pos):
        """Update desired EE pose from commanded joint positions (motor-space).

        ``cmd_pos`` is the pos array from ``r_ctrl.data['pos']`` (motor turns).
        """
        ctrl_urdf = np.asarray(
            self.cfg.q2urdf(np.array(cmd_pos, dtype=np.float32)), dtype=np.float64
        )[:7]
        ctrl_pos, ctrl_quat = self.cfg.ik.fk(ctrl_urdf.tolist())
        self.x_des[:3] = ctrl_pos
        self.x_des[3:6] = Rot.from_quat(ctrl_quat).as_rotvec()

    # ------------------------------------------------------------------
    def step(self, ps_acc, vs_filtered, cs):
        """Run one compliance tick.

        Parameters
        ----------
        ps_acc : (dof,) motor-space positions (turns, zeroed)
        vs_filtered : (dof,) filtered motor-space velocities
        cs : (dof,) motor currents (A)

        Returns
        -------
        comp_joints : (7,) compliant motor-space joint targets, or None on IK failure.
        """
        cfg = self.cfg

        # FK + dynamics in URDF space
        q = np.asarray(cfg.q2urdf(ps_acc.copy()), dtype=np.float64)[:7]
        v = np.asarray(cfg.q2urdf(vs_filtered.copy()), dtype=np.float64)[:7]

        # Motor torques -> URDF torques (skip J0 prismatic)
        tau = np.zeros(7, dtype=np.float64)
        tau[1:7] = (cs * cfg.kt)[:7].astype(np.float64)[1:7]

        # Bias torque (gravity + Coriolis)
        tau_bias = pin.rnea(self.model, self.data, q, v, self._zero7)
        tau_bias[0] = 0.0

        # Kinematics + Jacobian
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jac = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        # Wrench estimation: tau_ext = -(tau_motor - tau_bias)
        tau_ext = -(tau - tau_bias)
        Jf, Jt = jac[:3], jac[3:]

        if not self.axis_aligned:
            # Dense: full 3-DOF least-squares (Eq. 9)
            eye3 = np.eye(3)
            force_est = np.linalg.solve(Jf @ Jf.T + self.force_reg * eye3, Jf @ tau_ext)
            torque_est = (
                np.zeros(3, dtype=np.float64) if self.force_only
                else np.linalg.solve(Jt @ Jt.T + self.torque_reg * eye3, Jt @ tau_ext)
            )
        else:
            # Axis-aligned: 1-DOF problems along EE frame axes (Eq. 10)
            ee_rot = np.array(self.data.oMf[self.ee_frame_id].rotation)
            normal = ee_rot[:, self.normal_axis]
            tangent_axes = [i for i in range(3) if i != self.normal_axis]

            # Force along contact normal
            proj = (normal @ Jf)  # (n,)
            f_scalar = (proj @ tau_ext) / (proj @ proj + self.force_reg)
            force_est = f_scalar * normal

            # Torque along tangent axes
            torque_est = np.zeros(3, dtype=np.float64)
            if not self.force_only:
                for ax in tangent_axes:
                    t_axis = ee_rot[:, ax]
                    proj_t = (t_axis @ Jt)
                    t_scalar = (proj_t @ tau_ext) / (proj_t @ proj_t + self.torque_reg)
                    torque_est += t_scalar * t_axis

        wrench_raw = np.concatenate([force_est, torque_est])

        # First tick: calibrate bias + init pose from FK
        if not self.initialized:
            self.wrench_bias[:] = wrench_raw
            oMf = self.data.oMf[self.ee_frame_id]
            self.x[:3] = oMf.translation
            self.x[3:6] = pin.log3(oMf.rotation)
            self.v[:] = 0.0
            self.x_des[:] = self.x
            self.initialized = True

        wrench = wrench_raw - self.wrench_bias

        # Spring-damper dynamics (semi-implicit Euler)
        # Translation
        acc_pos = (
            self.kp_pos * (self.x_des[:3] - self.x[:3])
            - self.kd_pos * self.v[:3]
            + wrench[:3]
        ) / self.mass
        self.v[:3] += acc_pos * self.dt
        self.x[:3] += self.v[:3] * self.dt

        # Rotation
        ori_err = (
            Rot.from_rotvec(self.x_des[3:6]) * Rot.from_rotvec(self.x[3:6]).inv()
        ).as_rotvec()
        acc_rot = self.kp_rot * ori_err - self.kd_rot * self.v[3:6] + wrench[3:6]
        self.v[3:6] += acc_rot * self.dt
        self.x[3:6] = (
            Rot.from_rotvec(self.v[3:6] * self.dt) * Rot.from_rotvec(self.x[3:6])
        ).as_rotvec()

        # IK: task-space pose -> joint-space
        comp_quat = Rot.from_rotvec(self.x[3:6]).as_quat().tolist()
        cfg.ik.reset(q.tolist())
        comp_ik = cfg.ik.solve(self.x[:3].tolist(), comp_quat)
        if comp_ik is None or len(comp_ik) == 0:
            return None

        comp_urdf_full = np.zeros(cfg.dof, dtype=np.float32)
        comp_urdf_full[:7] = np.array(comp_ik[:7], dtype=np.float32)
        return cfg.urdf2q(comp_urdf_full)

ARM_NAME = os.path.basename(os.getcwd())
CFG = Config(ARM_NAME)

# Joint-range clamp: cal_min/cal_max (ranges.calibration.json) are in the same
# motor-turn/zero-relative frame as arm_ctrl.pos, so the control loop can reject
# out-of-range commands directly — no IK involvement. None when uncalibrated.
RANGE_LO, RANGE_HI = limits.load_joint_range(len(CFG.motors))
if RANGE_LO is not None:
    limits.warn_waypoints_out_of_range("home", CFG.home, RANGE_LO, RANGE_HI)
    if hasattr(CFG, "startup_waypoints"):
        limits.warn_waypoints_out_of_range("startup", CFG.startup_waypoints, RANGE_LO, RANGE_HI)

def run_read_only(port, packet):
    print("READ-ONLY arm daemon mode: no motor writes, no torque enable, no homing.", flush=True)
    if os.path.exists('zeros.txt'):
        ps0 = np.loadtxt('zeros.txt', dtype=np.float32)
    else:
        ps0 = np.zeros(len(CFG.motors), dtype=np.float32)
        print("READ-ONLY: zeros.txt missing; publishing raw turns relative to zero.", flush=True)

    ps_raw = np.zeros(len(CFG.motors), dtype=np.float32)
    ps_acc = np.zeros(len(CFG.motors), dtype=np.float32)
    ps_acc_prev = np.zeros(len(CFG.motors), dtype=np.float32)
    vs = np.zeros(len(CFG.motors), dtype=np.float32)
    vs_filtered = np.zeros(len(CFG.motors), dtype=np.float32)
    cs = np.zeros(len(CFG.motors), dtype=np.float32)
    temp = np.zeros(len(CFG.motors), dtype=np.float32)
    temp_filtered = np.zeros(len(CFG.motors), dtype=np.float32)
    temp_reject_count = np.zeros(len(CFG.motors), dtype=np.int32)
    vel_filter_initialized = False
    temp_initialized = False

    with Writer(f"{ARM_NAME}.state", Type("arm_state")) as w_state:
        while True:
            read_ok = True
            try:
                read_motors(port, packet, "Present_Position", ps_raw)
                read_motors(port, packet, "Present_Velocity", vs)
                read_motors(port, packet, "Present_Current", cs)
                read_motors(port, packet, "Present_Temperature", temp)
            except Exception as e:
                read_ok = False
                print(f"READ-ONLY read error: {e}", flush=True)

            if read_ok:
                ps_acc[:] = ps_raw - ps0
                ps_acc[1:] -= np.round(ps_acc[1:])
                ps_acc_prev[:] = ps_acc
                if not vel_filter_initialized:
                    vs_filtered[:] = vs
                    vel_filter_initialized = True
                else:
                    vs_filtered[:] = CFG.lpf_alpha * vs + (1 - CFG.lpf_alpha) * vs_filtered
                if not temp_initialized:
                    temp_filtered[:] = temp
                    temp_initialized = True
                else:
                    filter_temp(temp, temp_filtered, temp_reject_count, CFG.temp_max_delta, CFG.temp_stuck_ticks)

                with w_state.buf() as b:
                    b['pos'] = ps_acc
                    b['vel'] = vs_filtered
                    b['torque'] = cs * CFG.kt
                    b['temp'] = temp_filtered
                    b['current'] = cs

            time.sleep(CFG.dt)

if __name__ == "__main__":
    gc.disable()
    port, packet = init_feetech()
    if getattr(CFG, "read_only", False):
        run_read_only(port, packet)

    # ========================================================================
    # Motor register writes
    # ========================================================================
    # One-time startup config.
    st3120_mask = np.asarray(CFG.st3120_mask, dtype=np.bool_)
    hls_mask = ~st3120_mask
    write_motors(port, packet, "Torque_Enable", np.zeros(len(CFG.motors), dtype=np.uint8))  # SRAM, but must precede reconfig (torque off)
    time.sleep(0.05)

    # ----- EPROM writes -----
    write_motors(port, packet, "Lock", np.zeros(len(CFG.motors), dtype=np.uint8))  # SRAM, but gates the EPROM writes below (unlock)
    write_motors(port, packet, "Min_Position_Limit", np.zeros(len(CFG.motors), dtype=np.uint16))
    write_motors(port, packet, "Max_Position_Limit", np.zeros(len(CFG.motors), dtype=np.uint16))
    configure_motor_phase(port, packet)
    write_motors(port, packet, "Angle_Resolution", np.array(CFG.angle_resolution, dtype=np.uint8))
    write_motors(port, packet, "Operating_Mode", CFG.operating_mode)
    write_motors(port, packet, "Minimum_Startup_Torque", np.zeros(len(CFG.motors), dtype=np.uint8))
    write_motors(port, packet, "Max_Torque_Limit", np.asarray(CFG.max_torque_limit, dtype=np.uint16))
    hcl_arr = np.asarray(CFG.hard_current_limit, dtype=np.float32)
    protection_current_arr = np.asarray(CFG.protection_current, dtype=np.uint16)
    # Protection_Current: GroupSyncWrite silently fails for this EPROM register,
    # so use individual writes with EPROM flash delay
    prot_addr = 28  # Protection_Current register address
    for i, sid in enumerate(CFG.motors):
        prot_val = int(protection_current_arr[i])
        comm, err = packet.write2ByteTxRx(port, sid, prot_addr, prot_val)
        time.sleep(0.03)  # EPROM flash commit time
        if comm != scs.COMM_SUCCESS:
            print(f"Motor {sid}: Protection_Current write FAILED: {packet.getTxRxResult(comm)}", flush=True)
        else:
            # Individual readback (GroupSyncRead fails for this register)
            rb, comm2, _ = packet.read2ByteTxRx(port, sid, prot_addr)
            if comm2 == scs.COMM_SUCCESS:
                print(f"Motor {sid}: Protection_Current = {rb} (expected {prot_val}) {'OK' if rb == prot_val else 'MISMATCH!'}", flush=True)
            else:
                print(f"Motor {sid}: Protection_Current readback FAILED", flush=True)
    write_motors(port, packet, "Overcurrent_Protection_Time", np.full(len(CFG.motors), CFG.overcurrent_protection_time, dtype=np.uint8))
    write_motors(port, packet, "Max_Temperature_Limit", np.full(len(CFG.motors), CFG.max_temperature, dtype=np.uint8))
    write_motors(port, packet, "Unloading_Condition", np.full(len(CFG.motors), CFG.unloading_condition, dtype=np.uint8))
    write_motors(port, packet, "Return_Delay_Time", np.array(CFG.return_delay_time, dtype=np.uint8))
    write_motors(port, packet, "Lock", np.ones(len(CFG.motors), dtype=np.uint8))  # SRAM, but gates the EPROM writes above (re-lock)
    # Verify critical protection registers were written
    for reg, expected in [
        ("Protection_Current", protection_current_arr),
        ("Overcurrent_Protection_Time", CFG.overcurrent_protection_time),
        ("Max_Temperature_Limit", CFG.max_temperature),
        ("Unloading_Condition", CFG.unloading_condition),
    ]:
        readback = np.zeros(len(CFG.motors), dtype=np.float32)
        read_motors(port, packet, reg, readback)
        vals = readback.astype(int).tolist()
        expected_arr = np.asarray(expected, dtype=np.int32)
        if expected_arr.shape == ():
            ok = all(v == int(expected_arr) for v in vals)
            expected_print = int(expected_arr)
        else:
            ok = np.array_equal(readback.astype(np.int32), expected_arr)
            expected_print = expected_arr.tolist()
        print(f"  {reg}: expected={expected_print}, got={vals} {'OK' if ok else 'MISMATCH!'}", flush=True)

    # ----- SRAM writes -----
    write_motors(port, packet, "Acceleration", CFG.acceleration)
    write_motors(port, packet, "Goal_Velocity", np.asarray(CFG.goal_velocity, dtype=np.uint16))
    write_motors(port, packet, "Torque_Limit", np.asarray(CFG.torque_limit, dtype=np.uint16))
    write_motors(port, packet, "Target_Current", hcl_arr, mask=hls_mask)
    write_motors(port, packet, "Running_Time", np.zeros(len(CFG.motors), dtype=np.uint16), mask=st3120_mask)
    write_motors(port, packet, "Kp", CFG.Kp)
    write_motors(port, packet, "Ki", CFG.Ki)
    write_motors(port, packet, "Kd", CFG.Kd)

    # ========================================================================
    # Load zeros
    # ========================================================================
    # Load existing zeros, or initialize from current position.
    if os.path.exists('zeros.txt'):
        ps0 = np.loadtxt('zeros.txt', dtype=np.float32)
        ps0_read = np.zeros(len(CFG.motors), dtype=np.float32)
        read_motors(port, packet, "Present_Position", ps0_read)
        nan_mask = np.isnan(ps0)
        ps0[nan_mask] = ps0_read[nan_mask]
    else:
        # No zeros file - initialize from current position
        # The arm should be at physical zero when this runs!
        print("No zeros.txt found, initializing from current position...", flush=True)
        ps0 = np.zeros(len(CFG.motors), dtype=np.float32)
        read_motors(port, packet, "Present_Position", ps0)
        ps0_j0 = ps0[0]  # j0 is linear, keep full position
        ps0 = ps0 - np.floor(ps0)  # Keep only S1 offset (0-1 range)
        ps0[0] = ps0_j0  # Restore j0 full position
        print(f"Initial zeros (j0=full, j1-7=S1): {ps0}", flush=True)
    
    # ========================================================================
    # Zeroing procedure for J0
    # ========================================================================
    if getattr(CFG, "skip_startup_homing", False):
        print("Skipping J0 startup homing/calibration.", flush=True)
    else:
        # ===== ZEROING PROCEDURE FOR J0 (torque-mode homing) =====
        # Drive J0 UP into the top hardstop under constant torque (current mode), detect the
        # stall by zero velocity, then set the zero j0_top_offset_turns BELOW the stop so the
        # lift never rests on the hardstop in normal use (that back-off point becomes J0 max).
        # The zero is captured while J0 is still pressed at the stop in current mode -- one climb,
        # no re-press. J0 carries the arm against gravity (~0.6-0.9 Nm) and DROPS when torque is
        # removed (see J0_HOLD_EXPERIMENTS.md), so it sags after the climb when we switch to
        # position mode for the main loop, but that's harmless: the zero is already locked to the
        # encoder and the first commanded position recovers it.
        j0_dir = float(np.sign(CFG.j0_increment))          # raw-position direction for "up"
        # Current sign for "up" is the OPPOSITE of the position increment: on these HLS servos a
        # positive Target_Current DECREASES position (verified on hardware -- +sign drove the arm
        # DOWN). So drive up with -sign(j0_increment) current.
        j0_cur_dir = -j0_dir
        j0_kt = float(np.asarray(CFG.kt, dtype=np.float32)[0])
        j0_ramp_s = float(CFG.j0_cal_ramp_s)
        j0_off = j0_dir * float(CFG.j0_top_offset_turns)   # raw shift from the stop toward home
        j0_cur_full = j0_cur_dir * float(CFG.j0_cal_tau) / j0_kt        # signed Target_Current (A), up
        j0_cur_start = j0_cur_dir * float(CFG.j0_cal_tau_start) / j0_kt
        print(f"Homing j0 to TOP (torque mode, {CFG.j0_cal_tau:.2f} Nm up, current sign {j0_cur_dir:+.0f})...", flush=True)

        j0_mask = np.zeros(len(CFG.motors), dtype=np.bool_); j0_mask[0] = True
        # J0 -> current mode (op 2); other joints stay as configured (torque off, position mode).
        op_modes = np.zeros(len(CFG.motors), dtype=np.uint8); op_modes[0] = 2
        set_operating_mode(port, packet, op_modes, mask=j0_mask)
        write_motors(port, packet, "Target_Current", np.zeros(len(CFG.motors), dtype=np.float32), mask=j0_mask)
        torque_enable = np.zeros(len(CFG.motors), dtype=np.uint8); torque_enable[0] = 1
        write_motors(port, packet, "Torque_Enable", torque_enable)

        pos_cmd = np.zeros(len(CFG.motors), dtype=np.float32)
        cur_cmd = np.zeros(len(CFG.motors), dtype=np.float32)
        read_motors(port, packet, "Present_Position", pos_cmd)
        j0_start = float(pos_cmd[0])
        j0_stop_raw = j0_start
        stall_ref = j0_start          # position the stall window is measured against
        stall_since = None            # monotonic time J0 first went still (None = still moving)
        moved = False
        read_retries = 0
        t0 = time.monotonic()
        while True:
            time.sleep(0.01)
            now = time.monotonic()
            elapsed = now - t0
            if elapsed > CFG.j0_homing_timeout_s:
                write_motors(port, packet, "Target_Current", np.zeros(len(CFG.motors), dtype=np.float32), mask=j0_mask)
                write_motors(port, packet, "Torque_Enable", np.zeros(len(CFG.motors), dtype=np.uint8))
                raise RuntimeError(f"J0 homing timed out after {CFG.j0_homing_timeout_s}s "
                                   f"(moved={moved}); check j0_cal_tau (lift) and torque sign")
            ramp = min(1.0, elapsed / j0_ramp_s) if j0_ramp_s > 0 else 1.0
            cur_cmd[0] = j0_cur_start + (j0_cur_full - j0_cur_start) * ramp
            write_motors(port, packet, "Target_Current", cur_cmd, mask=j0_mask)
            try:
                read_motors(port, packet, "Present_Position", pos_cmd)
                read_retries = 0
            except Exception as e:
                read_retries += 1
                print(f"Homing j0: read error (retry {read_retries}): {e}", flush=True)
                if read_retries > 5:
                    raise RuntimeError("Too many read errors during homing")
                continue
            j0_stop_raw = float(pos_cmd[0])
            up_travel = (j0_stop_raw - j0_start) * j0_dir   # >0 = moved up toward the stop
            # Wrong-way guard: torque sign error OR torque too low to hold gravity (J0 sagging).
            if up_travel < -float(CFG.j0_cal_wrongdir_turns):
                write_motors(port, packet, "Target_Current", np.zeros(len(CFG.motors), dtype=np.float32), mask=j0_mask)
                write_motors(port, packet, "Torque_Enable", np.zeros(len(CFG.motors), dtype=np.uint8))
                raise RuntimeError(f"J0 ran {up_travel:+.3f} turns the WRONG way; flip j0 torque sign "
                                   f"or raise j0_cal_tau (cannot hold gravity)")
            if up_travel > float(CFG.j0_cal_min_travel_turns):
                moved = True
            # Stall = position stops changing (velocity ~0) for j0_cal_settle_s, after J0 has
            # lifted and torque has fully ramped. Position-based, not the coarsely-quantized
            # Present_Velocity (~0.0122 turns/s/count), so it neither false-stalls mid-climb nor
            # stops short of the top.
            if abs(j0_stop_raw - stall_ref) > float(CFG.j0_cal_move_eps_turns):
                stall_ref = j0_stop_raw
                stall_since = None
            elif moved and elapsed >= j0_ramp_s:
                if stall_since is None:
                    stall_since = now
                elif now - stall_since >= float(CFG.j0_cal_settle_s):
                    print(f"Homing j0: stopped at top for {CFG.j0_cal_settle_s:.1f}s "
                          f"(up_travel={up_travel:+.3f} turns) [WALL]", flush=True)
                    break

        # J0 is pressed firmly against the top hardstop right here: still in current mode, torque
        # on, at the end of the climb (position held within move_eps for the full settle). Single
        # climb -- no re-press. The catch (right arm): its J0 top sits near a multi-turn boundary,
        # so when the servo re-seeds its turn counter on the current->position torque toggle, the
        # reading jumps by a WHOLE TURN (the left's top is mid-turn, so it's stable). The servo's
        # mid-position recenter (Torque_Enable=128) is a no-op in current mode, so instead we MEASURE
        # that integer-turn shift and correct for it: capture the pressed top in the current-mode
        # reference, switch to position mode (what the main loop reads in), capture again, and snap
        # the zero into the position-mode reference. round() discards the sub-turn sag from the
        # torque-off switch, so only the whole-turn shift is removed.
        def _read_j0_median(n=5):
            vals = []
            for _ in range(n):
                read_motors(port, packet, "Present_Position", pos_cmd)
                v = float(pos_cmd[0])
                if np.isfinite(v):
                    vals.append(v)
                time.sleep(0.004)
            return float(np.median(vals)) if vals else float('nan')

        p_top = _read_j0_median()   # pressed top, current-mode reference

        # Hand J0 to position mode for the main control loop. The mode switch needs torque-off, so
        # J0 sags a little here -- that only moves the fractional part of the reading, which the
        # round() below discards.
        write_motors(port, packet, "Target_Current", np.zeros(len(CFG.motors), dtype=np.float32), mask=j0_mask)
        write_motors(port, packet, "Torque_Enable", np.zeros(len(CFG.motors), dtype=np.uint8), mask=j0_mask)
        op_modes[0] = 0
        set_operating_mode(port, packet, op_modes, mask=j0_mask)   # torque stays off (was off)
        p_post = _read_j0_median()  # same physical region, position-mode reference (sagged a bit)

        # Integer-turn reference shift between current- and position-mode (0 on a stable arm/joint).
        delta = p_top - p_post
        turn_shift = float(np.round(delta))
        sag_frac = delta - turn_shift
        # Zero = pressed top expressed in the position-mode reference, minus the offset. p_top has no
        # sag (it was pressed), so (p_top - turn_shift) is the true post-switch top.
        ps0[0] = (p_top - turn_shift) - j0_off
        print(f"[DIAG] J0 top: current-ref={p_top:.4f} position-ref={p_post:.4f} "
              f"turn_shift={turn_shift:+.0f} sag_frac={sag_frac:+.4f}", flush=True)
        if abs(sag_frac) > 0.4:
            print(f"WARNING: J0 mode-switch sag ({sag_frac:+.4f} turn) is near 0.5 -- the turn-shift "
                  f"correction could pick the wrong turn this run", flush=True)
        print(f"Motor position at TOP (position-ref): {p_top - turn_shift:.4f}; "
              f"J0 zero (top - {CFG.j0_top_offset_turns} turns): {ps0[0]:.4f}", flush=True)
        print(f"Updated j0 zero: {ps0[0]}, full zeros: {ps0}", flush=True)
        np.savetxt('zeros.txt', ps0)
        print(f"j0 reached TOP (home)", flush=True)
        # Restore J0's position-mode current budget. In position mode Target_Current is the MAX
        # running current, but the climb above left it at the press current; the main loop won't
        # restore it (J0 never entered tau mode in the main loop's tracking), so set it to the full
        # hard_current_limit now so later position-mode holds have full current budget.
        write_motors(port, packet, "Target_Current", hcl_arr, mask=j0_mask & hls_mask)
        write_motors(port, packet, "Torque_Enable", np.zeros(len(CFG.motors), dtype=np.uint8))

    # ========================================================================
    # Main control loop
    # ========================================================================
    # ps0[0] is full position (j0 linear). ps0[1:] is S1 offset (0-1 range).
    # Track turns by detecting S1 wraparound for rotational joints.
    ps_raw = np.zeros(len(CFG.motors), dtype=np.float32)
    read_motors(port, packet, "Present_Position", ps_raw)
    ps_raw_s1 = ps_raw - np.floor(ps_raw)  # Current S1 position (0-1)
    # Initialize turns so position starts near 0 (assumes arm is at physical zero)
    initial_diff = ps_raw_s1 - ps0
    turns = -np.round(initial_diff).astype(np.float32)
    turns[0] = 0  # j0 is linear, no turn tracking
    # Position = s1 + turns - ps0
    ps_acc = ps_raw_s1 + turns - ps0
    ps_acc[0] = ps_raw[0] - ps0[0]  # j0 is linear, use full position
    ps_acc_prev = ps_acc.copy()
    ps_raw_s1_prev = ps_raw_s1.copy()
    
    vs = np.zeros(len(CFG.motors), dtype=np.float32)
    vs_fd = np.zeros(len(CFG.motors), dtype=np.float32)
    vs_filtered = np.zeros(len(CFG.motors), dtype=np.float32)
    cs = np.zeros(len(CFG.motors), dtype=np.float32)
    temp = np.zeros(len(CFG.motors), dtype=np.float32)
    temp_filtered = np.zeros(len(CFG.motors), dtype=np.float32)
    temp_reject_count = np.zeros(len(CFG.motors), dtype=np.int32)
    temp_initialized = False
    pos_filtered = ps_acc.copy()
    current_near_limit = False  # set when any motor is within 1A of hard limit
    torque_enable = np.zeros(len(CFG.motors), dtype=np.bool_)
    tau_mode = np.zeros(len(CFG.motors), dtype=np.bool_)
    tau_cmd = np.zeros(len(CFG.motors), dtype=np.float32)
    current_cmd = np.zeros(len(CFG.motors), dtype=np.float32)
    current_cmd_prev = np.zeros(len(CFG.motors), dtype=np.float32)
    current_cmd_limit = np.asarray(CFG.protection_current, dtype=np.float32) * 0.0065
    goal_raw_safe = None  # last safe goal position for current-limiting fallback
    tau_watchdog_ns = int(CFG.dt * 1e9 * 2.0)
    last_tau_cmd_ns = 0
    using_tau_ctrl = False
    vel_filter_initialized = False
    print(f"Starting control loop.", flush=True)
    torque_was_enabled = np.zeros(len(CFG.motors), dtype=np.bool_)
    # Per-joint current limit cooldown tracking
    current_limit_cooldown = np.zeros(len(CFG.motors), dtype=np.float64)  # cooldown end time per joint (0 = not in cooldown)
    current_limit_recovery_start_pos = np.zeros(len(CFG.motors), dtype=np.float32)  # position at start of recovery
    current_limit_recovery_start = np.zeros(len(CFG.motors), dtype=np.float64)  # time when recovery interpolation started
    last_ctrl_pos = np.zeros(len(CFG.motors), dtype=np.float32)
    # J0 software integrator a daemon-side integral term
    # that drives J0's steady-state hold error to ~0, which the servo firmware I-term can't.
    # getattr-guarded so a config without these fields can't crash this shared daemon; the
    # default ki=0.0 leaves it disabled (baseline behaviour) in that case.
    j0_integ_ki = float(CFG.j0_integ_ki)
    j0_integ_clamp = float(CFG.j0_integ_clamp)
    j0_integ_gate = float(CFG.j0_integ_gate)
    j0_integ_band = float(CFG.j0_integ_band)
    j0_integ = 0.0          # accumulated J0 goal-position bias (turns)
    j0_cmd_prev = None      # previous commanded J0 position, for the stationary gate
    # J7 (gripper, index dof-1) current-relief loop — sheds holding current to keep a grasp cool.
    j7_relief_enable = bool(getattr(CFG, "j7_relief_enable", False))
    j7_relief_vel_stop = float(getattr(CFG, "j7_relief_vel_stop", 0.02))
    j7_relief_hold_time = float(getattr(CFG, "j7_relief_hold_time", 0.3))
    j7_relief_i_hold = float(getattr(CFG, "j7_relief_i_hold", 1.5))
    j7_relief_db = float(getattr(CFG, "j7_relief_db", 0.5))
    j7_relief_step = float(getattr(CFG, "j7_relief_step", 5e-4))
    j7_relief_bias_max = float(getattr(CFG, "j7_relief_bias_max", 0.1))
    j7_relief_gate = float(getattr(CFG, "j7_relief_gate", 5e-3))
    j7_relief_debug = bool(getattr(CFG, "j7_relief_debug", False))
    j7_bias = 0.0           # accumulated J7 goal backoff (turns)
    j7_stall_t = 0.0        # seconds the jaw has been held still
    j7_cmd_prev = None      # previous commanded J7 position, to detect a new grasp/move
    j7_dbg = 0              # tick counter for the [j7relief] diagnostic
    sustained_current_time = np.zeros(len(CFG.motors), dtype=np.float64)
    sustained_threshold = np.asarray(CFG.hard_current_limit, dtype=np.float32) * CFG.sustained_current_scale
    sustained_temp_time = np.zeros(len(CFG.motors), dtype=np.float64)

    # Gripper admittance state
    g = len(CFG.motors) - 1
    g_ref = float(ps_acc[g])   # admittance reference position
    g_vel = 0.0                # admittance reference velocity

    # Gripper torque-limit management state.
    # The gripper holds a static gentle SRAM Torque_Limit in normal operation, raised to the
    # higher cal value only while calibrating so the open/close sweeps reach the stop-detection
    # current (the gentle static limit would stall them short).
    gripper_tl_mask = np.zeros(len(CFG.motors), dtype=np.bool_); gripper_tl_mask[g] = True
    gripper_normal_tl = int(CFG.torque_limit[g])       # static SRAM Torque_Limit in normal operation
    gripper_cal_tl = int(CFG.gripper_cal_torque_limit)  # SRAM Torque_Limit held during calibration
    current_gripper_tl = gripper_normal_tl  # value last written (startup wrote CFG.torque_limit)
    calibrating = False

    # Compliance controller
    compliance_mode = False
    comp_ctrl = ComplianceController(CFG)

    with Writer(f"{ARM_NAME}.state", Type("arm_state")) as w_state, \
        Reader(f"{ARM_NAME}.torque", keeptime=False) as r_torque, \
        Reader(f"{ARM_NAME}.ctrl") as r_ctrl:
        
        IDLE_DISABLE_RESEND_S = 1.0  # rate-limit the no-controller torque-cut resend
        idle_disable_at = 0.0

        while True:
            if r_torque.ready():
                torque_enable = r_torque.data['enable']
                new_compliance = bool(r_torque.data['compliance_mode'])
                if new_compliance != compliance_mode:
                    print(f"COMPLIANCE MODE: {compliance_mode} -> {new_compliance}", flush=True)
                    if new_compliance:
                        comp_ctrl.reset()
                compliance_mode = new_compliance
                comp_ctrl.axis_aligned = bool(r_torque.data['axis_aligned'])
                comp_ctrl.force_only = bool(r_torque.data['force_only'])
                calibrating = bool(r_torque.data['calibrating'])
                new_tau_mode = np.asarray(r_torque.data['tau_mode'], dtype=np.bool_)
                tau_allowed = np.asarray(CFG.tau_mode_allowed, dtype=np.bool_)
                blocked_tau = new_tau_mode & ~tau_allowed
                if np.any(blocked_tau):
                    ids = [CFG.motors[i] for i in np.where(blocked_tau)[0]]
                    print(f"tau_mode blocked for motors {ids}; keeping position mode", flush=True)
                    new_tau_mode[blocked_tau] = False

                # Switch operating mode for joints changing between position/current control
                entering_tau = new_tau_mode & ~tau_mode
                leaving_tau = ~new_tau_mode & tau_mode
                mode_changing = entering_tau | leaving_tau
                if np.any(mode_changing):
                    write_motors(port, packet, "Torque_Enable", np.zeros(len(CFG.motors), dtype=np.uint8), mask=mode_changing)
                    new_op = np.where(new_tau_mode, 2, 0).astype(np.uint8)
                    set_operating_mode(port, packet, new_op, mask=mode_changing)
                    if np.any(leaving_tau):
                        write_motors(port, packet, "Goal_Velocity", np.asarray(CFG.goal_velocity, dtype=np.uint16), mask=leaving_tau)
                        write_motors(port, packet, "Target_Current", hcl_arr, mask=leaving_tau & hls_mask)
                        write_motors(port, packet, "Running_Time", np.zeros(len(CFG.motors), dtype=np.uint16), mask=leaving_tau & st3120_mask)
                        read_motors(port, packet, "Present_Position", ps_raw)
                        _goal = ps_raw.copy()
                        write_motors(port, packet, "Goal_Position", _goal, mask=leaving_tau)
                    if np.any(entering_tau):
                        write_motors(port, packet, "Target_Current", np.zeros(len(CFG.motors), dtype=np.float32), mask=entering_tau & hls_mask)
                    tau_mode[:] = new_tau_mode

                newly_enabled = torque_enable & ~torque_was_enabled
                if newly_enabled[0]:
                    j0_integ = 0.0      # clear J0 integrator windup on (re)enable
                    j0_cmd_prev = None
                if newly_enabled[g]:
                    j7_bias = 0.0       # clear J7 relief backoff on (re)enable
                    j7_stall_t = 0.0
                    j7_cmd_prev = None
                if np.any(newly_enabled):
                    pos_newly = newly_enabled & ~tau_mode
                    if np.any(pos_newly):
                        set_operating_mode(port, packet, np.zeros(len(CFG.motors), dtype=np.uint8), mask=pos_newly)
                        write_motors(port, packet, "Goal_Velocity", np.asarray(CFG.goal_velocity, dtype=np.uint16), mask=pos_newly)
                        read_motors(port, packet, "Present_Position", ps_raw)
                        ps_raw_s1 = ps_raw - np.floor(ps_raw)
                        pos_filtered[pos_newly] = (ps_raw_s1 + turns - ps0)[pos_newly]
                        if pos_newly[0]:
                            pos_filtered[0] = ps_raw[0] - ps0[0]
                        _goal = ps_raw.copy()
                        write_motors(port, packet, "Goal_Position", _goal, mask=pos_newly)
                    tau_newly = newly_enabled & tau_mode
                    if np.any(tau_newly):
                        set_operating_mode(port, packet, np.full(len(CFG.motors), 2, dtype=np.uint8), mask=tau_newly)
                        write_motors(port, packet, "Target_Current", np.zeros(len(CFG.motors), dtype=np.float32), mask=tau_newly)
                write_motors(port, packet, "Torque_Enable", torque_enable)
                torque_was_enabled = torque_enable.copy()
            if not r_ctrl.readable:
                # sync-writes are unacked: cut immediately on loss, then re-send at 1 Hz
                if time.monotonic() - idle_disable_at >= IDLE_DISABLE_RESEND_S:
                    write_motors(port, packet, "Torque_Enable", np.zeros(len(CFG.motors), dtype=np.bool_))
                    idle_disable_at = time.monotonic()
            else:
                idle_disable_at = 0.0
            
            if w_state.ready():
                read_motors_block(port, packet,
                                  ("Present_Position", "Present_Velocity", "Present_Current", "Present_Temperature"),
                                  (ps_raw, vs, cs, temp))
                ps_raw_s1 = ps_raw - np.floor(ps_raw)
                # Detect S1 wrap: if S1 jumped by more than 0.5, we crossed a turn boundary
                s1_delta = ps_raw_s1 - ps_raw_s1_prev
                turns = turns + np.where(s1_delta < -0.5, 1, np.where(s1_delta > 0.5, -1, 0))
                turns[0] = 0  # j0 is linear, no turn tracking
                ps_raw_s1_prev = ps_raw_s1.copy()
                # Position = s1 + turns - ps0 (unwrapped position relative to zero)
                ps_acc = ps_raw_s1 + turns - ps0
                ps_acc[0] = ps_raw[0] - ps0[0]  # j0 is linear, use full position
                vs_fd[:] = (ps_acc - ps_acc_prev) / CFG.dt
                ps_acc_prev = ps_acc.copy()
                if not np.all(np.isfinite(vs)):
                    vs[:] = vs_fd
                assert vs.shape == (len(CFG.motors),), f"Unexpected velocity shape: {vs.shape}"
                if not vel_filter_initialized:
                    vs_filtered[:] = vs
                    vel_filter_initialized = True
                else:
                    vs_filtered[:] = CFG.lpf_alpha * vs + (1 - CFG.lpf_alpha) * vs_filtered
                abs_cs = np.abs(cs)
                now = time.monotonic()
                limits.check_current(port, packet, CFG, abs_cs, now, torque_enable, torque_was_enabled,
                                     current_limit_cooldown, sustained_current_time, sustained_threshold)
                limits.service_recovery(port, packet, CFG, now, ps_raw, ps_acc, torque_enable, torque_was_enabled,
                                        pos_filtered, last_ctrl_pos, current_limit_cooldown,
                                        current_limit_recovery_start, current_limit_recovery_start_pos)
                if not temp_initialized:
                    temp_filtered[:] = temp
                    temp_initialized = True
                else:
                    filter_temp(temp, temp_filtered, temp_reject_count, CFG.temp_max_delta, CFG.temp_stuck_ticks)
                limits.check_temp(port, packet, CFG, temp_filtered, now, torque_enable, torque_was_enabled,
                                  current_limit_cooldown, sustained_temp_time)
                # Gripper torque-limit management (Torque_Limit written only on change):
                # static gentle limit in normal operation, raised to the cal value while
                # calibrating so the open/close sweeps reach the stop-detection current.
                desired_gripper_tl = gripper_cal_tl if calibrating else gripper_normal_tl
                if desired_gripper_tl != current_gripper_tl:
                    write_motors(port, packet, "Torque_Limit", np.full(len(CFG.motors), desired_gripper_tl, dtype=np.uint16), mask=gripper_tl_mask)
                    current_gripper_tl = desired_gripper_tl
                # Gripper admittance control (position mode, every tick)
                if torque_enable[g] and CFG.gripper_compliance:
                    # Force feedback from motor current
                    g_force = cs[g] * CFG.kt[g]
                    # Virtual mass-spring-damper: spring pulls toward target, current is external force
                    g_acc = (CFG.gripper_stiffness * (pos_filtered[g] - g_ref) - CFG.gripper_damping * g_vel + g_force) / CFG.gripper_mass
                    g_vel += g_acc * CFG.dt
                    g_ref += g_vel * CFG.dt
                    # Write admittance reference as goal position
                    g_goal = np.floor(ps_raw[g]) + g_ref + ps0[g] - turns[g]
                    g_mask = np.zeros(len(CFG.motors), dtype=np.bool_); g_mask[g] = True
                    g_cmd = np.full(len(CFG.motors), g_goal, dtype=np.float32)
                    write_motors(port, packet, "Goal_Position", g_cmd, mask=g_mask)
                # Compliance: wrench estimation + dynamics at state tick rate
                if compliance_mode and not np.any(tau_mode):
                    print(f"COMPLIANCE ACTIVE: running step", flush=True)
                    comp_joints = comp_ctrl.step(ps_acc, vs_filtered, cs)
                    if comp_joints is not None:
                        pos_filtered[:7] = comp_joints[:7]

            if r_ctrl.ready():
                raw_ctrl = r_ctrl.data['pos'].copy()
                target_pos = limits.clip_target(raw_ctrl, ps_acc)
                if RANGE_LO is not None:
                    target_pos = limits.clip_range(target_pos, RANGE_LO, RANGE_HI)
                last_ctrl_pos[:] = target_pos
                recovering_mask = current_limit_recovery_start > 0
                saved_recovery = pos_filtered.copy()
                pos_filtered = CFG.lpf_alpha * target_pos + (1 - CFG.lpf_alpha) * pos_filtered
                pos_filtered[recovering_mask] = saved_recovery[recovering_mask]
                if compliance_mode and comp_ctrl.initialized:
                    comp_ctrl.set_target(target_pos)
                # tau_mode joints: write current from ctrl tau
                if np.any(tau_mode):
                    tau_cmd[:] = r_ctrl.data['tau']
                    hcl_arr = np.asarray(CFG.hard_current_limit, dtype=np.float32)
                    current_cmd[:] = np.clip(tau_cmd / np.asarray(CFG.kt, dtype=np.float32), -hcl_arr, hcl_arr)
                    tau_mask = torque_enable & tau_mode
                    if CFG.gripper_compliance:
                        tau_mask[g] = False  # gripper handled separately via admittance
                    tau_mask &= hls_mask
                    if np.any(tau_mask):
                        write_motors(port, packet, "Target_Current", current_cmd, mask=tau_mask)
                pos_mask = torque_enable & ~tau_mode
                if CFG.gripper_compliance:
                    pos_mask[g] = False  # gripper goal written by admittance loop
                if np.any(pos_mask):
                    goal_raw = np.floor(ps_raw) + pos_filtered + ps0 - turns
                    goal_raw[0] = pos_filtered[0] + ps0[0]
                    # J0 software integrator: accumulate the residual hold error and bias the
                    # position command so the servo drives J0's steady-state error -> 0. Engages
                    # only while the J0 setpoint is ~stationary (holding) so fast moves don't wind
                    # it up; clamped for anti-windup; reset on disable. See J0_SOFTWARE_INTEGRATOR.md.
                    if j0_integ_ki > 0.0 and pos_mask[0]:
                        j0_err = float(pos_filtered[0] - ps_acc[0])
                        cmd_stationary = j0_cmd_prev is not None and abs(float(pos_filtered[0]) - j0_cmd_prev) < j0_integ_gate
                        if cmd_stationary and abs(j0_err) < j0_integ_band:
                            j0_integ += j0_integ_ki * j0_err * CFG.dt
                            j0_integ = float(np.clip(j0_integ, -j0_integ_clamp, j0_integ_clamp))
                        j0_cmd_prev = float(pos_filtered[0])
                        goal_raw[0] = pos_filtered[0] + ps0[0] + j0_integ
                    else:
                        j0_integ = 0.0
                        j0_cmd_prev = None
                    # J7 (gripper) current-relief: keep a held grasp from overheating. When the jaw
                    # is held still (|vel|<vel_stop for hold_time) and drawing more than i_hold, walk
                    # the goal toward the actual position -> shrinks the servo error -> sheds current
                    # (heat) down to i_hold. If current drops below i_hold (grip slipping), relax the
                    # backoff to re-grip. goal_raw[g] holds the turn-tracked command; we bias it.
                    if j7_relief_enable and pos_mask[g]:
                        gc = float(pos_filtered[g])
                        if j7_cmd_prev is None or abs(gc - j7_cmd_prev) > j7_relief_gate:
                            j7_bias = 0.0; j7_stall_t = 0.0   # new grasp/move -> re-seat before relieving
                        j7_cmd_prev = gc
                        if abs(float(vs_filtered[g])) > j7_relief_vel_stop:
                            j7_stall_t = 0.0
                        else:
                            j7_stall_t += CFG.dt
                        icur = abs(float(cs[g]))
                        err_eff = float(pos_filtered[g] - ps_acc[g]) + j7_bias   # effective (command+bias) - actual
                        if j7_stall_t >= j7_relief_hold_time and icur > j7_relief_i_hold + j7_relief_db:
                            j7_bias -= j7_relief_step * float(np.sign(err_eff))  # shrink error -> shed current
                        elif icur < j7_relief_i_hold - j7_relief_db:
                            j7_bias -= j7_relief_step * float(np.sign(j7_bias))  # restore grip (relax backoff)
                        j7_bias = float(np.clip(j7_bias, -j7_relief_bias_max, j7_relief_bias_max))
                        goal_raw[g] = goal_raw[g] + j7_bias
                        if j7_relief_debug:
                            j7_dbg += 1
                            if j7_dbg % 50 == 0:   # ~3 Hz at the 150 Hz ctrl tick
                                print(f"[j7relief] cmd={gc:+.3f} act={float(ps_acc[g]):+.3f} "
                                      f"bias={j7_bias:+.4f} goal={float(goal_raw[g]):+.3f} "
                                      f"cur={float(cs[g]):+.2f}A stall={j7_stall_t:.2f}s "
                                      f"vel={float(vs_filtered[g]):+.3f}", flush=True)
                    else:
                        j7_bias = 0.0; j7_stall_t = 0.0; j7_cmd_prev = None
                    write_motors(port, packet, "Goal_Position", goal_raw, mask=pos_mask)
            
            with w_state.buf() as b:
                b['pos'] = ps_acc
                b['vel'] = vs_filtered
                b['torque'] = cs * CFG.kt
                b['temp'] = temp_filtered
                b['current'] = cs
    port.closePort()
