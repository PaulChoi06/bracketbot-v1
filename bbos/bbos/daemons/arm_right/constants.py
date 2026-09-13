from bbos import register, realtime, state, Config
import numpy as np
from pathlib import Path

left_cfg = Config("arm_left")
# ============================================================================
# Configs
# ============================================================================
@register
class arm_right:
    # --- Hardware & motor constants -----------------------------------------
    port = "/dev/ttyARMRIGHT"
    baudrate = 1000000
    motors = [1, 2, 3, 4, 5, 6, 7, 8]
    dof = len(motors)
    read_only = left_cfg.read_only
    kt = left_cfg.kt
    st3120_mask = left_cfg.st3120_mask

    # --- Position control ---------------------------------------------------
    Kp = left_cfg.Kp
    Kd = left_cfg.Kd
    Ki = left_cfg.Ki
    j0_integ_ki = left_cfg.j0_integ_ki        # J0 software integrator (see J0_SOFTWARE_INTEGRATOR.md)
    j0_integ_clamp = left_cfg.j0_integ_clamp
    j0_integ_gate = left_cfg.j0_integ_gate
    j0_integ_band = left_cfg.j0_integ_band
    j7_relief_enable = left_cfg.j7_relief_enable      # J7 (gripper) current-relief loop
    j7_relief_vel_stop = left_cfg.j7_relief_vel_stop
    j7_relief_hold_time = left_cfg.j7_relief_hold_time
    j7_relief_i_hold = left_cfg.j7_relief_i_hold
    j7_relief_db = left_cfg.j7_relief_db
    j7_relief_step = left_cfg.j7_relief_step
    j7_relief_bias_max = left_cfg.j7_relief_bias_max
    j7_relief_gate = left_cfg.j7_relief_gate
    j7_relief_debug = left_cfg.j7_relief_debug
    gripper_sign = 1   # right gripper motor is NOT reversed (see arm_left.gripper_sign)
    angle_resolution = [1] * dof
    return_delay_time = [0] * dof
    acceleration = [254] * dof
    operating_mode = [0] * dof
    dt = 1.0 / 150  # 150 Hz
    wheel_radius = left_cfg.wheel_radius # wheel radius of vertical stage
    lpf_alpha = left_cfg.lpf_alpha
    max_pos_diff = left_cfg.max_pos_diff
    interp_duration = left_cfg.interp_duration

    # --- Kinematics (URDF / IK) ---------------------------------------------
    urdf_path = left_cfg.urdf_path
    # Preferred posture, from the 5 clustered picks in ~/ik_preferences.json.
    # The old j3=-0.25 had the wrong sign for the posture actually chosen (j3~+0.6).
    ik = left_cfg.ik_solver(ee_link="right_eef",nominal_config=[0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0])
    joint_names = ["rj0", "rj1", "rj2", "rj3", "rj4", "rj5", "rj6", "right_left_gripper"]
    # See arm_left.ik_sign. Different from the left arm: bb1 gave the right arm its
    # own axis-sign pattern (RJ1/RJ3/RJ5 were -Z), so the mapping differs.
    # Verified joint-by-joint on bracketbot-092 with view_arms.py --control.
    ik_sign = np.array([-1, -1, -1, -1, 1, -1, 1], dtype=np.float32)

    # --- Zeroing & calibration ----------------------------------------------
    j0_increment = 6/4096  # 6 encoder ticks; sign is the "up" (toward top stop) direction
    j0_cal_step = left_cfg.j0_cal_step
    # J0 torque-mode homing (mirror of arm_left; see J0_HOLD_EXPERIMENTS.md / constants there).
    j0_cal_tau = left_cfg.j0_cal_tau
    j0_cal_tau_start = left_cfg.j0_cal_tau_start
    j0_cal_ramp_s = left_cfg.j0_cal_ramp_s
    j0_top_offset_turns = left_cfg.j0_top_offset_turns
    j0_cal_settle_s = left_cfg.j0_cal_settle_s
    j0_cal_vel_stop = left_cfg.j0_cal_vel_stop
    j0_cal_move_eps_turns = left_cfg.j0_cal_move_eps_turns
    j0_cal_min_travel_turns = left_cfg.j0_cal_min_travel_turns
    j0_cal_wrongdir_turns = left_cfg.j0_cal_wrongdir_turns
    cal_step = left_cfg.cal_step
    j1_cal_clearance = -0.1  # mirror of left: frontward (clearance) on the right arm is the negative direction
    j0_homing_timeout_s = left_cfg.j0_homing_timeout_s
    skip_startup_homing = left_cfg.skip_startup_homing
    mode_settle_retries = left_cfg.mode_settle_retries

    # ========================================================================
    # Motor limits
    # ========================================================================
    # Mirror of arm_left; values inherited via left_cfg.
    # --- J0 homing current (hard-stop detection) ----------------------------
    j0_current_limit = left_cfg.j0_current_limit

    # --- Current limits -----------------------------------------------------
    protection_current = left_cfg.protection_current
    hard_current_limit = left_cfg.hard_current_limit
    sustained_current_scale = left_cfg.sustained_current_scale
    sustained_current_time_limit = left_cfg.sustained_current_time_limit
    current_slew_limit = left_cfg.current_slew_limit
    current_limit_cooldown_s = left_cfg.current_limit_cooldown_s
    current_limit_interp_s = left_cfg.current_limit_interp_s
    overcurrent_protection_time = left_cfg.overcurrent_protection_time

    # --- Temperature limits -------------------------------------------------
    max_temperature = left_cfg.max_temperature
    software_temp_limit = left_cfg.software_temp_limit
    sustained_temp_limit = left_cfg.sustained_temp_limit
    sustained_temp_time_limit = left_cfg.sustained_temp_time_limit
    overtemp_cooldown_s = left_cfg.overtemp_cooldown_s
    temp_max_delta = left_cfg.temp_max_delta
    temp_stuck_ticks = left_cfg.temp_stuck_ticks

    # --- Torque limits ------------------------------------------------------
    max_torque_limit = left_cfg.max_torque_limit
    torque_limit = left_cfg.torque_limit

    # --- Velocity limit -----------------------------------------------------
    goal_velocity = left_cfg.goal_velocity

    # --- Protection trip condition ------------------------------------------
    unloading_condition = left_cfg.unloading_condition

    # ========================================================================
    # Control, compliance & gripper
    # ========================================================================
    tau_mode_allowed = left_cfg.tau_mode_allowed
    home = np.array([0,  0, 0, -0.25, 0, 0,  0,  0.1], dtype=np.float32)
    latency: float = 65.49
    latency_std: float = 53.65

    # Startup homing waypoints are the mirror (negation in motor-turn space) of
    # the left arm's; home (CFG.home) is appended by the consumer.
    startup_seg_durations = left_cfg.startup_seg_durations
    startup_waypoints = -left_cfg.startup_waypoints

    # --- Compliance control -------------------------------------------------
    compliance_mass = left_cfg.compliance_mass
    compliance_kp_pos = left_cfg.compliance_kp_pos
    compliance_kp_rot = left_cfg.compliance_kp_rot
    compliance_force_reg = left_cfg.compliance_force_reg
    compliance_torque_reg = left_cfg.compliance_torque_reg
    compliance_normal_axis = left_cfg.compliance_normal_axis
    ee_frame = "right_eef"

    # --- Gripper ------------------------------------------------------------
    gripper_kp = left_cfg.gripper_kp
    gripper_kd = left_cfg.gripper_kd
    gripper_mass = left_cfg.gripper_mass
    gripper_stiffness = left_cfg.gripper_stiffness
    gripper_damping = left_cfg.gripper_damping
    gripper_compliance = left_cfg.gripper_compliance
    gripper_cal_torque_limit = left_cfg.gripper_cal_torque_limit

    @staticmethod
    def q2urdf(q):
        q = q * (2 * np.pi)
        q[0] = -q[0] * arm_right.wheel_radius
        q[7] = q[7] * arm_right.gripper_sign
        q[:7] = q[:7] * arm_right.ik_sign
        return q

    @staticmethod
    def urdf2q(urdf_pos):
        q = urdf_pos.copy()
        q[:7] = q[:7] * arm_right.ik_sign         # sign is its own inverse
        q[0] = -q[0] / arm_right.wheel_radius
        q[7] = q[7] * arm_right.gripper_sign
        q = q / (2 * np.pi)
        return q
