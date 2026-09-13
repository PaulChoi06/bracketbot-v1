"""Configuration for the baseboard that provides the robot's drive and IMU topics.

Baseboard-only controls and diagnostics use the ``base.`` topic prefix.
"""
from bbos import register, realtime

from pathlib import Path

import numpy as np


@register
class base:
    # Stable udev name matching the ttyARMLEFT/ttyARMRIGHT convention.
    port: str = "/dev/ttySTM"
    baud: int = 1000000

    # Must exceed 10 Hz because the command timeout is 100 ms.
    command_hz: float = 50.0
    drive_command_timeout_s: float = 0.1

    v_max: float = 0.3
    w_max: float = 1.0

    # Baseboard link protocol version checked by both host and STM32 firmware.
    expected_link_version: int = 9

    # Per-robot geometry and wiring sent in the runtime config packet.
    left_axis: int = 1
    right_axis: int = 0
    wheel_diam: float = 0.165
    action_scale_nm: float = 6.5

    # Twist-mode Stribeck friction feedforward. tune.py can override these live.
    ff_tau_s_nm: float = 0.8
    ff_tau_k_nm: float = 0.6
    ff_yaw_nm: float = 0.5

    twist_profile_enabled: bool = True
    twist_profile_amax: float = 2.0
    twist_profile_jmax_acc: float = 8.0
    twist_profile_jmax_dec: float = 5.0

    # Disarm beyond this pitch.
    pitch_limit_rad: float = 0.6981317

    MODE_BALANCE: int = 0
    MODE_LEAN: int = 1
    MODE_TWIST: int = 2

    default_mode: int = MODE_TWIST
    lean_angle_deg: float = 4.0
    mode_command_timeout_s: float = 0.25

    # ICM-42688-P UI filter plus Madgwick correction gains. `calibrate base`
    # option 6 writes these to the STM and persists them across power cycles.
    imu_filter_bandwidth_hz: int = 500
    imu_filter_order: int = 2
    imu_filter_beta: float = 0.006
    imu_filter_settle_beta: float = 0.12

    # Balancing policy.
    policy_onnx: str = "/home/bracketbot/bbos/bbos/daemons/base/models/balancing_terrain.onnx"

    # Must match the deployed lean policy's training range.
    lean_angle_min_deg: float = 1.0
    lean_angle_max_deg: float = 15.0

    calibration_path: str = str(Path(__file__).resolve().parent / "config.yaml")


@register
class drive:
    # Robot geometry and limits
    robot_width: float = 0.3275
    wheel_diam: float = 0.165
    max_linear_vel: float = 0.3
    max_angular_vel: float = 0.9
    # Public baseboard convention: leaning the top forward is positive.
    sign_pitch: float = 1.0


ODRIVE_WATCHDOG_TIMEOUT = 0.5

_ODRIVE_AXIS_SETTINGS = {
    "motor.config.pole_pairs": 15,
    "motor.config.torque_constant": 0.516875,
    "motor.config.calibration_current": 5.0,
    "motor.config.resistance_calib_max_voltage": 4.0,
    "motor.config.current_lim": 40.0,
    "motor.config.current_lim_margin": 5.0,
    "motor.config.requested_current_range": 60.0,
    "motor.config.current_control_bandwidth": 100.0,
    "motor.config.motor_type": 0,
    "encoder.config.mode": 1,
    "encoder.config.cpr": 90,
    "encoder.config.calib_scan_distance": 150.0,
    "encoder.config.bandwidth": 100.0,
    "encoder.config.use_index": 0,
    "controller.config.pos_gain": 1.0,
    "controller.config.vel_gain": 4.0,
    "controller.config.vel_integrator_gain": 4.5,
    "controller.config.vel_integrator_limit": 3.0,
    "controller.config.vel_lpf_bandwidth": 100.0,
    "controller.config.vel_slew_rate": 5.0,
    "controller.config.vel_deadband": 0.0,
    "controller.config.torque_slew_rate": 0.0,
    "controller.config.vel_limit": 10.0,
    "controller.config.vel_limit_tolerance": 1.2,
    "controller.config.vel_ramp_rate": 50.0,
    "controller.config.torque_ramp_rate": 0.01,
    "controller.config.enable_vel_limit": 1,
    "controller.config.spinout_electrical_power_threshold": 1000.0,
    "controller.config.spinout_mechanical_power_threshold": -1000.0,
    "controller.config.enable_overspeed_error": 1,
    "controller.config.input_mode": 1,
    "config.startup_closed_loop_control": 0,
    "config.watchdog_timeout": ODRIVE_WATCHDOG_TIMEOUT,
    "config.can.encoder_rate_ms": 2,
    "config.can.heartbeat_rate_ms": 100,
    "config.can.bus_vi_rate_ms": 100,
    "config.can.motor_error_rate_ms": 100,
    "config.can.encoder_error_rate_ms": 100,
    "config.can.controller_error_rate_ms": 100,
    "config.can.iq_rate_ms": 20,
}

ODRIVE_SETTINGS = {
    "can.config.baud_rate": 1000000,
    "can.config.protocol": 1,
    "config.gpio9_mode": 1,
    "config.gpio10_mode": 1,
    "config.gpio11_mode": 1,
    "config.gpio12_mode": 1,
    "config.gpio13_mode": 1,
    "config.gpio14_mode": 1,
    "config.brake_resistance": 2.0,
    "config.enable_brake_resistor": 1,
}
for _axis in (0, 1):
    ODRIVE_SETTINGS.update({
        f"axis{_axis}.{path}": value
        for path, value in _ODRIVE_AXIS_SETTINGS.items()
    })
    ODRIVE_SETTINGS[f"axis{_axis}.config.enable_watchdog"] = 0
ODRIVE_SETTINGS.update({
    "axis0.config.can.node_id": 0,
    "axis1.config.can.node_id": 1,
})

@register
class odrive:
    # Odrive settings
    serial_port: str = "/dev/ttyTHS1"
    baudrate: int = 115200
    timeout: int = 15
    serial_timeout: float = 0.05
    can_fault_probe_timeout: float = 0.03
    left_axis: int = 1
    right_axis: int = 0
    axis_state_closed_loop: int = 8
    dir_left: int = 1
    dir_right: int = 1
    torque_bias: float = 0.05
    watchdog_timeout: float = ODRIVE_WATCHDOG_TIMEOUT
    settings: dict = ODRIVE_SETTINGS


@realtime(ms=100)
def drive_tune():
    # Live ODrive and baseboard tuning parameters.
    return [("vel_gain", np.float32), ("vel_integrator_gain", np.float32),
            ("vel_lpf_bandwidth", np.float32), ("vel_slew_rate", np.float32),
            ("vel_deadband", np.float32), ("torque_slew_rate", np.float32),
            ("vel_ramp_rate", np.float32), ("ff_tau_s_nm", np.float32),
            ("ff_tau_k_nm", np.float32), ("ff_yaw_nm", np.float32),
            # S-curve profile parameters, which are sent to the STM and applied to the ODrive.
            ("prof_enabled", np.float32), ("prof_amax", np.float32),
            ("prof_jmax_acc", np.float32), ("prof_jmax_dec", np.float32)]


@realtime(ms=10)
def drive_ctrl():
    # twist command topic
    return [("twist", (np.float32, 2)), ("twist_torque", (np.float32, 2))]


@realtime(ms=10)
def drive_state():
    """Per-wheel state as the STM reports it. Index order is [axis0, axis1]."""
    return [
        ("pos", (np.float32, 2)),
        ("vel", (np.float32, 2)),
        ("torque", (np.float32, 2)),
        ("ff", (np.float32, 2)),
        ("ctrl", (np.float32, 2)),
        ("iq", (np.float32, 2)),
        ("gains", (np.float32, 2)),
        ("pos_estimate", np.float32),
        ("yaw_error", np.float32),
    ]


@realtime(ms=10000)
def drive_status():
    """Slow drive health: bus voltage, per-axis error codes, link rate."""
    return [
        ("voltage", np.float32),
        ("errors", (np.float32, 2)),
        ("loop_hz", np.float32),
    ]


@realtime(ms=10)
def imu_orientation():
    # rpy in the respective order of the axes, in radians. 
    return [("rpy", (np.float32, 3))]


@realtime(ms=10)
def imu_raw():
    """Raw accelerometer (m/s²) and gyroscope (rad/s), bias-corrected"""
    return [("accel", np.float32, 3), ("gyro", np.float32, 3)]


@realtime(ms=1000)
def base_health():
    # STM health diagnostics, including link CRC errors, CAN FIFO overflow, and missed deadlines
    return [
        ("control_tick", np.uint32),
        ("missed_deadlines", np.uint32),
        ("exec_cycles", np.uint32),
        ("axis0_error", np.uint32),
        ("axis1_error", np.uint32),
        ("can_rx_lost", np.uint32),
        ("can_rx_fifo_full", np.uint32),
        ("can_hb_stale", np.uint32),
        ("flags", np.uint32),
        ("link_crc_errors", np.uint32),
    ]


@realtime(ms=50)
def base_mode():
    # Sets mode and lean angle. Refer to the base class for mode values.
    return [("mode", np.uint8), ("lean_angle_deg", np.float32)]


@realtime(ms=100)
def base_cmd_int_reset():
    return [("request", np.uint8)]


@realtime(ms=1000)
def imu_diagnostics():
    """Cumulative protocol-9 IMU/forward-pass timing diagnostics."""
    return [("words", np.uint32, 24)]
