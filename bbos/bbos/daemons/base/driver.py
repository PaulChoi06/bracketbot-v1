"""Shared baseboard serial protocol for the daemon, calibration, and diagnostics."""
import errno
import math
import os
import pathlib
import select
import struct
import time
from typing import NamedTuple

COMMAND_SYNC = 0xA55A
STATUS_SYNC = 0x5AA5
COMMAND_TYPE = 1
STATUS_TYPE = 2
CONFIG_TYPE = 3
DEFAULT_MODE_TYPE = 6
IMU_DIAG_TYPE = 7
IMU_DIAG_FIELDS = ('schema', 'snapshot_ms', 'odr_hz', 'polls', 'not_ready', 'raw_reads', 'accepted', 'filter_updates', 'bus_errors', 'data_rejected', 'filter_failed', 'forward_polls', 'forward_samples', 'dt_min_cycles', 'dt_max_cycles', 'estimated_missing', 'read_cycles_max', 'filter_cycles_max', 'filter_cycles_lo', 'filter_cycles_hi', 'pitch_offset_bits', 'beta_bits', 'settling', 'diag_dropped')

COMMAND_STRUCT = struct.Struct("<HBBII5fH")
CONFIG_STRUCT = struct.Struct("<HBBII2f2I3f3fffIf3I2fH")  # 90 bytes
DEFAULT_MODE_STRUCT = struct.Struct("<HBBIIH")          # 14 bytes

# Type 4 sends ODrive CAN frames; type 5 returns only frames requested by the host.
CANRELAY_TYPE = 4
CANFRAME_TYPE = 5
CANRELAY_BYTES = 26


# host flags basically means what bit corresponds to what message when jetson sends messages to the baseboard
# Ex. 0b00000101 means host is requesting arming and live torque (bits 0 and 2 are set)
HOST_FLAG_ARM_REQUEST = 1 << 0
HOST_FLAG_TWIST_FF_TUNE = 1 << 1
HOST_FLAG_LIVE_TORQUE = 1 << 2
HOST_FLAG_LEAN_MODE = 1 << 3
# Drive on wheel-velocity control instead of balancing.
HOST_FLAG_TWIST_MODE = 1 << 4
HOST_FLAG_CALIB_MODE = 1 << 5
HOST_FLAG_RUN_CALIB = 1 << 6
HOST_FLAG_RESET_CMD_INTEGRAL = 1 << 7
HOST_FLAG_TWIST_YAW_FF_TUNE = 1 << 8
HOST_FLAG_TWIST_PROFILE_TUNE_A = 1 << 9
HOST_FLAG_TWIST_PROFILE_TUNE_B = 1 << 10
HOST_FLAG_SET_RUNTIME_MODE = 1 << 11

#Basically describes the structure of the messages
#IMU is 9 floats, Iq is 2 floats, od status is 2 ints, 2 floats, 3 ints, etc
#all of these get sent in one status packet per control loop so each loop sends STATUS_BYTES bytes of data to the jetson
STATUS_PREFIX = struct.Struct("<HBB6I10f")   # header .. torque_axis_1_nm
STATUS_TAIL = struct.Struct("<2I2f3I")       # axis errors, vbus/ibus, CAN counters
STATUS_IMU = struct.Struct("<9f")            # accel[3], gyro[3], rpy[3]
STATUS_IQ = struct.Struct("<2f")             # measured phase current per axis
STATUS_BYTES = 142

FLAG_TORQUE_ACTIVE = 1 << 19
STATUS_CONFIG_VALID = 1 << 20
STATUS_LEAN_ACTIVE = 1 << 21
STATUS_RUNTIME_MODE_SHIFT = 28
STATUS_RUNTIME_MODE_MASK = 3 << STATUS_RUNTIME_MODE_SHIFT

# Arming prerequisites, in bit order, matching axis1_arm_prerequisites() in control.c. 
FLAG_NAMES = {
    0: "LINK_READY", 3: "ENC0_FRESH", 4: "ENC1_FRESH",
    5: "HEARTBEATS_OK", 6: "PITCH_SAFE", 7: "POLICY_RAN", 11: "IMU_READY",
    12: "IMU_FRESH", 14: "ARM_REQUESTED", 17: "CAN_TX_OK",
    20: "CONFIG_VALID",
}


STATUS_CALIB_ACTIVE = 1 << 23
STATUS_CALIB_OK = 1 << 24
STATUS_CALIB_FAIL = 1 << 25


def calibration_state(flags):
    """'active' | 'ok' | 'failed' | None, from a status packet's flags."""
    if flags & STATUS_CALIB_ACTIVE:
        return "active"
    if flags & STATUS_CALIB_OK:
        return "ok"
    if flags & STATUS_CALIB_FAIL:
        return "failed"
    return None


def why_not_armed(flags):
    """Return unmet arming prerequisites; an empty list means all are satisfied."""
    return [n for b, n in FLAG_NAMES.items() if not (flags & (1 << b))]


def _crc16_table():
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC16 = _crc16_table()


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16[((crc >> 8) ^ byte) & 0xFF]
    return crc


def imu_filter_registers(bandwidth_hz, order):
    """Encode readable filter settings for the fixed 2 kHz ICM-42688 path."""
    bandwidth_hz = int(bandwidth_hz)
    order = int(order)
    if bandwidth_hz != 500:
        raise ValueError(
            "IMU filter bandwidth must be 500 Hz at the fixed 2 kHz sample rate")
    if order not in (1, 2, 3):
        raise ValueError("IMU filter order must be 1, 2, or 3")
    order_code = order - 1
    gyro_config1 = (0x16 & ~0x0C) | (order_code << 2)
    gyro_accel_config0 = 0x11
    accel_config1 = (0x0D & ~0x18) | (order_code << 3)
    return gyro_config1, gyro_accel_config0, accel_config1


class Status(NamedTuple):
    """One decoded status packet, with the flag questions answered by name."""
    flags: int
    tick: int
    missed: int
    received_sequence: int
    exec_cycles: int
    pos0: float
    vel0: float
    pos1: float
    vel1: float
    pitch: float
    act0: float
    act1: float
    torque1: float
    ax0_err: int
    ax1_err: int
    vbus: float
    hb_stale: int
    rx_lost: int
    fifo_full: int
    accel: tuple
    gyro: tuple
    rpy: tuple
    iq: tuple

    @property
    def live(self) -> bool:
        return bool(self.flags & FLAG_TORQUE_ACTIVE)

    @property
    def config_valid(self) -> bool:
        return bool(self.flags & STATUS_CONFIG_VALID)

    @property
    def lean_active(self) -> bool:
        return bool(self.flags & STATUS_LEAN_ACTIVE)

    @property
    def runtime_mode(self) -> int:
        return (self.flags & STATUS_RUNTIME_MODE_MASK) >> STATUS_RUNTIME_MODE_SHIFT

    @property
    def missing(self):
        return why_not_armed(self.flags)


class Link:
    """Framed serial link to the STM. Owns the port exclusively."""

    def __init__(self, port, baud, expected_version):
        self.port, self.baud = port, baud
        self.expected_version = expected_version
        self.buf = bytearray()
        self._can_buf = bytearray()
        self.crc_errors = 0
        self.imu_diag = None
        self.sequence = time.monotonic_ns() & 0xFFFFFFFF
        self.reconnects = 0
        self._open()

    def _open(self):
        import termios
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[4] = attrs[5] = termios.B1000000
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        self.buf.clear()
        self._can_buf.clear()

    def reopen(self):
        """Reconnect after USB re-enumeration or a target reset invalidates the fd."""
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = -1
        while True:
            try:
                self._open()
                self.reconnects += 1
                print("[base] serial reopened (%s), reconnect #%d"
                      % (self.port, self.reconnects), flush=True)
                return
            except OSError as exc:
                print("[base] reopen failed (%s); retrying in 2s" % exc, flush=True)
                time.sleep(2.0)

    def send_config(self, cfg):
        """Write one per-robot config packet to the baseboard."""
        sign_left = float(cfg["sign_dir_left"])
        sign_right = float(cfg["sign_dir_right"])
        node_left = int(cfg["node_left"])
        node_right = int(cfg["node_right"])
        gyro_bias = tuple(float(v) for v in cfg["gyro_bias"])
        pitch_limit = float(cfg["pitch_limit_rad"])
        action_scale = float(cfg["action_scale_nm"])
        wheel_radius = float(cfg["wheel_radius_m"])
        pitch_offset = float(cfg["pitch_offset_rad"])
        robot_width = float(cfg["robot_width_m"])
        default_mode = int(cfg["default_mode"])
        lean_angle = float(cfg["lean_angle_deg"])
        imu_gyro_config1 = int(cfg["imu_gyro_config1"])
        imu_gyro_accel_config0 = int(cfg["imu_gyro_accel_config0"])
        imu_accel_config1 = int(cfg["imu_accel_config1"])
        imu_filter_beta = float(cfg["imu_filter_beta"])
        imu_filter_settle_beta = float(cfg["imu_filter_settle_beta"])
        finite = gyro_bias + (
            pitch_limit, action_scale, wheel_radius, pitch_offset, robot_width,
            lean_angle, imu_filter_beta, imu_filter_settle_beta,
        )
        if sign_left not in (-1.0, 1.0) or sign_right not in (-1.0, 1.0):
            raise ValueError("wheel signs must each be -1 or +1")
        if {node_left, node_right} != {0, 1}:
            raise ValueError("left/right nodes must be distinct values 0 and 1")
        if len(gyro_bias) != 3 or not all(math.isfinite(v) for v in finite):
            raise ValueError("baseboard config contains non-finite values")
        if sum(v * v for v in gyro_bias) > 0.05 * 0.05:
            raise ValueError("gyro bias magnitude exceeds 0.05 rad/s")
        if not (0.0 < pitch_limit < 1.6):
            raise ValueError("pitch limit must be between 0 and 1.6 rad")
        if not (0.0 < action_scale <= 20.0):
            raise ValueError("action scale must be in (0, 20] Nm")
        if not (0.01 < wheel_radius < 1.0):
            raise ValueError("wheel radius must be between 0.01 and 1.0 m")
        if not (-0.35 < pitch_offset < 0.35):
            raise ValueError("pitch offset must be between -0.35 and 0.35 rad")
        if not (0.05 < robot_width < 2.0):
            raise ValueError("robot width must be between 0.05 and 2.0 m")
        if default_mode not in (0, 1, 2):
            raise ValueError(
                "default mode must be BALANCE(0), LEAN(1), or TWIST(2)")
        if not (1.0 <= lean_angle <= 15.0):
            raise ValueError("lean angle must be between 1 and 15 degrees")
        filter_registers = (
            imu_gyro_config1, imu_gyro_accel_config0, imu_accel_config1,
        )
        if not all(0 <= value <= 0xFF for value in filter_registers):
            raise ValueError("IMU filter register values must fit in one byte")
        if not (0.0 <= imu_filter_beta <= 1.0 and
                0.0 <= imu_filter_settle_beta <= 1.0):
            raise ValueError("IMU filter beta values must be between 0 and 1")

        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        body = CONFIG_STRUCT.pack(
            COMMAND_SYNC, self.expected_version, CONFIG_TYPE,
            self.sequence, 0,
            sign_left, sign_right, node_left, node_right,
            gyro_bias[0], gyro_bias[1], gyro_bias[2],
            pitch_limit, action_scale, wheel_radius,
            pitch_offset, robot_width, default_mode, lean_angle,
            imu_gyro_config1, imu_gyro_accel_config0, imu_accel_config1,
            imu_filter_beta, imu_filter_settle_beta, 0)[:-2]
        return self._write(body + struct.pack("<H", crc16_ccitt(body)))

    def set_default_mode(self, mode):
        """Persist the fallback/power-on mode without changing runtime mode."""
        mode = int(mode)
        if mode not in (0, 1, 2):
            raise ValueError(
                "default mode must be BALANCE(0), LEAN(1), or TWIST(2)")
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        sequence = self.sequence
        body = DEFAULT_MODE_STRUCT.pack(
            COMMAND_SYNC, self.expected_version, DEFAULT_MODE_TYPE,
            sequence, mode, 0)[:-2]
        packet = body + struct.pack("<H", crc16_ccitt(body))
        writes = 0
        for _ in range(5):
            writes += bool(self._write(packet))
            time.sleep(0.02)
        if writes == 0:
            raise OSError("could not send default mode to the baseboard")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            for status in self.read(min(0.1, deadline - time.monotonic())):
                if (status.received_sequence == sequence and
                        status.config_valid):
                    return
        raise OSError("baseboard did not confirm the persisted default mode")

    def apply_config(self, cfg):
        """Stand down before pushing config so wheel signs never change under torque."""
        self.stand_down()
        sequence = self.send_config_burst(cfg)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            for status in self.read(min(0.1, deadline - time.monotonic())):
                if (status.received_sequence == sequence and
                        status.config_valid):
                    return
        raise OSError("baseboard did not confirm the persisted config")

    def stand_down(self):
        """Latch firmware maintenance mode until normal live control resumes."""
        stand_down_writes = 0
        for _ in range(8):
            stand_down_writes += bool(self.send(0.0, 0.0, calib=True))
            time.sleep(0.02)
        if stand_down_writes == 0:
            raise OSError("could not send a non-live command to the baseboard")

    def send_config_burst(self, cfg):
        """Send redundant config packets during an explicit maintenance action."""
        config_writes = 0
        for _ in range(5):
            config_writes += bool(self.send_config(cfg))
            time.sleep(0.02)
        if config_writes == 0:
            raise OSError("could not send config to the baseboard")
        return self.sequence

    def _write(self, data):
        try:
            return os.write(self.fd, data) == len(data)
        except BlockingIOError:
            # EAGAIN is USB backpressure; drop this setpoint instead of reopening.
            return False
        except OSError as exc:
            if exc.errno == errno.EAGAIN:
                return False
            print("[base] write failed (%s) -- reopening" % exc, flush=True)
            self.reopen()
            return False

    def send(self, v, w, runtime_mode=None, lean_deg=0.0,
             calib=False, run_calib=False, reset_cmd_integral=False,
             live=False, torque_limit_nm=0.0, ff_tau_s_nm=None,
             ff_tau_k_nm=None, ff_yaw_nm=None, profile_amax=None,
             profile_jmax_acc=None, profile_jmax_dec=None,
             profile_enabled=None):
        """Send a command with one optional live tuning payload."""
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        # Calibration keeps the board disarmed while the host controls ODrive.
        flags = HOST_FLAG_CALIB_MODE if calib else HOST_FLAG_ARM_REQUEST
        torque_limit_nm = float(torque_limit_nm)
        if live:
            if calib:
                raise ValueError("live torque is not allowed in calibration mode")
            if not torque_limit_nm > 0.0:
                raise ValueError(
                    "live torque requires a positive torque_limit_nm")
            flags |= HOST_FLAG_LIVE_TORQUE
        else:
            torque_limit_nm = 0.0
        if run_calib:
            # Firmware edge-detects this flag and calibrates once while disarmed.
            flags |= HOST_FLAG_RUN_CALIB
        if reset_cmd_integral:
            flags |= HOST_FLAG_RESET_CMD_INTEGRAL
        mode_value_0 = 0.0
        mode_value_1 = 0.0
        if (ff_tau_s_nm is None) != (ff_tau_k_nm is None):
            raise ValueError("both feedforward torque values must be provided")
        if (profile_amax is None) != (profile_jmax_acc is None):
            raise ValueError("profile acceleration and acceleration jerk must be provided together")
        if (profile_jmax_dec is None) != (profile_enabled is None):
            raise ValueError("profile deceleration jerk and enable must be provided together")
        if runtime_mode is not None:
            runtime_mode = int(runtime_mode)
            if runtime_mode not in (0, 1, 2):
                raise ValueError(
                    "runtime mode must be BALANCE(0), LEAN(1), or TWIST(2)")
            flags |= HOST_FLAG_SET_RUNTIME_MODE
            if runtime_mode == 1:
                mode_value_0 = float(lean_deg)
                if not math.isfinite(mode_value_0) or not 1.0 <= mode_value_0 <= 15.0:
                    raise ValueError("lean angle must be between 1 and 15 degrees")
                flags |= HOST_FLAG_LEAN_MODE
            elif runtime_mode == 2:
                flags |= HOST_FLAG_TWIST_MODE
        if ff_tau_s_nm is not None:
            mode_value_0 = float(ff_tau_s_nm)
            mode_value_1 = float(ff_tau_k_nm)
            if (not math.isfinite(mode_value_0) or
                    not math.isfinite(mode_value_1) or
                    not 0.0 <= mode_value_0 <= 20.0 or
                    not 0.0 <= mode_value_1 <= 20.0):
                raise ValueError("feedforward torque values must be in [0, 20] Nm")
            flags |= HOST_FLAG_TWIST_FF_TUNE
        elif ff_yaw_nm is not None:
            mode_value_0 = float(ff_yaw_nm)
            mode_value_1 = 0.0
            if not math.isfinite(mode_value_0) or not 0.0 <= mode_value_0 <= 20.0:
                raise ValueError("yaw feedforward torque must be in [0, 20] Nm")
            flags |= HOST_FLAG_TWIST_YAW_FF_TUNE
        elif profile_amax is not None:
            mode_value_0 = float(profile_amax)
            mode_value_1 = float(profile_jmax_acc)
            if (not math.isfinite(mode_value_0) or
                    not math.isfinite(mode_value_1) or
                    not 0.05 <= mode_value_0 <= 20.0 or
                    not 0.1 <= mode_value_1 <= 200.0):
                raise ValueError("profile acceleration/jerk is out of range")
            flags |= HOST_FLAG_TWIST_PROFILE_TUNE_A
        elif profile_jmax_dec is not None:
            mode_value_0 = float(profile_jmax_dec)
            mode_value_1 = 1.0 if profile_enabled else 0.0
            if (not math.isfinite(mode_value_0) or
                    not 0.1 <= mode_value_0 <= 200.0):
                raise ValueError("profile deceleration jerk is out of range")
            flags |= HOST_FLAG_TWIST_PROFILE_TUNE_B
        body = COMMAND_STRUCT.pack(
            COMMAND_SYNC, self.expected_version, COMMAND_TYPE,
            self.sequence, flags,
            float(v), float(w), torque_limit_nm,
            mode_value_0, mode_value_1, 0)[:-2]
        return self._write(body + struct.pack("<H", crc16_ccitt(body)))

    # ---- CAN relay -------------------------------------------------------

    def can_send(self, can_id, data=b"", dlc=None):
        """Put a raw frame on the ODrive bus for protocols such as RxSdo."""
        data = bytes(data)
        if dlc is None:
            dlc = len(data)
        dlc = int(dlc)
        if len(data) > 8 or not 0 <= dlc <= 8 or len(data) > dlc:
            raise ValueError("CAN payload and DLC must fit in one 8-byte frame")
        payload = data + b"\x00" * (8 - len(data))
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        pkt = bytearray(CANRELAY_BYTES)
        struct.pack_into("<HBBII", pkt, 0, COMMAND_SYNC, self.expected_version,
                         CANRELAY_TYPE, self.sequence, can_id)
        pkt[12] = dlc
        pkt[16:24] = payload
        struct.pack_into("<H", pkt, CANRELAY_BYTES - 2,
                         crc16_ccitt(bytes(pkt[:CANRELAY_BYTES - 2])))
        self._write(bytes(pkt))

    # ---- ODrive CANSimple helpers ---------------------------------------
    # Calibration-only CANSimple commands; other commands use can_send directly.
    CMD_SET_AXIS_STATE = 0x007
    CMD_SET_CONTROLLER_MODE = 0x00B
    CMD_SET_INPUT_VEL = 0x00D
    CMD_CLEAR_ERRORS = 0x018
    CMD_SET_VEL_GAINS = 0x01B

    AXIS_STATE_IDLE = 1
    AXIS_STATE_CLOSED_LOOP = 8

    def _odrive(self, node, cmd, data=b""):
        self.can_send((int(node) << 5) | cmd, data)

    def odrive_request_state(self, node, state):
        self._odrive(node, self.CMD_SET_AXIS_STATE,
                     struct.pack("<I", int(state)))

    def odrive_set_velocity_mode(self, node):
        """control_mode=VELOCITY(2), input_mode=PASSTHROUGH(1)."""
        self._odrive(node, self.CMD_SET_CONTROLLER_MODE,
                     struct.pack("<II", 2, 1))

    def odrive_set_torque_mode(self, node):
        """control_mode=TORQUE(1) -- what balancing runs in."""
        self._odrive(node, self.CMD_SET_CONTROLLER_MODE,
                     struct.pack("<II", 1, 1))

    def odrive_set_velocity(self, node, turns_s, torque_ff=0.0):
        self._odrive(node, self.CMD_SET_INPUT_VEL,
                     struct.pack("<ff", float(turns_s), float(torque_ff)))

    def odrive_set_vel_gains(self, node, vel_gain, vel_integrator_gain):
        """Set live velocity gains through CANSimple, which does not require IDLE."""
        self._odrive(node, self.CMD_SET_VEL_GAINS,
                     struct.pack("<ff", float(vel_gain),
                                 float(vel_integrator_gain)))

    def odrive_clear_errors(self, node):
        self._odrive(node, self.CMD_CLEAR_ERRORS)

    def can_frames(self, timeout=0.0):
        """Return relayed ``(can_id, dlc, data)`` frames while preserving status."""
        if timeout > 0.0:
            self.read(timeout)          # pulls bytes in; sorts them by type
        out = []
        buf = self._can_buf
        sync = struct.pack("<H", STATUS_SYNC)
        while len(buf) >= CANRELAY_BYTES:
            i = buf.find(sync)
            if i < 0:
                del buf[:-1]
                break
            if i:
                del buf[:i]
            if len(buf) < CANRELAY_BYTES:
                break
            pkt = bytes(buf[:CANRELAY_BYTES])
            if pkt[3] != CANFRAME_TYPE:
                del buf[:2]
                continue
            want = struct.unpack_from("<H", pkt, CANRELAY_BYTES - 2)[0]
            if crc16_ccitt(pkt[:CANRELAY_BYTES - 2]) != want:
                del buf[:2]
                continue
            del buf[:CANRELAY_BYTES]
            out.append((struct.unpack_from("<I", pkt, 8)[0], pkt[12],
                        pkt[16:16 + min(pkt[12], 8)]))
        return out

    def read(self, timeout):
        """Wait up to ``timeout`` for data, then drain all available packets."""
        first = True
        while True:
            try:
                readable, _, _ = select.select([self.fd], [], [],
                                               timeout if first else 0.0)
                first = False
                if not readable:
                    break
                chunk = os.read(self.fd, 4096)
            except BlockingIOError:
                # EAGAIN after select is a spurious wakeup or port contention.
                break
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    break
                print("[base] read failed (%s) -- reopening" % exc, flush=True)
                self.reopen()
                return []
            if not chunk:
                break
            self.buf.extend(chunk)

        out = []
        sync = struct.pack("<H", STATUS_SYNC)
        while len(self.buf) >= 4:
            start = self.buf.find(sync)
            if start < 0:
                del self.buf[:-1]
                break
            if start:
                del self.buf[:start]
            if len(self.buf) < 4:
                break
            packet_type = self.buf[3]
            if packet_type == CANFRAME_TYPE:
                if len(self.buf) < CANRELAY_BYTES:
                    break
                self._can_buf.extend(self.buf[:CANRELAY_BYTES])
                del self.buf[:CANRELAY_BYTES]
                continue
            if packet_type == IMU_DIAG_TYPE:
                if len(self.buf) < 8: break
                count = struct.unpack_from("<I", self.buf, 4)[0]
                if count != len(IMU_DIAG_FIELDS):
                    del self.buf[:1]; continue
                length = 10 + count * 4
                if len(self.buf) < length: break
                packet = bytes(self.buf[:length])
                if crc16_ccitt(packet[:-2]) != struct.unpack_from("<H", packet, length-2)[0]:
                    self.crc_errors += 1; del self.buf[:1]; continue
                del self.buf[:length]
                if packet[2] == self.expected_version:
                    self.imu_diag = struct.unpack_from("<" + "I"*count, packet, 8)
                continue
            if packet_type != STATUS_TYPE:
                del self.buf[:1]
                continue
            if len(self.buf) < STATUS_BYTES:
                break
            packet = bytes(self.buf[:STATUS_BYTES])
            want = struct.unpack_from("<H", packet, STATUS_BYTES - 2)[0]
            if crc16_ccitt(packet[:-2]) != want:
                # Sync may appear in payloads; resync by one byte and validate headers.
                if (packet[2] == self.expected_version
                        and packet[3] == STATUS_TYPE):
                    self.crc_errors += 1
                del self.buf[:1]
                continue
            del self.buf[:STATUS_BYTES]
            head = STATUS_PREFIX.unpack_from(packet, 0)
            if head[1] != self.expected_version or head[2] != STATUS_TYPE:
                continue
            tail = STATUS_TAIL.unpack_from(packet, STATUS_PREFIX.size)
            imu = STATUS_IMU.unpack_from(packet,
                                         STATUS_PREFIX.size + STATUS_TAIL.size)
            iq = STATUS_IQ.unpack_from(
                packet, STATUS_PREFIX.size + STATUS_TAIL.size + STATUS_IMU.size)
            out.append(Status(
                flags=head[4], tick=head[5], missed=head[6],
                received_sequence=head[7], exec_cycles=head[8],
                pos0=head[9], vel0=head[10], pos1=head[11], vel1=head[12],
                pitch=head[13], act0=head[16], act1=head[17], torque1=head[18],
                ax0_err=tail[0], ax1_err=tail[1], vbus=tail[2],
                hb_stale=tail[4], rx_lost=tail[5], fifo_full=tail[6],
                accel=imu[0:3], gyro=imu[3:6], rpy=imu[6:9], iq=iq[0:2]))
        return out

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


def load_calibration(path):
    """Parse the flat YAML written by calibrate.py; a missing file is allowed."""
    out = {}
    if not path:
        return out
    try:
        text = pathlib.Path(path).read_text()
    except (FileNotFoundError, NotADirectoryError):
        return out
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        try:
            out[k.strip()] = float(v) if "." in v or "e" in v.lower() else int(v)
        except ValueError:
            out[k.strip()] = v.strip().strip("\'\"")
    return out


def resolve_gyro_bias(cfg, calib=None):
    """Read the final or temporary bootstrap bias from config.yaml."""
    if calib is None:
        calib = load_calibration(getattr(cfg, "calibration_path", None))
    final = ("gyro_bias_x", "gyro_bias_y", "gyro_bias_z")
    bootstrap = ("bootstrap_gyro_bias_x", "bootstrap_gyro_bias_y",
                 "bootstrap_gyro_bias_z")
    for axes in (final, bootstrap):
        if not all(key in calib for key in axes):
            continue
        gyro_bias = [float(calib[key]) for key in axes]
        if not all(abs(b) < 0.05 for b in gyro_bias):
            raise ValueError(
                "implausible gyro bias in %s: %s" % (cfg.calibration_path, gyro_bias))
        return gyro_bias, cfg.calibration_path
    raise SystemExit(
        f"[base] no IMU calibration in {cfg.calibration_path}.\n"
        "  This robot has not been calibrated. Put it on a stand and run:\n"
        "    calibrate base\n"
        "  Refusing to start rather than use an unknown gyro bias."
    )


def build_config(cfg=None, drive=None):
    """Build the shared per-robot configuration packet."""
    if cfg is None or drive is None:
        from bbos import Config
        if cfg is None:
            cfg = Config("base")
        if drive is None:
            drive = Config("drive")
    # calibrate.py's output wins where it exists.
    calib = load_calibration(getattr(cfg, "calibration_path", None))
    gyro_bias, bias_src = resolve_gyro_bias(cfg, calib)
    # Wheel signs are measured per robot; an unsafe default could invert feedback.
    missing = [k for k in ("sign_dir_left", "sign_dir_right") if k not in calib]
    if missing:
        raise ValueError(
            "no wheel signs in %s (missing %s) -- run `calibrate base` option 4"
            % (cfg.calibration_path, ", ".join(missing)))
    imu_registers = imu_filter_registers(
        cfg.imu_filter_bandwidth_hz, cfg.imu_filter_order)
    return {
        "sign_dir_left": float(calib["sign_dir_left"]),
        "sign_dir_right": float(calib["sign_dir_right"]),
        "node_left": cfg.left_axis,
        "node_right": cfg.right_axis,
        "gyro_bias": gyro_bias,
        "bias_src": bias_src,
        "pitch_limit_rad": cfg.pitch_limit_rad,
        "action_scale_nm": cfg.action_scale_nm,
        "wheel_radius_m": cfg.wheel_diam / 2.0,
        # Missing mounting correction safely defaults to zero and is logged.
        "pitch_offset_rad": float(calib.get("pitch_offset_rad", 0.0)),
        # Full track width, not the moment arm.
        "robot_width_m": drive.robot_width,
        "default_mode": int(cfg.default_mode),
        "lean_angle_deg": float(cfg.lean_angle_deg),
        "imu_filter_bandwidth_hz": int(cfg.imu_filter_bandwidth_hz),
        "imu_filter_order": int(cfg.imu_filter_order),
        "imu_gyro_config1": imu_registers[0],
        "imu_gyro_accel_config0": imu_registers[1],
        "imu_accel_config1": imu_registers[2],
        "imu_filter_beta": float(cfg.imu_filter_beta),
        "imu_filter_settle_beta": float(cfg.imu_filter_settle_beta),
    }


def describe_config(c):
    """One-line provenance for the log: which robot's numbers went to the board."""
    return (f"signs=({c['sign_dir_left']:+.0f},{c['sign_dir_right']:+.0f}) "
            f"nodes=({c['node_left']},{c['node_right']}) "
            f"bias={c['gyro_bias']} "
            f"(from {os.path.basename(c.get('bias_src', '?'))}) "
            f"pitch_lim={c['pitch_limit_rad']:.3f}rad "
            f"pitch_off={c['pitch_offset_rad']:+.4f}rad "
            f"default={['BALANCE', 'LEAN', 'TWIST'][c['default_mode']]}"
            f"@{c['lean_angle_deg']:g}deg "
            f"imu_filter={c['imu_filter_bandwidth_hz']}Hz/"
            f"order{c['imu_filter_order']} "
            f"imu_beta={c['imu_filter_beta']:g}/"
            f"{c['imu_filter_settle_beta']:g}")


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x
