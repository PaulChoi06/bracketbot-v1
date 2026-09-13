#!/usr/bin/env python3
"""Single maintenance and calibration entry point for the robot base."""
import argparse
import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bbos import Config, Reader  # noqa: E402
import driver  # noqa: E402

DAEMON_DIR = Path(__file__).resolve().parent
DAEMON_NAME = DAEMON_DIR.name
CFG = Config(DAEMON_NAME)
ODRIVE_CFG = Config("odrive")

BASE_DIR = Path(__file__).resolve().parent
STM_FIRMWARE_DIR = BASE_DIR / "firmware/stm"
BUILD_DIR = STM_FIRMWARE_DIR / "build-active"
STM_IMAGE = BUILD_DIR / "baseboard_balance.bin"
MODELS_DIR = BASE_DIR / "models"
POLICY_HEADER = STM_FIRMWARE_DIR / "balance/gen/policy_weights.h"
POLICY_EXPORTER = STM_FIRMWARE_DIR / "tools/export_onnx_weights.py"
FULL_SETUP_RESET_MARKER = BASE_DIR / ".full-setup-reset-required"

STM_RUNTIME_USB = "cafe:5630"
STM_DFU_USB = "0483:df11"
ODRIVE_RUNTIME_USB = "1209:0d32"

ODRIVE_IMAGE = BASE_DIR / "firmware/od-firmware/ODriveFirmware.bin"
ODRIVE_IMAGE_SHA256 = "c0b3eede12030123502b7dca84658ca1860c8a5118ac58fd776d4cf9308bfa7d"


def run(command, **kwargs):
    """Print and run an external maintenance command."""
    print("  $ " + " ".join(str(part) for part in command), flush=True)
    return subprocess.run([str(part) for part in command], **kwargs)


def usb_devices():
    try:
        result = subprocess.run(["lsusb"], capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return result.stdout.lower() if result.returncode == 0 else ""


def has_usb(device_id):
    return device_id.lower() in usb_devices()


def wait_usb(device_id, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if has_usb(device_id):
            return True
        time.sleep(0.25)
    return False


def ensure_host_tools():
    """Ensure the STM build and USB tools bundled by devenv are available."""
    commands = ("cmake", "ninja", "arm-none-eabi-gcc", "dfu-util", "git", "lsusb")
    missing = [command for command in commands if shutil.which(command) is None]
    if not missing:
        return True
    print("  missing tools from the base devenv: " + ", ".join(missing))
    print("  rebuild the robot image so the pinned base toolchain is included")
    return False


def policies():
    return sorted(MODELS_DIR.glob("*.onnx"))


def choose_policy():
    available = policies()
    if not available:
        print(f"  no ONNX policies found in {MODELS_DIR}")
        return None
    print("\nAvailable balance policies:")
    for index, path in enumerate(available, 1):
        print(f"  {index}) {path.name}")
    while True:
        answer = input("select policy [1]: ").strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(available):
            return available[int(answer) - 1]
        print(f"enter a number from 1 to {len(available)}")


def export_policy(model):
    fd, temporary_name = tempfile.mkstemp(
        prefix=".policy_weights.", dir=POLICY_HEADER.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        result = run([
            sys.executable, POLICY_EXPORTER, model,
            "--name", model.stem,
            "--profile", "baseboard-balance-18d",
            "--out", temporary,
        ])
        if result.returncode != 0:
            return False
        text = temporary.read_text()
        if not re.search(r"^#define\s+POLICY_N_IN\s+18\s*$", text, re.MULTILINE):
            print("  refusing policy: baseboard balance policies need 18 inputs")
            return False
        os.replace(temporary, POLICY_HEADER)
        print(f"  policy weights updated from {model.name}")
        return True
    finally:
        temporary.unlink(missing_ok=True)


def build_stm(policy=None):
    if policy is not None and not export_policy(Path(policy)):
        return None
    configure = run([
        "cmake", "-S", STM_FIRMWARE_DIR, "-B", BUILD_DIR, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
    ])
    if configure.returncode != 0:
        return None
    build = run(["cmake", "--build", BUILD_DIR,
                 "--target", "baseboard_balance", "--clean-first"])
    return STM_IMAGE if build.returncode == 0 and STM_IMAGE.is_file() else None


def manual_dfu_instructions():
    print("\n  Could not put the STM into DFU mode automatically.")
    print("  On the baseboard: hold BOOT, press and release RESET, then release BOOT.")
    print("  Then rerun `calibrate base` and choose the same option.")


def flash_stm(policy=None, staged_full_setup=False):
    """Build and flash STM firmware. Return done, retry, reset, or failed."""
    if staged_full_setup and FULL_SETUP_RESET_MARKER.exists():
        if has_usb(STM_RUNTIME_USB):
            FULL_SETUP_RESET_MARKER.unlink(missing_ok=True)
            print("  STM reset detected; continuing setup")
            return "done"
        print("\n  STM firmware is flashed but the runtime has not appeared.")
        print("  Release BOOT, press RESET, then rerun `calibrate base` option 2.")
        return "reset"

    dfu_util = shutil.which("dfu-util")
    if dfu_util is None:
        print("  dfu-util is unavailable")
        return "failed"

    started_in_dfu = has_usb(STM_DFU_USB)
    if not started_in_dfu and not has_usb(STM_RUNTIME_USB):
        manual_dfu_instructions()
        return "retry"

    image = build_stm(policy)
    if image is None:
        return "failed"

    if not started_in_dfu:
        print("\n==> entering STM ROM DFU")
        run([dfu_util, "-d", STM_RUNTIME_USB, "-a", "0", "-e"])
        if not wait_usb(STM_DFU_USB, 5.0):
            manual_dfu_instructions()
            return "retry"

    print("\n==> flashing STM")
    address = "0x08000000" if started_in_dfu and staged_full_setup else "0x08000000:leave"
    result = run(["sudo", "-n", dfu_util, "-d", STM_DFU_USB, "-a", "0",
                  "-s", address, "-D", image])

    if started_in_dfu and staged_full_setup:
        if result.returncode != 0:
            print("  STM flash failed")
            return "failed"
        FULL_SETUP_RESET_MARKER.touch()
        print("\n  STM firmware flashed successfully.")
        print("  Release BOOT, press RESET, then rerun `calibrate base` option 2.")
        return "reset"

    if not wait_usb(STM_RUNTIME_USB, 20.0):
        print("  STM runtime USB did not return after flashing")
        return "failed"
    if result.returncode != 0:
        print("  dfu-util lost final status, but the STM runtime returned")
    return "done"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def odrive_firmware_image():
    """Verify and return the fleet ODrive image stored with BBOS."""
    if not ODRIVE_IMAGE.is_file():
        print(f"  ODrive firmware is missing: {ODRIVE_IMAGE}")
        return None
    if file_sha256(ODRIVE_IMAGE) != ODRIVE_IMAGE_SHA256:
        print("  local ODrive firmware failed its pinned SHA-256 check")
        return None
    return ODRIVE_IMAGE


def bootstrap_odrive_can_usb():
    """Set only the CAN fields needed to continue setup through the baseboard."""
    settings = ODRIVE_CFG.settings
    baud = int(settings["can.config.baud_rate"])
    protocol = int(settings["can.config.protocol"])
    node0 = int(settings["axis0.config.can.node_id"])
    node1 = int(settings["axis1.config.can.node_id"])
    script = f"""
import odrive
from odrive.libodrive import TransportException

device = odrive.find_any(timeout=15)
device.can.config.baud_rate = {baud}
device.can.config.protocol = {protocol}
device.axis0.config.can.node_id = {node0}
device.axis1.config.can.node_id = {node1}
try:
    device.save_configuration()
except TransportException:
    pass
"""
    print(f"\n==> setting ODrive CAN bootstrap to {baud} bit/s over USB")
    if run([sys.executable, "-c", script]).returncode != 0:
        print("  could not save the ODrive CAN bootstrap over USB")
        return False
    if not wait_usb(ODRIVE_RUNTIME_USB, 20.0):
        print("  ODrive USB did not return after saving its CAN bootstrap")
        return False
    print(f"  ODrive CAN is ready at {baud} bit/s (nodes {node0}, {node1})")
    return True


def flash_odrive():
    """Flash BracketBot's pinned v3.6-56V ODrive image over native USB."""
    missing = [command for command in ("dfu-util", "lsusb")
               if shutil.which(command) is None]
    if missing:
        print("  missing tools from the base devenv: " + ", ".join(missing))
        return "failed"
    dfu_util = shutil.which("dfu-util")
    if shutil.which("sudo") is None or run(["sudo", "-n", "true"]).returncode != 0:
        print("  passwordless sudo is required for ODrive flashing")
        return "failed"

    print("\n  This flashes BracketBot OD-Firmware for ODrive v3.6-56V.")
    print("  Connect the powered ODrive directly to the robot computer over USB.")
    print("  The flash erases its configuration; the base will remain stopped.")
    if input("  Correct ODrive connected and robot restrained? [yes/no]: ").strip().lower() != "yes":
        print("  aborted")
        return "aborted"

    image = odrive_firmware_image()
    if image is None:
        return "failed"

    started_in_dfu = has_usb(STM_DFU_USB)
    if not started_in_dfu:
        if not has_usb(ODRIVE_RUNTIME_USB):
            print("  ODrive USB was not found (expected 1209:0d32)")
            return "failed"
        print("\n==> entering ODrive ROM DFU")
        enter_dfu = (
            "import odrive; from odrive.libodrive import TransportException; "
            "device = odrive.find_any(timeout=15); "
            "\ntry: device.enter_dfu_mode()"
            "\nexcept TransportException: pass"
        )
        run([sys.executable, "-c", enter_dfu])
        if not wait_usb(STM_DFU_USB, 10.0):
            print("  Could not enter ODrive DFU automatically.")
            print("  Set its DFU switch to DFU, power-cycle it, then rerun option 1.")
            return "retry"

    print("\n==> flashing BracketBot ODrive firmware")
    result = run(["sudo", "-n", dfu_util, "-d", STM_DFU_USB, "-a", "0",
                  "-s", "0x08000000:leave", "-D", image])
    if not wait_usb(ODRIVE_RUNTIME_USB, 20.0):
        if started_in_dfu and result.returncode == 0 and has_usb(STM_DFU_USB):
            print("  ODrive firmware flashed successfully.")
            print("  Set its DFU switch back to RUN and power-cycle the ODrive.")
            print("  Then use option 8 to configure it; do not flash it again.")
            return "reset"
        print("  ODrive runtime USB did not return after flashing")
        print("  Set its DFU switch back to RUN and power-cycle it if necessary.")
        return "failed"
    if result.returncode != 0:
        print("  dfu-util lost final status, but the ODrive runtime returned")
    if not bootstrap_odrive_can_usb():
        return "failed"
    return "done"


def current_policy_name():
    try:
        for line in POLICY_HEADER.read_text().splitlines():
            if line.startswith("#define POLICY_NAME"):
                return line.split('"', 2)[1]
    except (OSError, IndexError):
        pass
    return "unknown"

# Lower temporary gains prevent unloaded wheels from oscillating on the stand.
CALIB_VEL_GAIN = 2.0
CALIB_VEL_INTEGRATOR_GAIN = 2.5
STOPPED = DAEMON_DIR / ".stopped"
CONFIG_YAML = Path(CFG.calibration_path)

IMU_SECONDS = 180.0
IMU_RATE_HZ = 100.0
# Large bias or noise means the robot moved during calibration.
MAX_BIAS_NORM = 0.05
MAX_AXIS_STD = 0.01


def yaml_load(path):
    """Use the daemon's parser so calibration and runtime share one schema."""
    return driver.load_calibration(path)


def yaml_save(path, data, note):
    lines = ["# Per-robot calibration written by base/calibrate.py.",
             "# Applied by 'calibrate base'.",
             "# A normal daemon restart never changes live baseboard config.",
             "# " + note,
             ""]
    for k in sorted(data):
        lines.append(f"{k}: {data[k]!r}" if isinstance(data[k], str) else f"{k}: {data[k]}")
    path.write_text("\n".join(lines) + "\n")


def daemon_running():
    r = subprocess.run(
        ["pgrep", "-f", f"daemon.py {DAEMON_NAME}"], capture_output=True
    )
    return r.returncode == 0


def stop_daemon():
    """Stop supervision before taking exclusive ownership of /dev/ttySTM."""
    STOPPED.touch()
    for _ in range(50):
        if not daemon_running():
            return True
        time.sleep(0.2)
    return False


def start_daemon():
    STOPPED.unlink(missing_ok=True)


PHASE_RESISTANCE_RANGE = (0.15, 0.40)

# Endpoint IDs are pinned to the BracketBot OD-Firmware image above. Axis 1's
# generated object tree is the same layout as axis 0, offset by 322 IDs.
_AXIS_ENDPOINTS = {
    "config.can.bus_vi_rate_ms": (183, "uint32"),
    "config.can.controller_error_rate_ms": (178, "uint32"),
    "config.can.encoder_error_rate_ms": (177, "uint32"),
    "config.can.encoder_rate_ms": (175, "uint32"),
    "config.can.heartbeat_rate_ms": (174, "uint32"),
    "config.can.iq_rate_ms": (181, "uint32"),
    "config.can.motor_error_rate_ms": (176, "uint32"),
    "config.can.node_id": (172, "uint32"),
    "config.enable_watchdog": (146, "bool"),
    "config.startup_closed_loop_control": (140, "bool"),
    "config.watchdog_timeout": (145, "float"),
    "controller.config.control_mode": (275, "uint8"),
    "controller.config.enable_overspeed_error": (274, "bool"),
    "controller.config.enable_vel_limit": (271, "bool"),
    "controller.config.input_mode": (276, "uint8"),
    "controller.config.pos_gain": (277, "float"),
    "controller.config.spinout_electrical_power_threshold": (309, "float"),
    "controller.config.spinout_mechanical_power_threshold": (308, "float"),
    "controller.config.torque_ramp_rate": (284, "float"),
    "controller.config.torque_slew_rate": (288, "float"),
    "controller.config.vel_deadband": (287, "float"),
    "controller.config.vel_gain": (278, "float"),
    "controller.config.vel_integrator_gain": (279, "float"),
    "controller.config.vel_integrator_limit": (280, "float"),
    "controller.config.vel_limit": (281, "float"),
    "controller.config.vel_limit_tolerance": (282, "float"),
    "controller.config.vel_lpf_bandwidth": (285, "float"),
    "controller.config.vel_ramp_rate": (283, "float"),
    "controller.config.vel_slew_rate": (286, "float"),
    "current_state": (134, "uint8"),
    "encoder.config.bandwidth": (355, "float"),
    "encoder.config.calib_scan_distance": (357, "float"),
    "encoder.config.cpr": (349, "int32"),
    "encoder.config.mode": (343, "uint16"),
    "encoder.config.pre_calibrated": (353, "bool"),
    "encoder.config.use_index": (344, "bool"),
    "encoder.error": (325, "uint16"),
    "encoder.is_ready": (326, "bool"),
    "error": (130, "uint32"),
    "motor.config.calibration_current": (234, "float"),
    "motor.config.current_control_bandwidth": (246, "float"),
    "motor.config.current_lim": (240, "float"),
    "motor.config.current_lim_margin": (241, "float"),
    "motor.config.motor_type": (239, "uint8"),
    "motor.config.phase_resistance": (237, "float"),
    "motor.config.pole_pairs": (233, "int32"),
    "motor.config.pre_calibrated": (232, "bool"),
    "motor.config.requested_current_range": (245, "float"),
    "motor.config.resistance_calib_max_voltage": (235, "float"),
    "motor.config.torque_constant": (238, "float"),
    "motor.error": (185, "uint64"),
    "motor.is_calibrated": (187, "bool"),
}
ODRIVE_ENDPOINTS = {
    f"axis{axis}.{path}": (endpoint_id + axis * 322, value_type)
    for axis in (0, 1)
    for path, (endpoint_id, value_type) in _AXIS_ENDPOINTS.items()
}
ODRIVE_ENDPOINTS.update({
    "can.config.baud_rate": (69, "uint32"),
    "can.config.protocol": (70, "uint8"),
    "config.brake_resistance": (86, "float"),
    "config.enable_brake_resistor": (87, "bool"),
    "config.gpio9_mode": (110, "uint8"),
    "config.gpio10_mode": (111, "uint8"),
    "config.gpio11_mode": (112, "uint8"),
    "config.gpio12_mode": (113, "uint8"),
    "config.gpio13_mode": (114, "uint8"),
    "config.gpio14_mode": (115, "uint8"),
})

RXSDO, TXSDO = 0x05, 0x15
OP_READ, OP_WRITE, OP_SAVE = 0, 1, 2
SDO_STATUS = {0: "ok", 1: "denied", 2: "failed"}
SDO_FORMAT = {
    "float": "<f", "int32": "<i", "uint32": "<I", "int16": "<h",
    "uint16": "<H", "int8": "<b", "uint8": "<B", "bool": "<?",
}


class Sdo:
    """Read and write named ODrive endpoints through the STM CAN relay."""

    def __init__(self, link, node=0):
        self.link = link
        self.node = node

    def _entry(self, path):
        try:
            return ODRIVE_ENDPOINTS[path]
        except KeyError:
            raise KeyError(f"no pinned ODrive endpoint named {path!r}") from None

    def _exchange(self, payload, want_id, timeout=1.0):
        self.link.can_frames()
        self.link.can_send((self.node << 5) | RXSDO, payload)
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.link.send(0.0, 0.0, calib=True)
            for can_id, _, data in self.link.can_frames(0.02):
                if (can_id & 0x1f) != TXSDO or len(data) < 3:
                    continue
                if struct.unpack_from("<H", data, 1)[0] != want_id:
                    continue
                return data[0], data[3:]
        return None, None

    def read(self, path, retries=3):
        endpoint_id, value_type = self._entry(path)
        for _ in range(retries):
            status, value = self._exchange(
                struct.pack("<BH", OP_READ, endpoint_id), endpoint_id)
            if status is not None:
                break
        if status is None:
            raise TimeoutError(f"no TxSdo reply for {path} after {retries} tries")
        if status != 0:
            raise OSError(f"read {path}: {SDO_STATUS.get(status, status)}")
        value_format = SDO_FORMAT.get(value_type)
        if value_format:
            return struct.unpack_from(value_format, value)[0]
        if value_type in ("uint64", "int64"):
            return int.from_bytes(value[:5], "little")
        return value

    def write(self, path, value):
        endpoint_id, value_type = self._entry(path)
        value_format = SDO_FORMAT.get(value_type)
        if not value_format:
            raise TypeError(f"cannot write type {value_type} for {path}")
        payload = (struct.pack("<BH", OP_WRITE, endpoint_id) +
                   struct.pack(value_format, value))
        for _ in range(3):
            status, _ = self._exchange(payload, endpoint_id)
            if status is not None:
                break
        if status is None:
            raise TimeoutError(f"no TxSdo reply writing {path}")
        if status != 0:
            reason = SDO_STATUS.get(status, status)
            hint = " (writes need every axis IDLE)" if status == 1 else ""
            raise OSError(f"write {path}: {reason}{hint}")

    def save_configuration(self):
        status, _ = self._exchange(
            struct.pack("<BH", OP_SAVE, 0), 0, timeout=2.0)
        return status


def persist_calibration(link, nodes):
    """Verify and persist motor/encoder calibration while both axes are idle."""
    def insist(fn, *a, tries=6):
        """Retry SDO operations that lose replies under heavy CAN traffic."""
        last = None
        for _ in range(tries):
            try:
                return fn(*a)
            except Exception as exc:
                last = exc
                time.sleep(0.4)
        raise last

    print("\n=== PERSIST: writing the calibration to the drive ===")
    for node in nodes:
        link.odrive_request_state(node, driver.Link.AXIS_STATE_IDLE)
    link.send(0.0, 0.0, calib=True)
    time.sleep(0.5)

    sdos = {node: Sdo(link, node=node) for node in nodes}

    # Do not persist either axis until both calibrations verify.
    for node, sdo in sdos.items():
        try:
            calibrated = insist(sdo.read, f"axis{node}.motor.is_calibrated")
            ready = insist(sdo.read, f"axis{node}.encoder.is_ready")
            res = insist(sdo.read, f"axis{node}.motor.config.phase_resistance")
        except Exception as exc:
            print(f"  ! axis{node}: cannot read calibration state: {exc}")
            return False
        print(f"  axis{node}: motor.is_calibrated={calibrated} "
              f"encoder.is_ready={ready} phase_resistance={res:.4f}")
        if not calibrated or not ready:
            print(f"  ! axis{node} is not calibrated -- refusing to mark it"
                  " pre_calibrated")
            return False
        lo, hi = PHASE_RESISTANCE_RANGE
        if not (lo <= res <= hi):
            print(f"  ! axis{node} phase_resistance {res:.4f} is outside"
                  f" {lo}-{hi} ohm. Re-run the calibration; do NOT persist a"
                  " measurement this far off.")
            return False

    for node, sdo in sdos.items():
        for path in (f"axis{node}.motor.config.pre_calibrated",
                     f"axis{node}.encoder.config.pre_calibrated"):
            try:
                insist(sdo.write, path, True)
                if not insist(sdo.read, path):
                    print(f"  ! {path} did not take")
                    return False
                print(f"  set {path}")
            except Exception as exc:
                print(f"  ! {path}: {exc}")
                return False

    # The save reply only confirms acceptance; the ODrive then reboots.
    try:
        sdos[nodes[0]].save_configuration()
    except Exception as exc:
        print(f"  ! save_configuration: {exc}")
        return False
    print("  saved. The drive reboots now; wait for baseboard CAN heartbeats")
    print("  to recover before trusting it. Power-cycle the robot if they do not.")
    return True


AXIS_ERR = {
    0x1: "INVALID_STATE", 0x40: "MOTOR_FAILED", 0x80: "SENSORLESS_ESTIMATOR_FAILED",
    0x100: "ENCODER_FAILED", 0x200: "CONTROLLER_FAILED", 0x800: "WATCHDOG_TIMER_EXPIRED",
}
ENCODER_ERR = {
    0x1: "UNSTABLE_GAIN", 0x2: "CPR_POLEPAIRS_MISMATCH", 0x4: "NO_RESPONSE",
    0x8: "UNSUPPORTED_ENCODER_MODE", 0x10: "ILLEGAL_HALL_STATE",
    0x20: "INDEX_NOT_FOUND_YET", 0x40: "ABS_SPI_TIMEOUT",
}
MOTOR_ERR = {
    0x1: "PHASE_RESISTANCE_OUT_OF_RANGE", 0x2: "PHASE_INDUCTANCE_OUT_OF_RANGE",
    0x8: "DRV_FAULT", 0x10: "CONTROL_DEADLINE_MISSED", 0x100: "CURRENT_SENSE_SATURATION",
    0x400: "CURRENT_LIMIT_VIOLATION",
}
HINTS = {
    "ILLEGAL_HALL_STATE": "all three hall lines read the same -- they are open-collector, "
                          "so this is either config.gpio9..14 not set to 1 "
                          "(DIGITAL_PULL_UP) or a hall wire not making contact",
    "CPR_POLEPAIRS_MISMATCH": "cpr does not match pole_pairs*6 -- check encoder.config.cpr = 90",
    "PHASE_RESISTANCE_OUT_OF_RANGE": "motor leads or the motor itself; expect 0.15-0.40 ohm",
    "WATCHDOG_TIMER_EXPIRED": "enable_watchdog is on and nothing is feeding it; "
                              "the ODrive-config menu option pins it off before calibrating",
}


def _names(value, table):
    return [n for b, n in table.items() if value & b] or ["(none)"]


def link_is_stable(link, seconds=3.0):
    """Reject an unstable USB link before starting motor calibration."""
    before = link.reconnects
    deadline = time.time() + seconds
    while time.time() < deadline:
        link.send(0.0, 0.0, calib=True)
        list(link.read(0.05))
    grew = link.reconnects - before
    if grew:
        print(f"\n  !! the serial link reconnected {grew} time(s) in {seconds:.0f}s")
        print("  The baseboard USB link is re-enumerating. Reseat both")
        print("  ends of the USB cable, try another cable and another port, then")
        print("  re-run. Calibrating over this link will fail in a way that looks")
        print("  like a motor or encoder fault.")
        return False
    return True


def abort_odrive_calibration(link):
    """Return both axes to IDLE after an interrupted calibration."""
    sdos = [Sdo(link, node=node) for node in ODRIVE_NODES]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        link.send(0.0, 0.0, calib=True)
        for node in ODRIVE_NODES:
            link.odrive_request_state(node, driver.Link.AXIS_STATE_IDLE)
        time.sleep(0.1)
        try:
            states = [int(sdo.read(f"axis{node}.current_state", retries=1))
                      for node, sdo in zip(ODRIVE_NODES, sdos)]
        except Exception:
            continue
        if all(state == driver.Link.AXIS_STATE_IDLE for state in states):
            print("  both ODrive axes returned to IDLE")
            return True
    print("  !! ODrive axes did not return to IDLE; power-cycle the drive")
    return False


def run_odrive_calibration(link, timeout_s=120.0):
    """Request the firmware-driven full ODrive calibration and await its result."""
    print("\n=== ODRIVE: full calibration sequence ===")
    print("  Both motors will spin hard, one axis at a time. This takes up to")
    print("  ~90 s and cannot be interrupted safely once started.")

    deadline = time.time() + timeout_s
    seen_active = False
    last = None
    try:
        while time.time() < deadline:
            # The request is edge-detected; calibration mode keeps normal control off.
            link.send(0.0, 0.0, calib=True, run_calib=True)
            for st in link.read(0.05):
                state = driver.calibration_state(st.flags)
                if state != last:
                    last = state
                    if state:
                        print(f"    {state}")
                if state == "active":
                    seen_active = True
                elif state == "ok" and seen_active:
                    print("  calibration OK")
                    return True
                elif state == "failed":
                    print(f"  calibration FAILED (axis errors {st.ax0_err:#x}/"
                          f"{st.ax1_err:#x})")
                    # Read sub-errors because the axis error is only a summary.
                    for node, axis_err in ((0, st.ax0_err), (1, st.ax1_err)):
                        if not axis_err:
                            continue
                        print(f"    axis{node}: {', '.join(_names(axis_err, AXIS_ERR))}")
                        try:
                            sdo = Sdo(link, node=node)
                            for path, table in (("encoder.error", ENCODER_ERR),
                                                ("motor.error", MOTOR_ERR)):
                                link.send(0.0, 0.0, calib=True)
                                val = int(sdo.read(f"axis{node}.{path}") or 0)
                                if val:
                                    names = _names(val, table)
                                    print(f"      {path} {val:#x}: {', '.join(names)}")
                                    for n in names:
                                        if n in HINTS:
                                            print(f"        -> {HINTS[n]}")
                        except Exception as exc:
                            print(f"      (could not read detail over SDO: {exc})")
                    if link.reconnects:
                        print(f"    NOTE: the serial link reconnected {link.reconnects} time(s)"
                              " during this run -- suspect the USB cable before the drive")
                    abort_odrive_calibration(link)
                    return False
    except KeyboardInterrupt:
        print("\n  interrupted -- stopping both ODrive axes")
        abort_odrive_calibration(link)
        raise
    print(f"  timed out after {timeout_s:.0f}s with no terminal result")
    abort_odrive_calibration(link)
    return False


def enter_velocity_closed_loop(link, node, timeout_s=3.0):
    """Pace setup frames and prove the requested axis reached closed loop."""
    setup = (
        link.odrive_clear_errors,
        link.odrive_set_velocity_mode,
        lambda axis: link.odrive_set_vel_gains(
            axis, CALIB_VEL_GAIN, CALIB_VEL_INTEGRATOR_GAIN),
        lambda axis: link.odrive_set_velocity(axis, 0.0),
    )
    for command in setup:
        link.send(0.0, 0.0, calib=True)
        command(node)
        time.sleep(0.08)

    sdo = Sdo(link, node=node)
    deadline = time.monotonic() + timeout_s
    last_state = None
    last_error = 0
    last_read_error = None
    while time.monotonic() < deadline:
        link.send(0.0, 0.0, calib=True)
        link.odrive_request_state(node, driver.Link.AXIS_STATE_CLOSED_LOOP)
        time.sleep(0.10)
        try:
            last_state = int(sdo.read(f"axis{node}.current_state"))
            last_error = int(sdo.read(f"axis{node}.error"))
            last_read_error = None
        except Exception as exc:
            last_read_error = exc
            continue
        if last_state == driver.Link.AXIS_STATE_CLOSED_LOOP:
            return True
        if last_error:
            break

    if last_read_error is not None:
        print(f"    could not verify closed loop: {last_read_error}")
    else:
        print(f"    axis{node} did not enter closed loop "
              f"(state={last_state}, error={last_error:#x})")
    return False


def park_axis(link, node):
    """Stop an axis and restore its production mode and velocity gains."""
    link.send(0.0, 0.0, calib=True)
    link.odrive_set_velocity(node, 0.0)
    time.sleep(0.08)
    link.odrive_request_state(node, driver.Link.AXIS_STATE_IDLE)
    time.sleep(0.08)
    settings = ODRIVE_CFG.settings
    for _ in range(3):
        link.odrive_set_torque_mode(node)
        link.odrive_set_vel_gains(
            node,
            settings[f"axis{node}.controller.config.vel_gain"],
            settings[f"axis{node}.controller.config.vel_integrator_gain"],
        )
        time.sleep(0.08)


def run_full_odrive_calibration(link):
    """Run motor calibration with the saved watchdog disabled temporarily."""
    sdos = {node: Sdo(link, node=node) for node in ODRIVE_NODES}
    watchdog_before = {}
    try:
        odrive_make_idle(link)
        for node, sdo in sdos.items():
            path = f"axis{node}.config.enable_watchdog"
            watchdog_before[node] = bool(sdo.read(path))
            sdo.write(path, False)
            if bool(sdo.read(path)):
                raise RuntimeError(f"axis{node} watchdog did not turn off")
        print("  ODrive watchdogs disabled for calibration")

        for _ in range(3):
            link.send(0.0, 0.0, calib=True)
            for node in ODRIVE_NODES:
                link.odrive_clear_errors(node)
            time.sleep(0.1)
        errors = [int(sdos[node].read(f"axis{node}.error"))
                  for node in ODRIVE_NODES]
        if any(errors):
            print("  could not clear ODrive errors before calibration: "
                  f"{errors[0]:#x}/{errors[1]:#x}")
            return False

        # Clear_Errors changes the drive immediately, but its next heartbeat is
        # what updates the STM's cached axis errors. Do not start calibration
        # while that cache can still contain the old watchdog fault.
        cached_errors = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            link.send(0.0, 0.0, calib=True)
            for status in link.read(0.05):
                cached_errors = (status.ax0_err, status.ax1_err)
            if cached_errors == (0, 0):
                break
        if cached_errors != (0, 0):
            raise RuntimeError(
                f"STM still reports axis errors {cached_errors}")
        print("  ODrive errors cleared and confirmed by the STM")
        return run_odrive_calibration(link)
    except Exception as exc:
        print(f"  could not prepare the ODrive for calibration: {exc}")
        return False
    finally:
        odrive_make_idle(link)
        for node, enabled in watchdog_before.items():
            try:
                sdos[node].write(
                    f"axis{node}.config.enable_watchdog", enabled)
            except Exception as exc:
                print(f"  could not restore axis{node} watchdog: {exc}")


def calibrate_drive(link, full_odrive=None):
    """Measure encoder sign while a human identifies chassis-forward motion."""
    print("\n=== DRIVE: wheel direction ===")
    print("  !! BOTH WHEELS OFF THE GROUND -- they will be driven !!")
    if input("  Are both wheels off the ground and free? [yes/no]: ").strip().lower() != "yes":
        print("  aborted.")
        return None

    if full_odrive is None:
        ans = input("  Run the ODrive FULL_CALIBRATION_SEQUENCE first?\n"
                    "  Needed if this drive has never been calibrated, or if it\n"
                    "  reports axis error 0x1 (INVALID_STATE) and never spins.\n"
                    "  [yes/no]: ").strip().lower()
        full_odrive = ans == "yes"
    if full_odrive:
        if not link_is_stable(link):
            return None
        if not run_full_odrive_calibration(link):
            print("  aborting: the direction test is meaningless on a drive")
            print("  that cannot enter closed loop.")
            return None

    signs = {}
    for name, node in (("left", CFG.left_axis), ("right", CFG.right_axis)):
        print(f"\n  --- {name} wheel (node {node}) ---")
        while True:
            print("    spinning for 3 s ...")
            if not enter_velocity_closed_loop(link, node):
                park_axis(link, node)
                print("    refusing to send velocity while the axis is not closed loop")
                return None

            t_end = time.time() + 3.0
            samples = []
            next_tx = 0.0
            try:
                while time.time() < t_end:
                    # Pace commands at 50 Hz to satisfy timeouts without flooding CDC.
                    now = time.time()
                    if now >= next_tx:
                        next_tx = now + 0.02
                        # Feed the watchdog while keeping normal STM control disabled.
                        link.send(0.0, 0.0, calib=True)
                        link.odrive_set_velocity(node, 0.3)
                    # Encoder feedback arrives in the STM status packet.
                    for st in link.read(0.01):
                        samples.append(st.vel0 if node == CFG.right_axis else st.vel1)
            finally:
                park_axis(link, node)

            if not samples:
                print("    no encoder feedback -- check the drive is powered and on CAN")
                return None
            enc = sum(samples) / len(samples)
            print(f"    encoder reported {enc:+.3f} turns/s while commanded +0.3")
            if abs(enc) < 0.02:
                # Never record a human direction judgment for a stationary wheel.
                print("    the wheel did not turn. Not asking you to judge the")
                print("    direction of a stationary wheel -- the axis entered")
                print("    closed loop, so check the motor, encoder, and mechanics.")
                return None

            ans = input("    Did the robot try to move FORWARD or BACKWARD? [f/b/again]: ").strip().lower()
            if ans.startswith("a"):
                continue
            if ans.startswith("f"):
                signs[name] = 1.0 if enc > 0 else -1.0
            elif ans.startswith("b"):
                signs[name] = -1.0 if enc > 0 else 1.0
            else:
                continue
            print(f"    sign_dir_{name} = {signs[name]:+.0f}")
            break

    settings = ODRIVE_CFG.settings
    try:
        for node in ODRIVE_NODES:
            sdo = Sdo(link, node=node)
            paths = (
                f"axis{node}.controller.config.vel_gain",
                f"axis{node}.controller.config.vel_integrator_gain",
            )
            expected = tuple(float(settings[path]) for path in paths)
            for path, value in zip(paths, expected):
                sdo.write(path, value)
            actual = tuple(float(sdo.read(path)) for path in paths)
            if any(abs(got - wanted) >= 1e-4
                   for got, wanted in zip(actual, expected)):
                print(f"  axis{node} production velocity gains did not restore: "
                      f"{actual[0]}/{actual[1]} wanted "
                      f"{expected[0]}/{expected[1]}")
                return None
    except Exception as exc:
        print(f"  could not restore production velocity gains: {exc}")
        return None
    print("  production velocity gains restored before saving calibration")

    # Persist only after both wheel directions are verified.
    if not persist_calibration(link, [CFG.left_axis, CFG.right_axis]):
        print("  aborting: ODrive calibration could not be persisted")
        return None

    return {"sign_dir_left": signs["left"], "sign_dir_right": signs["right"]}


CAPTURE_TOOL = STM_FIRMWARE_DIR / "tools/baseboard_bias_capture.py"
FINAL_GYRO_BIAS_KEYS = ("gyro_bias_x", "gyro_bias_y", "gyro_bias_z")
BOOTSTRAP_GYRO_BIAS_KEYS = (
    "bootstrap_gyro_bias_x", "bootstrap_gyro_bias_y", "bootstrap_gyro_bias_z"
)


def ensure_imu_calibration(results):
    """Capture a bootstrap bias into config.yaml so the daemon can start."""
    if (all(key in results for key in FINAL_GYRO_BIAS_KEYS) or
            all(key in results for key in BOOTSTRAP_GYRO_BIAS_KEYS)):
        return True
    if not CAPTURE_TOOL.exists():
        print(f"  ! no capture tool at {CAPTURE_TOOL}")
        return False

    print("\n  no IMU calibration yet -- capturing an initial gyro bias")
    if not stop_daemon():
        print("  ! could not stop the base daemon to take the USB link")
        return False
    r = subprocess.run([sys.executable, str(CAPTURE_TOOL),
                        "--port", CFG.port,
                        "--driver-dir", str(Path(__file__).resolve().parent),
                        "--link-version", str(CFG.expected_link_version),
                        "--default-mode", str(CFG.default_mode),
                        "--lean-angle-deg", str(CFG.lean_angle_deg),
                        "--imu-filter-bandwidth-hz",
                        str(CFG.imu_filter_bandwidth_hz),
                        "--imu-filter-order", str(CFG.imu_filter_order),
                        "--imu-filter-beta", str(CFG.imu_filter_beta),
                        "--imu-filter-settle-beta",
                        str(CFG.imu_filter_settle_beta)],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.strip().splitlines()[-8:]:
        print(f"    {line}")
    match = re.search(r"^CALIBRATION_GYRO_BIAS=([^\n]+)$", out, re.MULTILINE)
    if r.returncode != 0 or match is None:
        print("  ! could not capture the IMU calibration; keep the robot still and retry")
        return False
    try:
        bias = [float(value) for value in match.group(1).split(",")]
    except ValueError:
        bias = []
    if len(bias) != 3 or not all(math.isfinite(value) for value in bias):
        print("  ! capture tool returned an invalid gyro bias")
        return False
    results.update(zip(BOOTSTRAP_GYRO_BIAS_KEYS, bias))
    yaml_save(CONFIG_YAML, results,
              "Initial IMU bias captured; final IMU calibration is pending.")
    print(f"  initial gyro bias saved in {CONFIG_YAML}")
    return True


def calibrate_imu(seconds=IMU_SECONDS):
    """Measure residual gyro bias from the baseboard IMU while stationary."""
    print(f"\n=== IMU: gyro bias ({seconds:.0f} s) ===")
    print("  Robot must be COMPLETELY STILL. Do not lean on it, do not walk into it.")
    if input("  Still and ready? [yes/no]: ").strip().lower() != "yes":
        print("  aborted.")
        return None

    # imu.raw is already corrected, so add its residual to the applied bias.
    applied = driver.resolve_gyro_bias(CFG)[0]
    print(f"  board is already correcting by "
          f"[{applied[0]:+.6f} {applied[1]:+.6f} {applied[2]:+.6f}]")

    n = 0
    s = [0.0, 0.0, 0.0]
    ss = [0.0, 0.0, 0.0]
    accel_n = 0
    accel_sum = 0.0
    t0 = time.time()
    with Reader("imu.raw", keeptime=False) as r:
        last_print = 0.0
        while time.time() - t0 < seconds:
            if r.ready():
                g = r.data["gyro"]
                a = r.data["accel"]
                for i in range(3):
                    v = float(g[i])
                    s[i] += v
                    ss[i] += v * v
                n += 1
                accel_sum += math.sqrt(sum(float(x) ** 2 for x in a))
                accel_n += 1
            el = time.time() - t0
            if el - last_print >= 15.0:
                last_print = el
                print(f"    {el:5.0f}/{seconds:.0f}s  n={n}")
            time.sleep(1.0 / (IMU_RATE_HZ * 2))

    if n < seconds * IMU_RATE_HZ * 0.5:
        print(f"  only {n} samples in {seconds:.0f}s -- is the base daemon running?")
        return None

    residual = [s[i] / n for i in range(3)]
    std = [math.sqrt(max(0.0, ss[i] / n - residual[i] ** 2)) for i in range(3)]
    bias = [applied[i] + residual[i] for i in range(3)]
    norm = math.sqrt(sum(b * b for b in bias))
    accel_norm = accel_sum / max(1, accel_n)

    print(f"  samples      {n}")
    print(f"  residual     [{residual[0]:+.6f} {residual[1]:+.6f} {residual[2]:+.6f}]"
          f"  (what is left after the board's correction)")
    print(f"  bias  rad/s  [{bias[0]:+.6f} {bias[1]:+.6f} {bias[2]:+.6f}]  |b|={norm:.6f}"
          f"  (absolute, this is what gets stored)")
    print(f"  std   rad/s  [{std[0]:.6f} {std[1]:.6f} {std[2]:.6f}]")
    print(f"  |accel|      {accel_norm:.3f} m/s^2  (expect ~9.81)")

    bad = []
    if norm > MAX_BIAS_NORM:
        bad.append(f"bias norm {norm:.4f} > {MAX_BIAS_NORM}")
    if max(std) > MAX_AXIS_STD:
        bad.append(f"axis std {max(std):.4f} > {MAX_AXIS_STD} (robot moved)")
    if abs(accel_norm - 9.81) > 0.25:
        bad.append(f"|accel| {accel_norm:.3f} implausible")
    if bad:
        print("  REJECTED:")
        for b in bad:
            print("    - " + b)
        return None

    return {"gyro_bias_x": bias[0], "gyro_bias_y": bias[1], "gyro_bias_z": bias[2]}


def run_calibration(choice, imu_seconds=IMU_SECONDS, full_odrive=False):
    """Run drive, IMU, or both calibrations and persist the robot YAML."""
    results = dict(yaml_load(CONFIG_YAML))
    pre_stopped = STOPPED.exists()

    try:
        if choice in ("drive", "both"):
            # Calibration owns the serial link and disables normal balance control.
            if not stop_daemon():
                print("could not stop the base daemon; aborting")
                return 1
            link = driver.Link(CFG.port, CFG.baud, CFG.expected_link_version)
            try:
                got = calibrate_drive(
                    link, full_odrive=True if full_odrive else None
                )
            finally:
                link.close()
            if got is None:
                return 1
            results.update(got)
            if choice == "both":
                yaml_save(CONFIG_YAML, results,
                          "Drive complete; IMU calibration is still pending.")

        if choice in ("imu", "both"):
            missing = [k for k in ("sign_dir_left", "sign_dir_right")
                       if k not in results]
            if missing:
                print("drive calibration must run before IMU calibration")
                return 1
            if not ensure_imu_calibration(results):
                return 1
            # imu.raw exists only while the base daemon is running.
            start_daemon()
            print("  waiting for the base daemon to republish imu.raw ...")
            time.sleep(12.0)
            got = calibrate_imu(imu_seconds)
            if got is None:
                return 1
            results.update(got)
            for key in BOOTSTRAP_GYRO_BIAS_KEYS:
                results.pop(key, None)

        results["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["hostname"] = os.uname().nodename
        yaml_save(CONFIG_YAML, results,
                  "Pushed to the baseboard with live output disabled.")

        print(f"\nwrote {CONFIG_YAML}")
        for k in sorted(results):
            print(f"  {k}: {results[k]}")

        # Push the saved values while normal balance output is disabled.
        if not stop_daemon():
            print("wrote the YAML but could not stop the base daemon to push it")
            return 1
        try:
            base_config = driver.build_config(CFG)
        except Exception as exc:
            print(f"wrote the YAML but could not build firmware config: {exc}")
            return 1

        link = driver.Link(CFG.port, CFG.baud, CFG.expected_link_version)
        try:
            link.apply_config(base_config)
        finally:
            link.close()
        print("pushed config to the baseboard with live output disabled")
        return 0
    finally:
        # Restore the supervisor state that existed before calibration.
        if pre_stopped:
            stop_daemon()
        else:
            start_daemon()


def confirm(message):
    print(f"\n{message}")
    try:
        return input("continue? [yes/no]: ").strip().lower() == "yes"
    except EOFError:
        return False


def stop_and_latch(required=True):
    if not stop_daemon():
        print("  could not stop the base daemon")
        return False
    try:
        link = driver.Link(CFG.port, CFG.baud, CFG.expected_link_version)
        try:
            link.stand_down()
        finally:
            link.close()
        print("  baseboard maintenance mode latched")
        return True
    except Exception as exc:
        if required:
            print(f"  could not latch baseboard maintenance mode: {exc}")
            return False
        print(f"  baseboard maintenance unavailable: {exc}")
        print("  continuing without it")
        return True


ODRIVE_NODES = (0, 1)


def desired_odrive_settings(watchdog=False):
    settings = dict(ODRIVE_CFG.settings)
    for axis in ODRIVE_NODES:
        settings[f"axis{axis}.config.enable_watchdog"] = int(watchdog)
    return list(settings.items())


def odrive_same(got, wanted):
    try:
        return abs(float(got) - float(wanted)) < 1e-4
    except (ValueError, TypeError):
        return False


def sdo_for(sdos, path):
    return sdos[1 if path.startswith("axis1.") else 0]


def odrive_make_idle(link):
    for _ in range(5):
        link.send(0.0, 0.0, calib=True)
        for node in ODRIVE_NODES:
            link.odrive_request_state(node, driver.Link.AXIS_STATE_IDLE)
        time.sleep(0.05)
    time.sleep(0.3)


def odrive_drift(sdos, wanted):
    drift = []
    for path, value in wanted:
        got = sdo_for(sdos, path).read(path)
        if not odrive_same(got, value):
            drift.append((path, got, value))
    return drift


def print_odrive_drift(drift, suffix):
    for path, got, value in drift:
        print("    %-52s %s -> %s %s" % (path, got, value, suffix))


def configure_odrive(apply=False, watchdog=False, save=True):
    """Check or persist constants.py settings through the STM CAN relay."""
    link = driver.Link(CFG.port, CFG.baud, CFG.expected_link_version)
    sdos = {node: Sdo(link, node=node) for node in ODRIVE_NODES}
    wanted = desired_odrive_settings(watchdog)
    try:
        if not apply:
            drift = odrive_drift(sdos, wanted)
            print_odrive_drift(drift, "wanted")
            print(f"  {len(drift)} setting(s) differed from spec")
            return not drift

        watchdog_before = {
            f"axis{node}.config.enable_watchdog":
                sdos[node].read(f"axis{node}.config.enable_watchdog")
            for node in ODRIVE_NODES
        }
        odrive_make_idle(link)
        for node in ODRIVE_NODES:
            sdos[node].write(f"axis{node}.config.enable_watchdog", False)

        drift = odrive_drift(sdos, wanted)
        if not watchdog:
            present = {path for path, _, _ in drift}
            for path, got in watchdog_before.items():
                if not odrive_same(got, 0) and path not in present:
                    drift.append((path, got, 0))
        drift.sort(key=lambda row: row[0].endswith(".config.enable_watchdog"))

        if not drift:
            print("  all ODrive settings already match")
            return True

        failed = []
        for path, got, value in drift:
            try:
                sdo_for(sdos, path).write(path, value)
                back = sdo_for(sdos, path).read(path)
            except Exception as exc:
                failed.append((path, str(exc), value))
                continue
            print("    %-52s %s -> %s" % (path, got, back))
            if not odrive_same(back, value):
                failed.append((path, back, value))

        if failed:
            print(f"  {len(failed)} setting(s) FAILED; not saving:")
            print_odrive_drift(failed, "wanted")
            return False
        if not save:
            print("  settings verified; not saved")
            return True

        status = sdos[0].save_configuration()
        if status != 0:
            print(f"  ODrive refused save_configuration: {status}")
            return False
        print("  saved; waiting for the ODrive to reboot")
        time.sleep(5.0)
        persisted = odrive_drift(sdos, wanted)
        if persisted:
            print("  settings changed after reboot:")
            print_odrive_drift(persisted, "wanted")
            return False
        print("  ODrive settings persisted and verified through the baseboard")
        return True
    except Exception as exc:
        print(f"  ODrive configuration over baseboard CAN failed: {exc}")
        return False
    finally:
        link.close()


def apply_robot_config():
    try:
        config = driver.build_config(CFG)
        link = driver.Link(CFG.port, CFG.baud, CFG.expected_link_version)
        try:
            link.apply_config(config)
        finally:
            link.close()
        print("  robot calibration applied to the STM")
        return True
    except Exception as exc:
        print(f"  could not apply robot calibration: {exc}")
        return False


def flash_stm_workflow(policy=None):
    if not ensure_host_tools():
        return 1
    pre_stopped = STOPPED.exists()
    stop_and_latch(required=False)
    result = flash_stm(policy=policy)
    if result != "done":
        if not pre_stopped and result == "failed":
            start_daemon()
        return 0 if result in ("retry", "reset") else 1
    if not stop_and_latch(required=True):
        return 1
    if not apply_robot_config():
        print("  leaving base stopped; run drive and IMU calibration first")
        return 1
    if not pre_stopped:
        start_daemon()
    print("\nSTM firmware updated.")
    return 0


def full_setup():
    """Run or redo the complete base setup, resumable across a manual DFU reset."""
    if not confirm("This reflashes the STM and redoes drive and IMU calibration.\n"
                   "Both wheels must be off the ground and the robot restrained."):
        print("aborted")
        return 1
    stop_and_latch(required=False)
    if not ensure_host_tools():
        return 1

    flash_result = flash_stm(staged_full_setup=True)
    if flash_result in ("retry", "reset"):
        return 0
    if flash_result != "done":
        return 1
    if not stop_and_latch(required=True):
        return 1

    print("\n==> configuring ODrive for calibration")
    if not configure_odrive(apply=True):
        print("  leaving base stopped because ODrive setup failed")
        return 1

    print("\n==> calibrating drive and IMU")
    if run_calibration("both", full_odrive=True) != 0:
        print("  leaving base stopped because calibration did not finish")
        return 1

    if not stop_and_latch(required=True):
        return 1
    print("\n==> enabling and verifying the ODrive watchdog")
    if not configure_odrive(apply=True, watchdog=True):
        print("  leaving base stopped because ODrive verification failed")
        return 1

    start_daemon()
    print("\nBase setup complete.")
    return 0


def update_odrive_config():
    if not confirm("Torque will drop while the ODrive configuration is saved."):
        print("aborted")
        return 1
    pre_stopped = STOPPED.exists()
    try:
        if not stop_and_latch(required=True):
            return 1
        return 0 if configure_odrive(apply=True, watchdog=True) else 1
    finally:
        if not pre_stopped:
            start_daemon()


def update_imu_filter():
    if not confirm("Torque will drop while the IMU filter is updated."):
        print("aborted")
        return 1
    pre_stopped = STOPPED.exists()
    try:
        if not stop_and_latch(required=True):
            return 1
        config = driver.build_config(CFG)
        link = driver.Link(CFG.port, CFG.baud, CFG.expected_link_version)
        try:
            link.apply_config(config)
        finally:
            link.close()
        print("  IMU filter persisted to the STM:")
        print("    hardware: %d Hz, order %d" % (
            config["imu_filter_bandwidth_hz"],
            config["imu_filter_order"],
        ))
        print("    beta: normal=%g settling=%g" % (
            config["imu_filter_beta"],
            config["imu_filter_settle_beta"],
        ))
        return 0
    except Exception as exc:
        print(f"  could not update the IMU filter: {exc}")
        return 1
    finally:
        if not pre_stopped:
            start_daemon()


def check_everything():
    checks = []
    calibration = yaml_load(CONFIG_YAML)
    required = ("sign_dir_left", "sign_dir_right",
                "gyro_bias_x", "gyro_bias_y", "gyro_bias_z")
    checks.append(("robot calibration YAML",
                   CONFIG_YAML.exists() and all(k in calibration for k in required)))
    policy = current_policy_name()
    checks.append(("compiled balance policy", policy != "unknown"))
    checks.append(("base daemon running", daemon_running()))
    checks.append(("STM runtime USB", has_usb(STM_RUNTIME_USB)))
    checks.append(("/dev/ttySTM", Path(CFG.port).exists()))
    print(f"  compiled policy: {policy}")

    pre_stopped = STOPPED.exists()
    status = None
    odrive_ok = False
    try:
        if checks[-1][1] and stop_and_latch(required=True):
            try:
                link = driver.Link(CFG.port, CFG.baud, CFG.expected_link_version)
                try:
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        packets = link.read(0.1)
                        if packets:
                            status = packets[-1]
                finally:
                    link.close()
            except Exception:
                status = None
            odrive_ok = configure_odrive(watchdog=True)
    finally:
        if not pre_stopped:
            start_daemon()

    checks.append(("STM protocol/status", status is not None))
    if status is not None:
        checks.extend([
            ("STM config persisted", status.config_valid),
            ("IMU ready", bool(status.flags & (1 << 11))),
            ("IMU fresh", bool(status.flags & (1 << 12))),
            ("IMU has no error", not bool(status.flags & (1 << 13))),
            ("ODrive axes have no errors", status.ax0_err == 0 and status.ax1_err == 0),
            ("ODrive bus voltage present", status.vbus > 5.0),
        ])
    checks.append(("ODrive settings", odrive_ok))

    print("\nBase check:")
    for name, passed in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    return 0 if all(passed for _, passed in checks) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("daemon", nargs="?", help=argparse.SUPPRESS)
    args = ap.parse_args()
    del args

    print("\nBase maintenance:")
    print("  1) Flash ODrive")
    print("  2) Full setup / redo everything")
    print("  3) Flash STM")
    print("  4) Calibrate drive")
    print("  5) Calibrate IMU")
    print("  6) Update IMU filter")
    print("  7) Update policy")
    print("  8) Update ODrive config")
    print("  9) Check everything")
    choice = input("Select [1-9]: ").strip()

    if choice == "1":
        pre_stopped = STOPPED.exists()
        if not stop_and_latch(required=False):
            if not pre_stopped:
                start_daemon()
            return 1
        result = flash_odrive()
        if result == "done":
            print("\nODrive firmware updated. Base remains stopped.")
            print("Run option 2 to finish the base setup.")
            return 0
        if result == "aborted" and not pre_stopped:
            start_daemon()
        elif result != "aborted":
            print("  base remains stopped; finish or recover the ODrive before restarting it")
        return 0 if result in ("retry", "reset") else 1
    if choice == "2":
        return full_setup()
    if choice == "3":
        if not confirm("Flashing the STM drops torque. Restrain the robot first."):
            return 1
        return flash_stm_workflow()
    if choice == "4":
        return run_calibration("drive")
    if choice == "5":
        return run_calibration("imu")
    if choice == "6":
        return update_imu_filter()
    if choice == "7":
        policy = choose_policy()
        if policy is None:
            return 1
        if not confirm(f"Flash the STM with {policy.name}? Torque will drop."):
            return 1
        return flash_stm_workflow(policy)
    if choice == "8":
        return update_odrive_config()
    if choice == "9":
        return check_everything()
    print("nothing selected")
    return 1


if __name__ == "__main__":
    sys.exit(main())
