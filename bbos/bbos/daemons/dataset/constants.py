"""Config and record types for the dataset daemon."""

from pathlib import Path

import numpy as np

from bbos import register, state


# ============================================================================
# Configs
# ============================================================================
def _api_key() -> str:
    """Empty when the key is absent or unreadable; only uploads need it."""
    try:
        return Path("/etc/BB_API_KEY").read_text().strip()
    except OSError:
        return ""


@register
class dataset:
    """Tunables for the dataset daemon, read once at startup."""

    bb_api_key: str = _api_key()
    bb_api_url: str = "https://api.bracketbot.com"
    upload_timeout_s: int = 1800     # 30 min; head AVIs are slow on wifi.
    upload_retries: int = 2
    # Per chunk, not per episode: 4x queue+active x 512MB = 2GB worst case,
    # which leaves ~3GB headroom on a 7.4GB box. Redo that math before raising.
    memory_cap_mb: int = 512
    pending_dir: str = str(Path(__file__).parent / ".data" / "pending")
    daemons_dir: str = str(Path(__file__).parent.parent)
    cal_arms: list = ["arm_left", "arm_right", "leader_left", "leader_right"]
    sensor_channels: dict = {
        "drive_state":        "drive.state",
        "drive_ctrl":         "drive.ctrl",
        "imu_orientation":    "imu.orientation",
        "imu_raw":            "imu.raw",
        "arm_left_state":     "arm_left.state",
        "arm_left_ctrl":      "arm_left.ctrl",
        "arm_right_state":    "arm_right.state",
        "arm_right_ctrl":     "arm_right.ctrl",
        "arm_left_target":    "arm_left.target",
        "arm_right_target":   "arm_right.target",
        "quest_joystick":     "quest.joystick",
        "leader_left_state":  "leader_left.state",
        "leader_right_state": "leader_right.state",
    }
    camera_channels: dict = {
        "head":      "camera.head.jpeg",
        "arm_left":  "camera.left.jpeg",
        "arm_right": "camera.right.jpeg",
        "test":      "camera.test_cam.jpeg",
    }


# ============================================================================
# IPC
# ============================================================================
@state
def dataset_flag():
    """Episode controls an app writes: dataset name, task text, toggle/drop."""
    return [
        ("prefix",         "S200", ()),
        ("name",           "S200", ()),
        ("text",           "S500", ()),
        ("toggle_episode", np.bool_),
        ("drop_episode",   np.bool_),
    ]
