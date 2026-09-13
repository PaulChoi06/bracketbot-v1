"""Config and record type for the telemetry daemon."""

from pathlib import Path

import numpy as np

from bbos import register, state


def _api_key() -> str:
    """Empty when the key is absent or unreadable; only uploads need it."""
    try:
        return Path("/etc/BB_API_KEY").read_text().strip()
    except OSError:
        return ""


# ============================================================================
# Config
# ============================================================================
@register
class telemetry:
    """Tunables for the telemetry daemon, read once at startup."""

    # --- API credentials ----------------------------------------------------
    bb_api_key: str = _api_key()
    bb_api_url: str = "https://api.bracketbot.com"

    # --- Sampling & upload cadence ------------------------------------------
    streams: tuple = (
        "drive.state", "drive.ctrl", "drive.status",
        "arm_left.state", "arm_left.ctrl",
        "arm_right.state", "arm_right.ctrl",
        "leader_left.state", "leader_right.state",
        "camera.head.status", "camera.left.status", "camera.right.status",
    )
    sample_interval_s: float = 0.05
    push_interval_s: float = 1.0
    max_pending_rows: int = 1000000
    max_rows_per_push: int = 10000   # Keeps one POST under the body limit.

    # --- Structure documents ------------------------------------------------
    # topic -> robot_snapshots.kind, sent only when its shape_hash moves.
    snapshot_streams: dict = {"usb.tree": "usb_tree"}
    max_snapshots_per_push: int = 10
    max_pending_snapshots: int = 50


# ============================================================================
# Types
# ============================================================================
@state
def telemetry_flag():
    """Episode name, label, and the toggle/drop flags."""
    return [
        ("name", "S200", ()),
        ("label", "S500", ()),
        ("toggle_episode", np.bool_),
        ("drop_episode", np.bool_),
    ]
