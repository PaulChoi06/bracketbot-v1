"""Config and types for the slam daemon"""
import os

import numpy as np
from bbos import Config
from bbos.registry import realtime, register, state

_DIR = os.path.dirname(os.path.abspath(__file__))
_depth = Config("depth")

# ============================================================================
# Config
# ============================================================================
@register
class slam:
    library: str = os.path.join(_DIR, "bbslam.so")
    engine: str = os.path.join(_DIR, "model.engine")
    calib: str = str(_depth.calib_path)
    mask_left: str = os.path.join(_DIR, "masks", "mask_l.png")
    mask_right: str = os.path.join(_DIR, "masks", "mask_r.png")
    base_from_camera: np.ndarray = _depth.camera_to_base_3x4
    input_topic: str = "camera.head.rgb"
    map_path: str = os.path.join(_DIR, ".maps", "slam.bbmap")
    history_path: str = os.path.join(_DIR, ".maps", "slam.history")
    poses_path: str = os.path.join(_DIR, ".maps", "slam.poses")
    map_save_interval_s: float = 30.0
    trace: bool = True   # slam.trace for the viewer: off skips every JPEG and payload
    record_bytes: int = 560
    trace_payload_max: int = 2 << 20
    log_interval: int = 300
    boot_reloc_max_attempts: int = 2000

# ============================================================================
# Types
# ============================================================================
@realtime(ms=33)
def slam_pose():
    return [
        ("pos", np.float32, 3),
        ("quat", np.float32, 4),
        ("vo_pos", np.float32, 3),
        ("vo_quat", np.float32, 4),
        ("pgo_count", np.int32),
    ]


@realtime(ms=500)
def slam_history_generation():
    return [
        ("generation", np.int64),
        ("num_poses", np.int32),
        ("pgo_count", np.int32),
    ]


@state
def slam_health():
    return [
        ("degraded", np.bool_),
        ("stalled", np.bool_),
        ("vo_lost", np.bool_),
        ("last_gap_ms", np.float32),
        ("localized", np.bool_),
        ("relocalized", np.bool_),
    ]


@realtime(ms=33)
def slam_trace():
    return [
        ("record", np.uint8, slam.record_bytes),
        ("lens", np.int32, 7),
        ("payload", np.uint8, slam.trace_payload_max),
    ]
