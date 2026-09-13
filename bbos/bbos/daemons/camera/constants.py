"""Config and record types for the camera daemon."""

import numpy as np

from bbos import register, realtime, state


# ============================================================================
# Camera configs
# ============================================================================
@register
class cam_left:
    """Left wrist camera."""

    rate: int = 30
    card_substr: str = "icspring camera"
    # Serial-less and identical, so identified by the arm on the same hub.
    sibling_arm: str = "/dev/ttyARMLEFT"
    width: int = 640
    height: int = 480
    fmt: str = "MJPEG"


@register
class cam_right:
    """Right wrist camera. Resolved like cam_left, via its hub-mate arm."""

    rate: int = 30
    card_substr: str = "icspring camera"
    sibling_arm: str = "/dev/ttyARMRIGHT"
    width: int = 640
    height: int = 480
    fmt: str = "MJPEG"


@register
class cam_head:
    """Head stereo camera; its card name is unique, so no sibling needed."""

    rate: int = 60
    # Realtek 0bda:5883; the old unit was "PC CAMERA TB2".
    card_substr: str = "USB Camera"
    width: int = 2560
    height: int = 960
    fmt: str = "MJPEG"
    decimate: int = 2   # Publish every Nth frame; capture stays at `rate`.

    @staticmethod
    def split(stereo_img: np.ndarray):
        """Split a stereo frame into its (left, right) eyes."""
        half = cam_head.width // 2
        return stereo_img[:, :half], stereo_img[:, half:]


# ============================================================================
# Types
# ============================================================================
@state
def camera_status():
    """Liveness of one camera, on camera.<name>.status.

    The frame topics cannot answer it: their Writers outlive a device loss.
    """
    return [
        ("streaming", np.uint8, ()),   # 1 = inside the publish loop.
        ("fps", np.float32, ()),       # Frames per second reaching shm.
    ]


@realtime(ms=33)
def camera_head():
    """Full-resolution RGB; only the head cam publishes it."""
    return [("rgb", np.uint8, (cam_head.height, cam_head.width, 3))]


# --- JPEG output ------------------------------------------------------------
# Fixed buffer with a length field.
JPEG_BUF_SIZE_WRIST = cam_left.width * cam_left.height
# Must equal kMaxBitstreamBytes in the bbjpeg native module: every JPEG the
# decoder accepts must also fit the shm field, so the bounds cannot disagree.
JPEG_BUF_SIZE_HEAD = 4 * 1024 * 1024


@realtime(ms=30)
def camera_wrist_jpeg():
    """One wrist JPEG plus its byte count."""
    return [
        ("jpeg_len", np.int32, ()),
        ("jpeg", np.uint8, (JPEG_BUF_SIZE_WRIST,)),
    ]


@realtime(ms=33)
def camera_head_jpeg():
    """One head JPEG plus its byte count."""
    return [
        ("jpeg_len", np.int32, ()),
        ("jpeg", np.uint8, (JPEG_BUF_SIZE_HEAD,)),
    ]
