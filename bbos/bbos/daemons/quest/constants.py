"""Config and record types for the quest daemon."""

from pathlib import Path

import numpy as np

from bbos.registry import register, realtime, state


# ============================================================================
# Configs
# ============================================================================
@register
class quest:
    """Tunables for the quest daemon, read once at startup."""

    port: int = 9000
    web_port: int = 9001            # HTTPS port for the WebXR webapp.
    controller_angle_deg: float = 15.0
    cert_file: str = str((Path(__file__).parent / "cert.pem").resolve())
    key_file: str = str((Path(__file__).parent / "key.pem").resolve())
    # z: shoulder height from base; the plane the head maps onto.
    robot_shoulder_height: float = 1.265
    link_timeout_s: float = 1.0     # No state packets this long -> gone.
    link_publish_s: float = 0.5     # quest.link keepalive for consumers.


# ============================================================================
# Types
# ============================================================================
@realtime(ms=20)
def quest_controllers():
    """Head-relative controller poses, triggers, sticks and buttons.

    Poses are re-centered on the head (XY+yaw) with absolute Z. No calibration
    scale or z_offset: consumers own any remapping.
    """
    return [
        ("T_head", np.float64, (4, 4)),
        ("left_pose", np.float64, (7,)),
        ("right_pose", np.float64, (7,)),
        ("left_trigger", np.float64, ()),
        ("left_squeeze", np.float64, ()),
        ("left_thumbstick", np.float64, (2,)),
        ("left_thumbstick_click", np.float64, ()),
        ("left_a", np.float64, ()),
        ("left_b", np.float64, ()),
        ("right_trigger", np.float64, ()),
        ("right_squeeze", np.float64, ()),
        ("right_thumbstick", np.float64, (2,)),
        ("right_thumbstick_click", np.float64, ()),
        ("right_a", np.float64, ()),
        ("right_b", np.float64, ()),
    ]


@realtime(ms=20)
def quest_joystick():
    """Raw thumbsticks, published by the teleop app for the dataset.

    Slim twin of quest_controllers, which is not recorded (whole-record only,
    poses included).
    """
    return [
        ("left", np.float32, (2,)),
        ("right", np.float32, (2,)),
    ]


@state
def quest_link():
    """Headset link liveness, for status indicators like the led daemon.

    @state, not @realtime: a periodic topic would pace the 20 ms loop.
    """
    return [
        # 1 = packets within quest.link_timeout_s.
        ("connected", np.uint8, ()),
    ]


@state
def quest_haptic():
    """Haptic command an app writes; the daemon forwards it to the headset.

    Non-periodic, written on demand.
    """
    return [
        ("hand", np.uint8, ()),         # 0 = left, 1 = right, 2 = both.
        ("frequency", np.float64, ()),  # Hz (0 = device default).
        ("amplitude", np.float64, ()),  # 0..1.
        ("duration", np.float64, ()),   # Seconds.
    ]
