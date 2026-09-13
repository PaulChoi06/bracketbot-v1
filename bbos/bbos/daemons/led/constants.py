from bbos.registry import *
import numpy as np


@register
class led:
    enabled: bool = True
    ctrl_stale_s: float = 3.0
    default_brightness: int = 157
    idle_rgb: tuple = (255, 255, 255)
    quest_rgb: tuple = (0, 255, 0)
    quest_blink_ms: int = 500
    link_stale_s: float = 5.0


@realtime(ms=200)
def led_ctrl():
    """LED intent: rgb, brightness (-1 = default_brightness), period_ms (0 = steady, else blink)."""
    return [
        ("rgb", np.uint8, 3),
        ("brightness", np.int16, ()),
        ("period_ms", np.uint16, ()),
    ]


@realtime(ms=200)
def led_state():
    """The colour currently on the LED."""
    return [
        ("rgb", np.uint8, 3),
    ]
