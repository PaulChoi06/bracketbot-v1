import glob
import logging
import os
import struct
import time

import numpy as np

from bbos import Config, Reader, Type, Writer

log = logging.getLogger("bb.led")

LED_SYMLINK = "/dev/bb-led"

LCMD_SET_MODE = 0x10
LCMD_SET_BRIGHTNESS = 0x20
LMODE_OFF = 0x00
LMODE_SOLID = 0x01
LMODE_BLINK = 0x02
LFLAG_LOOP = 0x01
TRANSITION_MS = 200

CFG = Config("led")

CTRL_RETRY_S = 5.0
DEFAULT_BRIGHTNESS = max(0, min(255, int(CFG.default_brightness)))
IDLE_RGB = tuple(CFG.idle_rgb)
IDLE_KEY = (IDLE_RGB, 0)
QUEST_RGB = tuple(CFG.quest_rgb)
QUEST_KEY = (QUEST_RGB, CFG.quest_blink_ms * 2)


def _find_hidraw():
    if os.path.exists(LED_SYMLINK):
        return LED_SYMLINK
    for dev in sorted(glob.glob("/dev/hidraw*")):
        name = os.path.basename(dev)
        try:
            with open(f"/sys/class/hidraw/{name}/device/report_descriptor", "rb") as f:
                d = f.read(3)
        except OSError:
            continue
        if len(d) >= 3 and d[0] == 0x06 and d[1] == 0x00 and d[2] == 0xFF:
            return dev
    return None


def _open():
    path = _find_hidraw()
    if path is None:
        log.warning("[led] no LED HID interface (usage page 0xFF00) found — LED disabled")
        return None
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError as e:
        log.warning("[led] cannot open %s (%s) — LED disabled", path, e)
        return None
    log.info("[led] LED HID open: %s", path)
    return fd


def _write(fd, payload):
    """Returns the fd, or None once the device has gone away."""
    if fd is None:
        return None
    buf = bytes([0x00]) + payload
    buf += bytes(64 - len(buf))
    try:
        os.write(fd, buf)
    except OSError as e:
        log.warning("[led] write failed (%s) — LED disabled", e)
        try:
            os.close(fd)
        except OSError:
            pass
        return None
    return fd


def _hdr(mode, transition_ms=TRANSITION_MS):
    return struct.pack("<BBBHBB", LCMD_SET_MODE, mode, LFLAG_LOOP,
                       transition_ms, LMODE_OFF, 0)


def _solid(rgb):
    return _hdr(LMODE_SOLID) + struct.pack("<BBB", *rgb)


def _blink(rgb, on_ms, off_ms):
    return _hdr(LMODE_BLINK, 0) + struct.pack("<BBBHHB", *rgb, on_ms, off_ms, 0)


def _off():
    return _hdr(LMODE_OFF, 0)


def _brightness(level):
    return struct.pack("<BB", LCMD_SET_BRIGHTNESS, level)


def main():
    fd = _open()
    r_ctrl = Reader("led.ctrl")
    r_link = Reader("quest.link")

    now = time.monotonic()
    last_hw_try = now
    last_brightness = None
    last_rgb = None
    last_app_owns = False
    app_cmd = None
    app_t = -1e9
    quest_link = False
    quest_link_t = -1e9

    with Writer("led.state", Type("led_state")) as w_state:
        while True:
            now = time.monotonic()

            if fd is None and now - last_hw_try >= CTRL_RETRY_S:
                fd = _open()
                last_hw_try = now
                last_rgb = None
                last_brightness = None

            if r_ctrl.ready():
                app_cmd = {"rgb": tuple(int(v) for v in r_ctrl.data["rgb"]),
                           "brightness": int(r_ctrl.data["brightness"]),
                           "period_ms": int(r_ctrl.data["period_ms"])}
                app_t = now

            if r_link.ready():
                quest_link = bool(r_link.data["connected"])
                quest_link_t = now
            quest_idle = quest_link and now - quest_link_t <= CFG.link_stale_s
            idle_rgb = QUEST_RGB if quest_idle else IDLE_RGB
            idle_key = QUEST_KEY if quest_idle else IDLE_KEY

            if app_cmd is not None and 0 <= now - app_t <= CFG.ctrl_stale_s:
                b = app_cmd["brightness"]
                app_owns, rgb = True, app_cmd["rgb"]
                brightness = DEFAULT_BRIGHTNESS if b < 0 else max(0, min(255, b))
            else:
                app_owns, rgb, brightness = False, None, DEFAULT_BRIGHTNESS

            if fd is not None and CFG.enabled:
                if app_owns != last_app_owns:
                    last_rgb = None
                if not app_owns and last_rgb != idle_key:
                    if quest_idle:
                        fd = _write(fd, _blink(QUEST_RGB, CFG.quest_blink_ms,
                                               CFG.quest_blink_ms))
                    else:
                        fd = _write(fd, _solid(IDLE_RGB))
                    last_rgb = idle_key
                if rgb is not None:
                    period = app_cmd["period_ms"]
                    key = (rgb, period)
                    if key != last_rgb:
                        if not any(rgb):
                            fd = _write(fd, _off())
                        elif period:
                            half = period // 2
                            fd = _write(fd, _blink(rgb, half, half))
                        else:
                            fd = _write(fd, _solid(rgb))
                        last_rgb = key
                if brightness != last_brightness:
                    fd = _write(fd, _brightness(brightness))
                    last_brightness = brightness
            last_app_owns = app_owns

            with w_state.buf() as b:
                b["rgb"] = np.array(rgb or idle_rgb, dtype=np.uint8)


if __name__ == "__main__":
    main()
