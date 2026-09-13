"""HTTPS host for the WebXR webapp and its raw-data uplink.

The headset browser loads the page from this daemon over TLS (WebXR needs a
secure context), enters an immersive session and streams RAW WebXR data back
over a WebSocket: the head and grip XRRigidTransforms plus the xr-standard
gamepad buttons and axes, untransformed. Packets go to the daemon's `on_state`
callback, which folds them into the same shared arrays the UDP receiver
fills, so quest.controllers, quest.link and haptics behave the same whether
the operator runs the native app or the webapp.

This module owns the WebXR wire format end to end: `decode` turns one packet
into the poses and controller-state slots the daemon stores, so daemon.py
only does shared-array plumbing. Wire-format changes must stay in sync with
webapp/index.html.

The server owns its own asyncio loop on a background thread; no Reader or
Writer ever touches async code.
"""

import asyncio
import json
import ssl
import threading
from pathlib import Path

from aiohttp import WSMsgType, web

INDEX = Path(__file__).parent / "webapp" / "index.html"
WS_HEARTBEAT_S = 5.0        # Ping idle sockets so a dead headset is noticed.
MAX_MSG_SIZE = 1 << 16      # A state packet is a few hundred bytes.

_loop = None                # The server thread's event loop.
_clients = set()            # Live WebSocketResponse objects.

# --- Wire format (must match webapp/index.html) -----------------------------
# The xr-standard gamepad layout (WebXR Gamepads Module) is also the slot
# order of the daemon's *_controller_state_shared -- both descend from the
# same mapping.
BTN_TRIGGER = 0
BTN_SQUEEZE = 1
BTN_TOUCHPAD = 2
BTN_THUMB_CLICK = 3
BTN_AX = 4                  # A (right) / X (left).
BTN_BY = 5                  # B (right) / Y (left).
AXIS_TOUCHPAD = 0           # axes[0:2]
AXIS_THUMBSTICK = 2         # axes[2:4]


# ============================================================================
# Packet decode
# ============================================================================
def _at(seq, i):
    """seq[i] as a float; 0.0 when the browser omitted that entry."""
    try:
        return float(seq[i])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _pose(node):
    """A {"pos", "quat"} node as position + xyzw, or None if incomplete."""
    pos = (node or {}).get("pos") or ()
    quat = (node or {}).get("quat") or ()
    if len(pos) != 3 or len(quat) != 4:
        return None
    return tuple(float(v) for v in (*pos, *quat))


def _hand(node, press_th):
    """One raw hand as (pose | None, 12 state slots), or None if absent.

    Axes pass through untouched. The xr-standard mapping documents
    thumbstick Y as +down, but the Quest browser hands back the OpenXR value
    (+up) that the native app also streams -- checked on hardware, where
    negating here inverted forward/back. Do not "fix" this to match the spec.
    """
    if not node:
        return None
    btn = node.get("buttons") or ()
    press = node.get("pressed") or ()
    axes = node.get("axes") or ()
    trig, grip = _at(btn, BTN_TRIGGER), _at(btn, BTN_SQUEEZE)
    return _pose(node), [
        1.0 if trig > press_th else 0.0,    # 0  Trigger (bool).
        1.0 if grip > press_th else 0.0,    # 1  Squeeze (bool).
        _at(press, BTN_TOUCHPAD),           # 2  Touchpad (n/a on Touch).
        _at(press, BTN_THUMB_CLICK),        # 3
        _at(press, BTN_AX),                 # 4
        _at(press, BTN_BY),                 # 5
        trig,                               # 6  triggerValue.
        grip,                               # 7  squeezeValue.
        _at(axes, AXIS_TOUCHPAD),           # 8  touchpadValue x.
        _at(axes, AXIS_TOUCHPAD + 1),       # 9  touchpadValue y.
        _at(axes, AXIS_THUMBSTICK),         # 10 thumbstickValue x.
        _at(axes, AXIS_THUMBSTICK + 1),     # 11 thumbstickValue y.
    ]


def decode(msg, press_th):
    """One raw WebXR packet -> (head pose, left hand, right hand).

    Poses stay position + xyzw quaternion in WebXR local-floor space, which
    matches the native app's OpenXR STAGE space, as gripSpace matches its
    grip pose -- so the daemon's frame math is identical for both paths.
    """
    return (_pose(msg.get("head")),
            _hand(msg.get("left"), press_th),
            _hand(msg.get("right"), press_th))


def connected():
    """How many pages currently hold a WebSocket open."""
    return len(_clients)


async def _index(request):
    """Serve the single-file webapp, uncached so edits land on reload."""
    return web.FileResponse(INDEX, headers={"Cache-Control": "no-store"})


async def _ws(request):
    """Read raw WebXR packets from one page until it disconnects."""
    ws = web.WebSocketResponse(heartbeat=WS_HEARTBEAT_S,
                               max_msg_size=MAX_MSG_SIZE)
    await ws.prepare(request)
    _clients.add(ws)
    peer = request.remote
    print(f"[quest] webxr page connected: {peer}", flush=True)
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                request.app["on_state"](json.loads(msg.data))
            except Exception as e:
                print(f"[quest] webxr packet dropped: {e}", flush=True)
    finally:
        _clients.discard(ws)
        print(f"[quest] webxr page gone: {peer}", flush=True)
    return ws


def _broadcast(msg):
    """Fan one text frame out to every page. Runs on the server loop."""
    for ws in list(_clients):
        if not ws.closed:
            _loop.create_task(ws.send_str(msg))


def send_haptic(hand, frequency, amplitude, duration):
    """Push a haptic command to every connected page, from any thread.

    WebXR's GamepadHapticActuator.pulse takes only intensity and duration;
    frequency rides along so the page can use it if the API ever grows one.
    """
    if _loop is None or not _clients:
        return
    _loop.call_soon_threadsafe(_broadcast, json.dumps({
        "type": "haptic",
        "hand": int(hand),
        "frequency": float(frequency),
        "amplitude": float(amplitude),
        "duration": float(duration),
    }))


def _run(port, cert_file, key_file, on_state):
    """Event loop thread: TLS site on `port`, forever."""
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    app = web.Application()
    app["on_state"] = on_state
    app.router.add_get("/", _index)
    app.router.add_get("/ws", _ws)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_file, key_file)

    runner = web.AppRunner(app)
    _loop.run_until_complete(runner.setup())
    _loop.run_until_complete(
        web.TCPSite(runner, "0.0.0.0", port, ssl_context=ctx).start())
    _loop.run_forever()


def start(port, cert_file, key_file, on_state):
    """Serve the webapp on `port` from a daemon thread."""
    threading.Thread(target=_run, daemon=True,
                     args=(port, cert_file, key_file, on_state)).start()
