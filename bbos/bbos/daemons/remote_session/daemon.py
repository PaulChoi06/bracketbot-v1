"""remote_session daemon: poll bb-cloud, run a LiveKit teleop session.

Polls every poll_interval_s. When the backend says 'start', connects to
LiveKit and runs the session until told to 'stop' or it ends on its own.
Readers and Writers are open only while a session is active.

The state machine lives server-side in Redis (see bb-cloud /v1/teleop/poll).
The poll body is {running, error?}; the response carries {command, roomId?}
where command is start, continue, stop or idle.
"""

import asyncio
import contextlib
import json
import os
import queue
import sys
import threading
import time

import cv2
import httpx
import numpy as np
from livekit import rtc

from bbos import Config, Reader, Type, Writer

# Sibling modules. The daemon runs as a script so its own dir is already on
# sys.path; this makes the import robust anyway.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quest import Quest  # noqa: E402
from session import (  # noqa: E402
    BACKEND_URL, CFG, SERIAL, handle_audio, haptic_packet, mint_token, poll,
    publish_feed, publish_haptics, publish_mic, push_queue,
)
from teleop import QuestArm  # noqa: E402

HOME_DURATION = float(CFG.home_duration_s)
DRIVE_SPEED_SCALE = float(CFG.drive_speed_scale)


# ============================================================================
# Constants
# ============================================================================
WHEEL_VEL_COMBOS = {
    "w":  (0.20, 0.20),
    "s":  (-0.20, -0.20),
    "a":  (-0.15, 0.15),
    "d":  (0.15, -0.15),
    "wa": (0.05, 0.28),
    "wd": (0.28, 0.05),
    "sa": (-0.05, -0.28),
    "sd": (-0.28, -0.05),
    "":   (0.0, 0.0),
}

# Quest right-thumbstick base drive; stick-click doubles both. Not
# drive-speed-scaled, that knob is for WASD.
QUEST_SPEED_LIN = 0.15
QUEST_SPEED_ANG = 1.0
QUEST_BOOST = 2.0

# Head LED while a session is live: blinking red, so anyone near the robot can
# tell a remote operator has it and not the local headset, which idles green.
# brightness -1 defers to the daemon's master. Ownership is the stream itself, so
# ending the session hands the LED back with nothing to undo on teardown.
LED_RGB = (255, 0, 0)
LED_BLINK_MS = 500

# --- Timing -----------------------------------------------------------------
QUEST_STALE_S = 0.5             # No packet for this long -> stop the base.
LED_PUBLISH_S = 0.5             # Under led.ctrl_stale_s, which is 3 s.
CAPS_WAIT_S = 5.0               # Wait for bbos_thread to publish capabilities.
JOIN_S = 2.0                    # Thread join budget on teardown.


# ============================================================================
# Movement
# ============================================================================
# Every operator input source resolves to one body twist [v, w] (m/s, rad/s)
# written to drive.ctrl.
def wheels_to_twist(vl, vr, robot_width):
    """Differential-drive wheel velocities (m/s) -> body twist [v, w]."""
    v = (vl + vr) / 2.0
    w = (vr - vl) / (2.0 * (robot_width * 0.5))
    return np.array([v, w], dtype=np.float32)


def wasd_twist(combo, robot_width):
    """Browser WASD combo -> body twist [v, w] (drive-speed-scaled)."""
    vl, vr = WHEEL_VEL_COMBOS.get(combo, (0.0, 0.0))
    return wheels_to_twist(vl * DRIVE_SPEED_SCALE, vr * DRIVE_SPEED_SCALE,
                           robot_width)


def stick_twist(leftright, fwdback, boost):
    """Quest right thumbstick -> body twist [v, w]."""
    m = QUEST_BOOST if boost else 1.0
    return np.array([-fwdback * QUEST_SPEED_LIN * m,
                     -leftright * QUEST_SPEED_ANG * m], dtype=np.float32)


# ============================================================================
# Helpers
# ============================================================================


def decode_jpeg(data: np.ndarray, jpeg_len: int):
    """Decode the first jpeg_len bytes to RGB, or None if undecodable."""
    buf = data[:jpeg_len].tobytes()
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================================
# bbos thread
# ============================================================================
def bbos_thread(queues: dict, caps_out: dict, caps_ready: threading.Event,
                stop_event: threading.Event):
    """Hold the Readers and Writers and run the data loop.

    Opens Readers unconditionally and each exclusive Writer in try/except,
    records what succeeded in caps_out, signals caps_ready, then loops until
    stop_event is set.
    """
    head_cfg = Config("cam_head")
    drive_cfg = Config("drive")
    cfg_l = Config("arm_left")
    cfg_r = Config("arm_right")

    with contextlib.ExitStack() as stack:
        # Readers: cameras/state/mic, never exclusive.
        r_head = stack.enter_context(Reader("camera.head.jpeg"))
        r_left_cam = stack.enter_context(Reader("camera.left.jpeg"))
        r_right_cam = stack.enter_context(Reader("camera.right.jpeg"))
        r_left_state = stack.enter_context(Reader("arm_left.state"))
        r_right_state = stack.enter_context(Reader("arm_right.state"))
        r_mic = stack.enter_context(Reader("mic.audio"))

        w_ctrl = w_spk = w_lt = w_rt = w_left = w_right = w_ds = None
        w_led = None

        try:
            w_ctrl = stack.enter_context(
                Writer("drive.ctrl", Type("drive_ctrl")))
            caps_out["drive"] = True
        except Exception as e:
            print(f"  [cap] drive unavailable: {e}", flush=True)

        # Episode recording. Only a Quest's right A drives it, so it is not a
        # browser capability and stays out of caps_out.
        try:
            w_ds = stack.enter_context(
                Writer("dataset.flag", Type("dataset_flag")))
        except Exception as e:
            print(f"  [cap] episode recording unavailable: {e}", flush=True)

        # keeptime=False: led.ctrl is a 200 ms topic and would otherwise pace
        # this loop down from fps.
        try:
            w_led = stack.enter_context(
                Writer("led.ctrl", Type("led_ctrl"), keeptime=False))
        except Exception as e:
            print(f"  [cap] head led unavailable: {e}", flush=True)

        try:
            w_spk = stack.enter_context(
                Writer("speaker.audio", Type("speaker_audio"),
                       keeptime=False, buf_ms=400))   # 400ms: see sound.py
            caps_out["speak"] = True
        except Exception as e:
            print(f"  [cap] speak unavailable: {e}", flush=True)

        try:
            w_lt = stack.enter_context(
                Writer("arm_left.torque", Type("arm_torque")))
            w_left = stack.enter_context(
                Writer("arm_left.ctrl", Type("arm_ctrl")))
            caps_out["arm_left"] = True
        except Exception as e:
            print(f"  [cap] arm_left unavailable: {e}", flush=True)

        try:
            w_rt = stack.enter_context(
                Writer("arm_right.torque", Type("arm_torque")))
            w_right = stack.enter_context(
                Writer("arm_right.ctrl", Type("arm_ctrl")))
            caps_out["arm_right"] = True
        except Exception as e:
            print(f"  [cap] arm_right unavailable: {e}", flush=True)

        caps_ready.set()
        print(f"  capabilities: {caps_out}", flush=True)

        head_q = queues["head"]
        left_q = queues["left"]
        right_q = queues["right"]
        cmd_queue = queues["cmd"]
        audio_queue = queues["audio"]
        manip_queue = queues["manip"]
        mic_q = queues["mic"]
        quest_q = queues["quest"]

        combo = ""
        quest = Quest(haptic_q=queues["haptic"], pack=haptic_packet)
        quest_state = None      # Latest decoded state; None => no quest.
        quest_state_t = 0.0     # Monotonic time of that packet.
        quest_twist = None      # Base twist from the right thumbstick.
        manip_on = False
        homing = False
        home_t0 = 0.0
        home_start_l = cfg_l.home.copy()
        home_start_r = cfg_r.home.copy()
        q_left = cfg_l.home.copy()
        q_right = cfg_r.home.copy()
        cur_left = cfg_l.home.copy()
        cur_right = cfg_r.home.copy()
        logged = set()

        # --- Quest arm teleop ---
        arms_ok = bool(caps_out.get("arm_left") and caps_out.get("arm_right"))
        quest_arm = QuestArm(
            cfg_l, cfg_r,
            {"left": w_left, "right": w_right,
             "left_torque": w_lt, "right_torque": w_rt,
             "speaker": w_spk, "dataset": w_ds},
            HOME_DURATION,
        )
        if arms_ok:
            arms_ok = quest_arm.init_ik()

        # Safety backstop: cut torque on ANY exit, including an exception
        # that skips the park below. Runs before the writers close.
        stack.callback(quest_arm.estop)
        stack.callback(quest_arm.sound.close)

        led_t = -1e9

        while not stop_event.is_set():
            now = time.monotonic()
            if w_led is not None and now - led_t >= LED_PUBLISH_S:
                with w_led.buf() as b:
                    b["rgb"] = np.array(LED_RGB, dtype=np.uint8)
                    b["brightness"] = np.int16(-1)
                    b["period_ms"] = np.uint16(LED_BLINK_MS * 2)
                led_t = now

            # Decode newest packets first so this iteration acts on a fresh
            # pose. quest_new gates the IK; the write below is every frame.
            quest_new = False
            try:
                while True:
                    if quest.feed(quest_q.get_nowait()):
                        quest_state = quest.state
                        quest_state_t = time.monotonic()
                        quest_new = True
            except queue.Empty:
                pass
            if quest_state is not None:
                if time.monotonic() - quest_state_t > QUEST_STALE_S:
                    quest_twist = np.zeros(2, dtype=np.float32)
                else:
                    th = quest_state["right_thumbstick"]
                    quest_twist = stick_twist(
                        float(th[0]), float(th[1]),
                        bool(quest_state["right_thumbstick_click"]))

            # Camera relay is browser-only (Quest can't receive video yet),
            # and decoding 3 JPEGs here would jitter the arm control.
            if quest_state is None and r_head.ready():
                jpeg_len = int(r_head.data["jpeg_len"])
                if jpeg_len > 0:
                    img = decode_jpeg(r_head.data["jpeg"], jpeg_len)
                    if img is not None:
                        left_half, _ = head_cfg.split(img)
                        if left_half is not None:
                            if "head" not in logged:
                                print(f"  head: {left_half.shape}", flush=True)
                                logged.add("head")
                            push_queue(head_q, left_half.copy())

            if quest_state is None and r_left_cam.ready():
                jpeg_len = int(r_left_cam.data["jpeg_len"])
                if jpeg_len > 0:
                    img = decode_jpeg(r_left_cam.data["jpeg"], jpeg_len)
                    if img is not None:
                        if "left" not in logged:
                            print(f"  left: {img.shape}", flush=True)
                            logged.add("left")
                        push_queue(left_q, img.copy())

            if quest_state is None and r_right_cam.ready():
                jpeg_len = int(r_right_cam.data["jpeg_len"])
                if jpeg_len > 0:
                    img = decode_jpeg(r_right_cam.data["jpeg"], jpeg_len)
                    if img is not None:
                        if "right" not in logged:
                            print(f"  right: {img.shape}", flush=True)
                            logged.add("right")
                        push_queue(right_q, img.copy())

            if r_left_state.ready():
                cur_left = r_left_state.data["pos"].copy()
            if r_right_state.ready():
                cur_right = r_right_state.data["pos"].copy()

            # A Quest client drives the arms via IK from its 6DoF poses;
            # otherwise the browser toggle enables/homes/holds them.
            _drift_l, _drift_r = quest_arm.precision_drift()
            quest.send_haptics(quest_arm.teleop_active, _drift_l, _drift_r)
            if quest_state is not None and arms_ok:
                quest_arm.step(quest, cur_left, cur_right, quest_new)
            else:
                # Browser manipulation toggle, only if both arms are there.
                try:
                    enabled = manip_queue.get_nowait()
                    want_enable = (bool(enabled)
                                   and caps_out.get("arm_left")
                                   and caps_out.get("arm_right"))
                    if want_enable and not manip_on:
                        if w_lt is not None:
                            with w_lt.buf() as b:
                                b["enable"] = np.ones(cfg_l.dof,
                                                      dtype=np.bool_)
                        if w_rt is not None:
                            with w_rt.buf() as b:
                                b["enable"] = np.ones(cfg_r.dof,
                                                      dtype=np.bool_)
                        home_start_l = cur_left.copy()
                        home_start_r = cur_right.copy()
                        q_left = cur_left.copy()
                        q_right = cur_right.copy()
                        home_t0 = time.monotonic()
                        homing = True
                        manip_on = True
                        print("  Arms enabled, homing...", flush=True)
                    elif not enabled and manip_on:
                        if w_lt is not None:
                            with w_lt.buf() as b:
                                b["enable"] = np.zeros(cfg_l.dof,
                                                       dtype=np.bool_)
                        if w_rt is not None:
                            with w_rt.buf() as b:
                                b["enable"] = np.zeros(cfg_r.dof,
                                                       dtype=np.bool_)
                        manip_on = False
                        homing = False
                        print("  Arms disabled", flush=True)
                except queue.Empty:
                    pass

                if homing:
                    alpha = min((time.monotonic() - home_t0)
                                / HOME_DURATION, 1.0)
                    q_left = home_start_l + alpha * (cfg_l.home - home_start_l)
                    q_right = home_start_r + alpha * (cfg_r.home
                                                      - home_start_r)
                    if alpha >= 1.0:
                        homing = False
                        q_left = cfg_l.home.copy()
                        q_right = cfg_r.home.copy()
                        print("  Homing complete", flush=True)
                elif not manip_on:
                    q_left = cur_left.copy()
                    q_right = cur_right.copy()

                if (caps_out.get("arm_left") and w_left is not None
                        and w_left.ready()):
                    w_left["pos"] = q_left
                if (caps_out.get("arm_right") and w_right is not None
                        and w_right.ready()):
                    w_right["pos"] = q_right

            try:
                combo = cmd_queue.get_nowait()
            except queue.Empty:
                pass

            if (caps_out.get("drive") and w_ctrl is not None
                    and w_ctrl.ready()):
                if quest_twist is not None:
                    w_ctrl["twist"] = quest_twist
                else:
                    w_ctrl["twist"] = wasd_twist(combo, drive_cfg.robot_width)

            # Audio relay is browser-only too; skip it for a Quest session.
            if (quest_state is None and caps_out.get("speak")
                    and w_spk is not None):
                try:
                    audio = audio_queue.get_nowait()
                    w_spk["audio"] = audio
                    if "spk_first" not in logged:
                        print(f"  [audio] first speaker chunk written "
                              f"shape={getattr(audio, 'shape', None)}",
                              flush=True)
                        logged.add("spk_first")
                except queue.Empty:
                    pass

            if quest_state is None and r_mic.ready():
                if "mic_first" not in logged:
                    print(f"  [audio] first mic sample readable "
                          f"shape={r_mic.data['audio'].shape}", flush=True)
                    logged.add("mic_first")
                push_queue(mic_q, r_mic.data["audio"].copy())

        # Session ending: park the arms through the startup waypoints
        # before cutting torque, so they don't free-fall from a raised pose.
        quest_arm.park(cur_left, cur_right)
        print("  bbos_thread stop_event received", flush=True)


# ============================================================================
# Session lifetime
# ============================================================================
async def run_session(client: httpx.AsyncClient, room_id: str):
    """Run one teleop session until cancelled or the user disconnects.

    Mints a token, opens R/W, connects, publishes tracks. Raises with a short
    reason string on setup failure.
    """
    try:
        tok = await mint_token(client, room_id)
    except Exception as e:
        raise RuntimeError("token_mint_failed") from e

    room = rtc.Room()
    queues = {
        "head": queue.Queue(maxsize=2),
        "left": queue.Queue(maxsize=2),
        "right": queue.Queue(maxsize=2),
        "cmd": queue.Queue(maxsize=8),
        "audio": queue.Queue(maxsize=8),
        "manip": queue.Queue(maxsize=4),
        "mic": queue.Queue(maxsize=8),
        "quest": queue.Queue(maxsize=4),   # Raw packets, newest-wins.
        "haptic": queue.Queue(maxsize=8),  # Buzzes, robot -> app.
    }
    caps: dict = {"drive": False, "speak": False,
                  "arm_left": False, "arm_right": False}
    caps_ready = threading.Event()
    stop_event = threading.Event()
    disc_event = asyncio.Event()

    @room.on("data_received")
    def on_data(pkt: rtc.DataPacket):
        try:
            if pkt.topic == "movement":
                keys = json.loads(pkt.data).get("keys", [])
                combo = "".join(k for k in ["w", "s", "a", "d"] if k in keys)
                push_queue(queues["cmd"], combo)
            elif pkt.topic == "manipulation":
                enabled = json.loads(pkt.data).get("enabled", False)
                push_queue(queues["manip"], enabled)
            elif pkt.topic == "quest_state":
                push_queue(queues["quest"], bytes(pkt.data))
        except Exception as e:
            print(f"  [!] data_received error: {e}", flush=True)

    @room.on("track_subscribed")
    def on_track(track: rtc.Track, _pub, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"  audio track from {participant.identity}", flush=True)
            asyncio.create_task(handle_audio(track, queues["audio"]))

    loop = asyncio.get_event_loop()

    @room.on("participant_disconnected")
    def on_disc(participant: rtc.RemoteParticipant):
        print(f"  participant disconnected: {participant.identity}",
              flush=True)
        push_queue(queues["cmd"], "")
        push_queue(queues["manip"], False)
        loop.call_soon_threadsafe(disc_event.set)

    thread = threading.Thread(
        target=bbos_thread,
        args=(queues, caps, caps_ready, stop_event),
        name="bbos_thread",
        daemon=True,
    )
    thread.start()

    # Wait briefly for bbos_thread to open R/W and publish capabilities.
    await loop.run_in_executor(None, caps_ready.wait, CAPS_WAIT_S)

    try:
        await room.connect(tok["url"], tok["token"])
    except Exception as e:
        stop_event.set()
        thread.join(timeout=JOIN_S)
        raise RuntimeError("livekit_connect_failed") from e

    print(f"  connected to LiveKit room={tok.get('room')}", flush=True)

    # Publish capabilities so the UI can disable unavailable controls.
    try:
        await room.local_participant.publish_data(
            json.dumps(caps).encode(), reliable=True, topic="capabilities")
    except Exception as e:
        print(f"  [!] capabilities publish failed: {e}", flush=True)

    tasks = [
        asyncio.create_task(publish_feed(room, queues["left"], "cam-left")),
        asyncio.create_task(publish_feed(room, queues["head"], "cam-wrist")),
        asyncio.create_task(publish_feed(room, queues["right"], "cam-right")),
        asyncio.create_task(publish_mic(room, queues["mic"])),
        asyncio.create_task(publish_haptics(room, queues["haptic"])),
    ]
    disc_task = asyncio.create_task(disc_event.wait())

    try:
        done, pending = await asyncio.wait(
            [*tasks, disc_task], return_when=asyncio.FIRST_COMPLETED)
        if disc_task in done:
            print("  session ending: participant disconnected", flush=True)
        else:
            # A publish task exited; probably an error.
            for t in done:
                exc = t.exception()
                if exc:
                    print(f"  [!] task exited with {exc}", flush=True)
    finally:
        for t in tasks:
            t.cancel()
        if not disc_task.done():
            disc_task.cancel()
        with contextlib.suppress(Exception):
            await room.disconnect()
        stop_event.set()
        await loop.run_in_executor(None, thread.join, JOIN_S)
        print("  session torn down", flush=True)


# ============================================================================
# Daemon
# ============================================================================
async def main():
    """Poll bb-cloud forever, starting and stopping sessions on command."""
    print("[shell] now running daemon: remote_session", flush=True)
    print(f"[+] serial={SERIAL} polling {BACKEND_URL}/v1/teleop/poll "
          f"every {CFG.poll_interval_s}s", flush=True)

    session_task = None
    last_error = None

    timeout = httpx.Timeout(float(CFG.http_timeout_s))
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            # Reap a finished session.
            running = False
            if session_task is not None:
                if session_task.done():
                    try:
                        session_task.result()
                    except asyncio.CancelledError:
                        pass
                    except RuntimeError as e:
                        last_error = str(e) if str(e) else "session_crashed"
                        print(f"  [!] session ended with {e}", flush=True)
                    except Exception as e:
                        last_error = "session_crashed"
                        print(f"  [!] session crashed: {e}", flush=True)
                    session_task = None
                else:
                    running = True

            err = last_error
            last_error = None

            try:
                resp = await poll(client, running, err)
            except Exception as e:
                print(f"  [!] poll failed: {e}", flush=True)
                await asyncio.sleep(CFG.poll_interval_s)
                continue

            cmd = resp.get("command")
            room_id = resp.get("roomId")

            if cmd == "start":
                if session_task is None and room_id:
                    print(f"  starting session room={room_id}", flush=True)
                    session_task = asyncio.create_task(
                        run_session(client, room_id))
            elif cmd == "stop":
                if session_task is not None:
                    print("  server said stop; cancelling session", flush=True)
                    session_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await session_task
                    session_task = None
            elif cmd == "idle":
                if session_task is not None:
                    print("  server said idle with active session; cancelling",
                          flush=True)
                    session_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await session_task
                    session_task = None
            # "continue": no-op.

            await asyncio.sleep(CFG.poll_interval_s)


if __name__ == "__main__":
    asyncio.run(main())
