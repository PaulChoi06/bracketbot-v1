"""LiveKit transport for the remote_session daemon.

Backend polling and tokens, plus the camera and mic tracks. daemon.py owns the
room, the data channel and the control loop.
"""

import asyncio
import queue
import struct

import cv2
import httpx
import numpy as np
from livekit import rtc

from bbos import Config

CFG = Config("remote_session")
BACKEND_URL = CFG.bb_api_url.rstrip("/")
API_KEY = CFG.bb_api_key
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
SERIAL = open("/proc/device-tree/serial-number").read().strip("\x00").strip()
FPS = int(CFG.fps)

FIRST_FRAME_S = 5               # Wait for a camera's first frame.
NO_FRAMES_SLEEP_S = 60          # Park a feed task whose camera never came up.

# Haptics, robot -> app. Exactly 17 bytes little-endian: the headset drops any
# other length silently. Reliable, since a buzz is a one-shot event.
HAPTIC_TOPIC = "quest_haptic"
MAGIC_HAPTIC = b"BBQH"
_HAPTIC_FMT = "<4sBfff"         # magic, hand, freq, amp, dur.

AUDIO_RATE = 16000
AUDIO_CHUNK = 1600              # Samples per frame, both directions.
MIC_LOG_EVERY = 100


def push_queue(q: queue.Queue, item):
    """Put item on q, dropping the oldest entry if it is full."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

async def poll(client: httpx.AsyncClient, running: bool, error):
    """Report our state and return the backend's {command, roomId?}."""
    body = {"serial": SERIAL, "running": running}
    if error:
        body["error"] = error
    r = await client.post(f"{BACKEND_URL}/v1/teleop/poll", json=body,
                          headers=HEADERS)
    r.raise_for_status()
    return r.json()

async def mint_token(client: httpx.AsyncClient, room_id: str):
    """Get a LiveKit url + token for one room."""
    r = await client.post(
        f"{BACKEND_URL}/v1/livekit/token",
        json={"room": room_id},
        headers=HEADERS,
    )
    r.raise_for_status()
    return r.json()

def rgb_to_i420(frame: np.ndarray) -> bytes:
    """RGB frame -> I420 bytes, the format LiveKit wants."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420).tobytes()

async def publish_feed(room: rtc.Room, q: queue.Queue, track_name: str):
    """Publish one camera queue as a video track, forever."""
    try:
        first = await asyncio.get_event_loop().run_in_executor(
            None, q.get, True, FIRST_FRAME_S)
    except queue.Empty:
        print(f"  [skip] no frames for {track_name} after "
              f"{FIRST_FRAME_S}s — camera daemon down?", flush=True)
        while True:
            await asyncio.sleep(NO_FRAMES_SLEEP_S)
    h, w = first.shape[:2]
    w = w & ~1
    h = h & ~1

    source = rtc.VideoSource(w, h)
    track = rtc.LocalVideoTrack.create_video_track(track_name, source)
    opts = rtc.TrackPublishOptions()
    opts.source = rtc.TrackSource.SOURCE_CAMERA
    await room.local_participant.publish_track(track, opts)
    print(f"  published {track_name} ({w}x{h})", flush=True)

    source.capture_frame(
        rtc.VideoFrame(w, h, rtc.VideoBufferType.I420,
                       rgb_to_i420(first[:h, :w]))
    )

    while True:
        try:
            img = q.get_nowait()
            source.capture_frame(
                rtc.VideoFrame(w, h, rtc.VideoBufferType.I420,
                               rgb_to_i420(img[:h, :w]))
            )
        except queue.Empty:
            pass
        await asyncio.sleep(1 / FPS)

async def handle_audio(track: rtc.Track, audio_queue: queue.Queue):
    """Chunk an incoming browser audio track onto audio_queue."""
    stream = rtc.AudioStream(track, sample_rate=AUDIO_RATE, num_channels=1)
    buf = np.empty(0, dtype=np.int16)
    seen_first = False
    async for event in stream:
        if not seen_first:
            print(f"  [audio] first browser audio frame: "
                  f"samples={len(event.frame.data) // 2} "
                  f"sr={event.frame.sample_rate} "
                  f"ch={event.frame.num_channels}", flush=True)
            seen_first = True
        samples = np.frombuffer(event.frame.data, dtype=np.int16)
        buf = np.concatenate([buf, samples])
        while len(buf) >= AUDIO_CHUNK:
            chunk = buf[:AUDIO_CHUNK].reshape(AUDIO_CHUNK, 1)
            push_queue(audio_queue, chunk)
            buf = buf[AUDIO_CHUNK:]

async def publish_mic(room: rtc.Room, mic_queue: queue.Queue):
    """Publish the robot mic queue as an audio track, forever."""
    source = rtc.AudioSource(AUDIO_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("robot-mic", source)
    opts = rtc.TrackPublishOptions()
    opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, opts)
    print("  published robot-mic", flush=True)

    loop = asyncio.get_event_loop()
    n = 0
    while True:
        chunk = await loop.run_in_executor(None, mic_queue.get)
        samples = chunk[:, 0] if chunk.ndim == 2 else chunk
        frame = rtc.AudioFrame(
            data=samples.tobytes(),
            samples_per_channel=AUDIO_CHUNK,
            sample_rate=AUDIO_RATE,
            num_channels=1,
        )
        await source.capture_frame(frame)
        n += 1
        if n == 1 or n % MIC_LOG_EVERY == 0:
            print(f"  [audio] mic frame #{n} published", flush=True)


def haptic_packet(hand, freq, amp, dur):
    """Pack one buzz the way the headset expects it."""
    return struct.pack(_HAPTIC_FMT, MAGIC_HAPTIC, int(hand), float(freq),
                       float(amp), float(dur))


async def publish_haptics(room: rtc.Room, q: queue.Queue):
    """Forward queued haptic packets to the app, forever."""
    while True:
        try:
            pkt = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(1 / FPS)
            continue
        try:
            await room.local_participant.publish_data(
                pkt, reliable=True, topic=HAPTIC_TOPIC)
        except Exception as e:
            print(f"  [!] haptic publish failed: {e}", flush=True)
