"""camera daemon: publish MJPEG (and head RGB) from every V4L2 capture node.

One thread per camera, each a linear loop on its own fd. A lost camera is
reopened on a timer; the Writers outlive that, so no topic vanishes. A monitor
thread publishes camera.<name>.status, since a live topic no longer proves one.
"""

import ctypes
import errno
import fcntl
import glob
import mmap
import os
import select
import threading
import time

import numpy as np
import v4l2
from bbjpeg import BBJpeg

from bbos import Writer
# Registry direct, constants for its side effect: register only our own
# Configs and Types, instead of the full scan bbos.Config would trigger.
from bbos.registry import Config, Type
import constants


# ============================================================================
# Constants
# ============================================================================
# Units.
MS_PER_S = 1000
KIB = 1024

# Device lifecycle.
RETRY_S = 2.0                   # Wait before reopening a lost camera.
DQBUF_TIMEOUT_S = 3.0           # Bound on the wait for one frame.
DEFAULT_NUM_BUFS = 2            # Few buffers: head frames are large (CMA).
MONO_RESAMPLE_S = 1.0           # How often to re-sample the clock offset.

# Publishing.
WRIST_MAX_PIXELS = 640 * 480    # Past this a camera is the head: bigger shm.
WRITER_BUF_MS = 500

# Status monitor.
STATUS_PERIOD_S = 1.0           # Monitor tick, so also the fps window.

# Logging.
LOG_FIRST = 5                   # Log every publish up to this many, then...
LOG_EVERY = 100                 # ...one in this many.

CAMERAS = [
    # Config name, topic, publish RGB.
    ("cam_head", "camera.head", True),
    ("cam_left", "camera.left", False),
    ("cam_right", "camera.right", False),
]


# ============================================================================
# Shared state
# ============================================================================
# One physical camera must not be opened twice. Guarded across resolve+open.
_open_lock = threading.Lock()
_opened_real_paths = set()

# CLOCK_REALTIME - CLOCK_MONOTONIC, resampled to track NTP steps. Converts
# the V4L2 hardware timebase to the Unix epoch.
_mono_real = [0, 0.0]


# ============================================================================
# Helpers
# ============================================================================
def _mono_to_real_ns():
    """Offset to add to a monotonic ns stamp to get epoch ns."""
    now = time.monotonic()
    if now - _mono_real[1] > MONO_RESAMPLE_S:
        _mono_real[0] = time.time_ns() - time.monotonic_ns()
        _mono_real[1] = now
    return _mono_real[0]


def ioctl_retry(fd, req, arg):
    """Call ioctl, riding out EINTR."""
    while True:
        try:
            return fcntl.ioctl(fd, req, arg)
        except InterruptedError:
            pass


# ============================================================================
# Device resolution
# ============================================================================
def _usb_device_dir(sysfs_path: str):
    """Walk up a sysfs path to the USB device dir, the one with idVendor."""
    d = sysfs_path
    while d and d not in ("/", "/sys"):
        if os.path.exists(os.path.join(d, "idVendor")):
            return d
        d = os.path.dirname(d)
    return None


def resolve_via_sibling_arm(CFG) -> "str | None":
    """Find the wrist camera sharing a USB hub with CFG.sibling_arm.

    Follows arm symlink -> arm USB device -> parent hub -> the capture node
    under it, which survives video-node renumbering and port swaps. None if
    the arm is not up.
    """
    arm_link = getattr(CFG, "sibling_arm", "")
    card_substr = getattr(CFG, "card_substr", "")
    if not arm_link or not os.path.exists(arm_link):
        return None
    tty = os.path.basename(os.path.realpath(arm_link))
    arm_usb = _usb_device_dir(os.path.realpath(f"/sys/class/tty/{tty}/device"))
    if not arm_usb:
        return None
    hub = os.path.dirname(arm_usb)  # The wrist hub shared by cam and arm.
    for sysdev in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            if open(os.path.join(sysdev, "index")).read().strip() != "0":
                continue
            name = open(os.path.join(sysdev, "name")).read().strip()
            if card_substr and card_substr not in name:
                continue
            cam_usb = _usb_device_dir(os.path.realpath(sysdev))
            if not cam_usb or os.path.dirname(cam_usb) != hub:
                continue
            candidate = "/dev/" + os.path.basename(sysdev)
            if os.path.realpath(candidate) in _opened_real_paths:
                continue
            return candidate
        except Exception:
            continue
    return None


def resolve_video_device(CFG) -> str:
    """Resolve a camera config to a device path, skipping opened devices.

    Caller must hold _open_lock.
    """
    card_substr = getattr(CFG, "card_substr", "")

    # No name-based fallback: an absent arm must raise and retry rather than
    # risk grabbing the wrong wrist camera.
    if getattr(CFG, "sibling_arm", ""):
        cam = resolve_via_sibling_arm(CFG)
        if cam:
            return cam
        raise FileNotFoundError(
            f"wrist camera unresolved: sibling arm {CFG.sibling_arm} "
            f"not found (arm unplugged or arm daemon not up yet?)"
        )

    # The head card name is unique, so the first index=0 node is right.
    if not card_substr:
        raise FileNotFoundError(
            "camera config has neither sibling_arm nor card_substr")

    for sysdev in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            if open(os.path.join(sysdev, "index")).read().strip() != "0":
                continue
            name = open(os.path.join(sysdev, "name")).read().strip()
            if card_substr not in name:
                continue
            candidate = "/dev/" + os.path.basename(sysdev)
            if os.path.realpath(candidate) in _opened_real_paths:
                continue
            return candidate
        except Exception:
            continue

    raise FileNotFoundError(
        f"could not find camera '{card_substr}' (capture node index=0); "
        f"none present or all matching devices already in use"
    )


# ============================================================================
# Capture device
# ============================================================================
class CameraDevice:
    """One resolved, negotiated, streaming V4L2 MJPEG capture device.

    dequeue() polls this fd alone, so there is no multiplexer. The wait is
    bounded because an unplugged UVC device never wakes a blocking DQBUF,
    which would park the thread in the driver for good.
    """

    def __init__(self, CFG):
        """Resolve, open, negotiate and start streaming, or close and raise."""
        self.fd = None
        self.mmaps = []
        self.real_path = None
        self.poller = None
        with _open_lock:
            dev_path = resolve_video_device(CFG)
            self.real_path = os.path.realpath(dev_path)
            self.fd = os.open(dev_path, os.O_RDWR)
            _opened_real_paths.add(self.real_path)
        try:
            self._negotiate(CFG)
            self._stream_on(CFG)
        except BaseException:
            self.close()
            raise

    def _negotiate(self, CFG):
        """Set format and frame rate, failing loudly if the driver adjusts."""
        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        ioctl_retry(self.fd, v4l2.VIDIOC_G_FMT, fmt)
        fmt.fmt.pix.width = CFG.width
        fmt.fmt.pix.height = CFG.height
        fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_MJPEG
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        ioctl_retry(self.fd, v4l2.VIDIOC_S_FMT, fmt)
        # S_FMT may silently adjust; the decoder and shm types are sized from
        # the config, so a mismatch must fail here, not downstream.
        got = (fmt.fmt.pix.width, fmt.fmt.pix.height, fmt.fmt.pix.pixelformat)
        want = (CFG.width, CFG.height, v4l2.V4L2_PIX_FMT_MJPEG)
        if got != want:
            raise OSError(
                errno.EINVAL,
                f"S_FMT negotiated {got[0]}x{got[1]} fourcc=0x{got[2]:08x}, "
                f"wanted {want[0]}x{want[1]} MJPG",
            )

        parm = v4l2.v4l2_streamparm()
        parm.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        parm.parm.capture.timeperframe.numerator = 1
        parm.parm.capture.timeperframe.denominator = CFG.rate
        ioctl_retry(self.fd, v4l2.VIDIOC_S_PARM, parm)
        tf = parm.parm.capture.timeperframe
        if (tf.numerator == 0
                or round(tf.denominator / tf.numerator) != CFG.rate):
            raise OSError(
                errno.EINVAL,
                f"S_PARM granted {tf.denominator}/{tf.numerator} fps, "
                f"wanted {CFG.rate}",
            )

    def _stream_on(self, CFG):
        """Request and map the buffers, queue them, and start the stream."""
        req = v4l2.v4l2_requestbuffers()
        req.count = getattr(CFG, "num_bufs", DEFAULT_NUM_BUFS)
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP
        ioctl_retry(self.fd, v4l2.VIDIOC_REQBUFS, req)
        if req.count < 1:
            raise OSError(errno.ENOMEM, "driver allocated 0 buffers")
        for i in range(req.count):
            buf = v4l2.v4l2_buffer()
            buf.index = i
            buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            ioctl_retry(self.fd, v4l2.VIDIOC_QUERYBUF, buf)
            self.mmaps.append(mmap.mmap(
                self.fd, buf.length, mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset))
            ioctl_retry(self.fd, v4l2.VIDIOC_QBUF, buf)
        buf_type = ctypes.c_int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        ioctl_retry(self.fd, v4l2.VIDIOC_STREAMON, buf_type)
        self.poller = select.poll()
        self.poller.register(self.fd, select.POLLIN)

    def dequeue(self):
        """Wait for a frame; return (v4l2 buf, jpeg memoryview, epoch ns).

        Raises on a disconnected or stalled device so the caller reopens it.
        """
        events = self.poller.poll(DQBUF_TIMEOUT_S * MS_PER_S)
        if not events:
            raise TimeoutError(f"no frame for {DQBUF_TIMEOUT_S:.0f}s")
        revents = events[0][1]
        if revents & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
            raise OSError(errno.ENODEV,
                          f"capture device gone (revents=0x{revents:x})")
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        ioctl_retry(self.fd, v4l2.VIDIOC_DQBUF, buf)
        mono_ns = (buf.timestamp.secs * 1_000_000_000
                   + buf.timestamp.usecs * 1000)
        raw = memoryview(self.mmaps[buf.index])[:buf.bytesused]
        return buf, raw, mono_ns + _mono_to_real_ns()

    def requeue(self, buf):
        """Hand one buffer back to the driver."""
        ioctl_retry(self.fd, v4l2.VIDIOC_QBUF, buf)

    def close(self):
        """Stop streaming, unmap, and release the fd. Safe to call twice."""
        if self.fd is not None:
            buf_type = ctypes.c_int(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMOFF, buf_type)
            except OSError:
                pass
        for mm in self.mmaps:
            try:
                mm.close()
            except (BufferError, OSError):
                pass
        self.mmaps = []
        if self.fd is not None:
            # Release driver buffers; helps some UVC reopen edge cases.
            req = v4l2.v4l2_requestbuffers()
            req.count = 0
            req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            req.memory = v4l2.V4L2_MEMORY_MMAP
            try:
                fcntl.ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req)
            except OSError:
                pass
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        if self.real_path is not None:
            with _open_lock:
                _opened_real_paths.discard(self.real_path)
            self.real_path = None


# ============================================================================
# Publish loop
# ============================================================================
def stream(tag, cam, publish_every, jpeg_capacity,
           jpeg_writer, rgb_writer, decoder, st):
    """Publish frames until a device or decoder error is raised.

    Per-frame problems (oversize or corrupt JPEG) are logged and dropped here;
    anything that escapes is a device-level error for the caller.
    """
    captured = 0
    published = 0
    while True:
        buf, raw, ts = cam.dequeue()
        captured += 1
        publish = (captured - 1) % publish_every == 0
        rgb_pending = False
        n = 0
        t0 = time.monotonic()
        try:
            if publish:
                n = raw.nbytes
                if n == 0 or n > jpeg_capacity:
                    print(f"[!] {tag}: dropping {n}-byte JPEG "
                          f"(shm capacity {jpeg_capacity})", flush=True)
                    publish = False
            if publish:
                with jpeg_writer.buf() as jb:
                    jb["jpeg_len"] = n
                    jb["jpeg"][:n] = np.frombuffer(raw, dtype=np.uint8)
                    jb["timestamp"] = ts
                if decoder is not None:
                    # submit() copies into hardware staging first, so the
                    # buffer can be requeued while the decode runs.
                    try:
                        decoder.submit(raw)
                        rgb_pending = True
                    except RuntimeError as e:
                        # One corrupt bitstream must not take down the camera.
                        print(f"[!] {tag}: decode submit failed ({e}); "
                              f"no RGB this frame", flush=True)
        finally:
            raw.release()
            cam.requeue(buf)
        if not publish:
            continue
        t1 = time.monotonic()
        if rgb_pending:
            with rgb_writer.buf() as b:
                decoder.finish_into(b["rgb"])
                b["timestamp"] = ts
        t2 = time.monotonic()
        published += 1
        st.published += 1
        if published <= LOG_FIRST or published % LOG_EVERY == 0:
            print(
                f"[.] {tag}: captured={captured} published={published} "
                f"(jpeg={MS_PER_S * (t1 - t0):.1f}ms/{n // KIB}KB, "
                f"rgb={MS_PER_S * (t2 - t1):.1f}ms)",
                flush=True,
            )


def camera_thread(cfg_name, topic, publish_rgb, st):
    """Own one camera: open it, stream, and reopen forever on failure."""
    tag = topic.split(".")[1]
    CFG = Config(cfg_name)
    if CFG.fmt != "MJPEG":
        raise ValueError(f"{cfg_name}: only MJPEG is supported, got {CFG.fmt}")
    publish_every = int(getattr(CFG, "decimate", 1))
    if publish_every < 1:
        raise ValueError(
            f"{cfg_name}: decimate must be >= 1, got {publish_every}")
    # Two shm JPEG buffer sizes exist (head stereo vs wrist); pick by
    # resolution.
    is_head = CFG.width * CFG.height > WRIST_MAX_PIXELS
    jpeg_type = "camera_head_jpeg" if is_head else "camera_wrist_jpeg"
    jpeg_capacity = (constants.JPEG_BUF_SIZE_HEAD if is_head
                     else constants.JPEG_BUF_SIZE_WRIST)

    jpeg_writer = rgb_writer = None
    decoder = None
    while True:
        cam = None
        try:
            # Kept across device loss so no topic disappears. Never
            # `with Writer(...)`: its __exit__ swallows exceptions.
            if jpeg_writer is None:
                jpeg_writer = Writer(f"{topic}.jpeg", Type(jpeg_type),
                                     keeptime=False,
                                     buf_ms=WRITER_BUF_MS).__enter__()
            if publish_rgb and rgb_writer is None:
                rgb_writer = Writer(f"{topic}.rgb", Type("camera_head"),
                                    keeptime=False,
                                    buf_ms=WRITER_BUF_MS).__enter__()

            cam = CameraDevice(CFG)
            print(f"[+] {tag}: streaming {cam.real_path} "
                  f"{CFG.width}x{CFG.height}@{CFG.rate} MJPEG, "
                  f"publish 1/{publish_every}", flush=True)
            if publish_rgb and decoder is None:
                decoder = BBJpeg(rgb_writer, CFG.width, CFG.height)
            st.streaming = True
            try:
                stream(tag, cam, publish_every, jpeg_capacity,
                       jpeg_writer, rgb_writer, decoder, st)
            finally:
                # finally, not the handler: a BaseException would skip it.
                st.streaming = False
        except Exception as e:
            print(f"[!] {tag}: {type(e).__name__}: {e}; "
                  f"reopening in {RETRY_S:.0f}s", flush=True)
            # A failed finish_into leaves a frame pending in the decoder, which
            # would reject every later submit; rebuild it with the camera.
            if decoder is not None:
                try:
                    decoder.close()
                except Exception:
                    pass
                decoder = None
            if cam is not None:
                cam.close()
            time.sleep(RETRY_S)


# ============================================================================
# Status
# ============================================================================
class CamStatus:
    """Counters one camera thread writes and the monitor thread reads.

    No lock: exactly one writer per field, so no update can be lost.
    """

    def __init__(self, topic):
        self.topic = topic
        self.published = 0          # Monotonic across reopens.
        self.streaming = False


def status_thread(states):
    """Publish camera.<name>.status forever, one record per camera.

    Its own thread: a thread blocked in poll() cannot report its own stall.
    """
    live = []
    for st in states:
        try:
            live.append((st, Writer(f"{st.topic}.status",
                                    Type("camera_status"),
                                    keeptime=False).__enter__()))
        except Exception as e:
            print(f"[!] {st.topic}: status writer unavailable ({e}); "
                  f"continuing without", flush=True)
    if not live:
        return
    prev = {st.topic: st.published for st, _ in live}
    prev_t = time.monotonic()
    while True:
        time.sleep(STATUS_PERIOD_S)
        now = time.monotonic()
        dt = now - prev_t
        prev_t = now
        for st, w in live:
            # Per camera, not per tick: a raise must not skip the rest.
            try:
                # Sample once: the capture thread keeps writing underneath.
                published, streaming = st.published, st.streaming
                fps = (published - prev[st.topic]) / dt if dt > 0 else 0.0
                prev[st.topic] = published
                with w.buf() as b:
                    b["streaming"] = np.uint8(streaming)
                    b["fps"] = fps
            except Exception as e:
                print(f"[!] {st.topic}: status: {type(e).__name__}: {e}",
                      flush=True)


# ============================================================================
# Daemon
# ============================================================================
def main():
    """Run one thread per camera, plus the monitor, and wait on them."""
    threads = []
    states = []
    for cfg_name, topic, publish_rgb in CAMERAS:
        st = CamStatus(topic)
        states.append(st)
        t = threading.Thread(target=camera_thread, name=topic,
                             args=(cfg_name, topic, publish_rgb, st))
        t.start()
        threads.append(t)
    threading.Thread(target=status_thread, name="camera.status",
                     args=(states,), daemon=True).start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
