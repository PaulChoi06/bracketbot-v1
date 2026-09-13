"""dataset daemon: record episodes to npz + mkv and upload them.

dataset.flag toggles recording. Sensor samples and camera JPEGs accumulate in
an EpisodeBuffer, which rolls to a new chunk at memory_cap_mb; each chunk is
handed to a background worker that encodes, uploads, and on failure spills to
pending_dir for the next start to retry.
"""

import io
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from bbos import Config, Reader
from bbos.time import Loop


# ============================================================================
# Constants
# ============================================================================
BYTES_PER_MB = 1024 * 1024
UUID_CHARS = 8                  # Episode uid; collisions are not a concern.
ENTRY_OVERHEAD = 16             # Per-frame bookkeeping, for the mem tally.

# --- HTTP -------------------------------------------------------------------
HTTP_OK = range(200, 300)
HTTP_TIMEOUT_S = 30
LIST_PAGE_SIZE = 1000           # maxKeys per /v1/downloads/list page.
EP_INDEX_RE = re.compile(r"/ep(\d+)")   # Matches legacy and chunked ep keys.

# --- ffmpeg -----------------------------------------------------------------
FFMPEG_FPS = 30
PIPE_CHUNK = 65536
FFMPEG_DRAIN_S = 300            # Encode can outlast the pipe write by a lot.
FFMPEG_EXIT_S = 10
FFMPEG_KILL_S = 5
ERR_CHARS = 300                 # How much ffmpeg stderr to echo.

# --- Loop -------------------------------------------------------------------
LOOP_MS = 5                     # Out-paces the fastest sensor, so none drop.
FINALIZE_QUEUE_MAX = 2          # 1 active + 2 queued + 1 finalizing chunks.
SHUTDOWN_JOIN_S = 600
PROGRESS_EVERY = 10000          # Loop iterations between progress lines.


# ============================================================================
# Calibration
# ============================================================================
def load_calibration(cfg):
    """Read per-arm calibration from disk.

    Called at each episode start so hot-edits take effect without a restart.
    """
    cal = {}
    daemons_dir = Path(cfg.daemons_dir)
    for arm in cfg.cal_arms:
        arm_dir = daemons_dir / arm
        cal_path = arm_dir / "ranges.calibration.json"
        zeros_path = arm_dir / "zeros.txt"  # Deprecated, with the old arms.
        if cal_path.exists():
            with open(cal_path) as f:
                ranges = json.load(f)
            cal[f"cal_{arm}_min"] = np.array(ranges["cal_min"],
                                             dtype=np.float32)
            cal[f"cal_{arm}_max"] = np.array(ranges["cal_max"],
                                             dtype=np.float32)
            if "gravity_rest" in ranges:     # Deprecated, as above.
                cal[f"cal_{arm}_gravity_rest"] = np.array(
                    ranges["gravity_rest"], dtype=np.float32)
        if zeros_path.exists():
            cal[f"cal_{arm}_zeros"] = np.loadtxt(zeros_path, dtype=np.float32)
    return cal


# ============================================================================
# Episode buffer
# ============================================================================
class EpisodeBuffer:
    """One chunk of one episode, held in memory until it is finalized."""

    def __init__(self, prefix, run_name, text, ep_idx, ep_uuid, chunk_idx,
                 calibration):
        """Open an empty chunk; start_ns stamps it now."""
        self.prefix = prefix
        self.run_name = run_name
        self.text = text
        self.ep_idx = ep_idx
        self.ep_uuid = ep_uuid
        self.chunk_idx = chunk_idx
        self.calibration = calibration
        self.start_ns = time.time_ns()
        # Stamped by enqueue_finalize, not _build_npz, which runs later.
        self.end_ns = None
        self.sensors: dict[str, list[np.ndarray]] = defaultdict(list)
        self.cameras: dict[str, deque] = defaultdict(deque)
        self._mem = 0

    @property
    def mem_mb(self):
        """Bytes held, in MB, as the chunk-roll threshold reads it."""
        return self._mem / BYTES_PER_MB

    def add_sensor(self, name, sample):
        """Copy one sensor sample in; the reader reuses its buffer."""
        copy = sample.copy()
        self.sensors[name].append(copy)
        self._mem += copy.nbytes

    def add_camera(self, name, ts_ns, jpeg_bytes):
        """Append one JPEG frame with its hardware timestamp."""
        self.cameras[name].append((ts_ns, jpeg_bytes))
        self._mem += len(jpeg_bytes) + ENTRY_OVERHEAD


def _build_npz(episode, status="completed"):
    """Pack sensors, frame indices, meta and calibration into npz bytes."""
    arrays = {}
    for name, samples in episode.sensors.items():
        if samples:
            arrays[name] = np.array(samples)
    for cam, frames in episode.cameras.items():
        if frames:
            arrays[f"video_{cam}_timestamp_ns"] = np.array(
                [ts for ts, _ in frames], dtype=np.int64
            )
            arrays[f"video_{cam}_frame_idx"] = np.arange(len(frames),
                                                         dtype=np.int32)
    arrays["meta_start_time_ns"] = np.array(episode.start_ns)
    # end_ns comes from the recording loop; time.time_ns() here would run in
    # the upload worker and overrun the last sample under backlog.
    arrays["meta_end_time_ns"] = np.array(
        episode.end_ns if episode.end_ns is not None else time.time_ns())
    arrays["meta_status"] = np.array(status, dtype="U16")
    arrays["meta_label"] = np.array(episode.prefix, dtype="U500")
    # Empty when the run was launched without --text; preprocess then falls
    # back to its yaml task.
    arrays["meta_task"] = np.array(episode.text, dtype="U500")
    arrays["meta_chunk_idx"] = np.array(episode.chunk_idx, dtype=np.int32)
    arrays.update(episode.calibration)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _build_mkv(frames, fps=FFMPEG_FPS):
    """Encode JPEGs into an MKV container, draining `frames` as it goes.

    `frames` must be a deque and is empty when this returns. Peak RAM is about
    one camera's bytes plus the pipe buffers.
    """
    if not frames:
        return b""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "mjpeg", "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "copy", "-f", "matroska", "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError:
        print("[dataset] ffmpeg not found, skipping video encoding")
        return b""

    out_buf = io.BytesIO()

    def drain_stdout():
        while True:
            chunk = proc.stdout.read(PIPE_CHUNK)
            if not chunk:
                break
            out_buf.write(chunk)

    reader = threading.Thread(target=drain_stdout, daemon=True)
    reader.start()

    try:
        while frames:
            _ts, jpeg = frames.popleft()
            proc.stdin.write(jpeg)
        proc.stdin.close()
    except BrokenPipeError:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        proc.wait(timeout=FFMPEG_KILL_S)
        err = proc.stderr.read().decode()[:ERR_CHARS] if proc.stderr else ""
        print(f"[dataset] ffmpeg pipe broken: {err}")
        return b""

    reader.join(timeout=FFMPEG_DRAIN_S)
    try:
        rc = proc.wait(timeout=FFMPEG_EXIT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("[dataset] ffmpeg timed out")
        return b""

    if rc != 0:
        err = proc.stderr.read().decode()[:ERR_CHARS] if proc.stderr else ""
        print(f"[dataset] ffmpeg error: {err}")
        return b""

    return out_buf.getvalue()


# ============================================================================
# Upload
# ============================================================================
def _presign(key, content_type, cfg, api_key, episode=None):
    """Ask bb-server for a presigned PUT for one key."""
    body = {"key": key, "contentType": content_type}
    if episode:
        body["episode"] = episode
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{cfg.bb_api_url}/v1/uploads/presign",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[dataset] Presign failed for {key}: {e}")
        return None


def _upload_bytes(data, presigned, timeout):
    """PUT the bytes at the presigned url. True if the store accepted them."""
    headers = {"Content-Type": presigned.get("contentType",
                                             "application/octet-stream")}
    headers.update(presigned.get("headers") or {})
    req = urllib.request.Request(
        presigned["url"],
        data=data,
        method="PUT",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in HTTP_OK
    except Exception as e:
        print(f"[dataset] Upload PUT failed: {e}")
        return False


def _upload_one(key, data, content_type, cfg, api_key, episode=None):
    """Single-key upload with retry/backoff. Returns success bool."""
    retries = cfg.upload_retries
    for attempt in range(retries):
        presigned = _presign(key, content_type, cfg, api_key, episode)
        if presigned and _upload_bytes(data, presigned, cfg.upload_timeout_s):
            size_mb = len(data) / BYTES_PER_MB
            print(f"[dataset]   uploaded {key} ({size_mb:.1f}MB)")
            return True
        if attempt < retries - 1:
            wait = 2 ** attempt
            print(f"[dataset]   upload failed, "
                  f"retry {attempt + 2}/{retries} in {wait}s")
            time.sleep(wait)
    print(f"[dataset]   FAILED after {retries} attempts: {key}")
    return False


def _upload_all(files, cfg, api_key):
    """Upload key -> (bytes, content_type, episode). Returns the keys that won.

    Used by _retry_pending.
    """
    uploaded: set = set()
    for key, (data, content_type, episode) in files.items():
        if _upload_one(key, data, content_type, cfg, api_key, episode):
            uploaded.add(key)
    return uploaded


def _delete_keys(keys, cfg, api_key):
    """POST /v1/uploads/delete. Best-effort, log on failure."""
    if not keys or not api_key:
        return
    try:
        req = urllib.request.Request(
            f"{cfg.bb_api_url}/v1/uploads/delete",
            data=json.dumps({"keys": list(keys)}).encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            ok = resp.status in HTTP_OK
        if ok:
            print(f"[dataset] Deleted {len(keys)} orphaned key(s)")
        else:
            print(f"[dataset] Delete returned status {resp.status} "
                  f"for {len(keys)} key(s)")
    except Exception as e:
        print(f"[dataset] Delete request failed for {len(keys)} key(s): {e}")


# ============================================================================
# Drop coordination
# ============================================================================
# Shared between the main thread (drop path) and the finalize worker.
_drop_lock = threading.Lock()
_dropped_uuids: set = set()
_uploaded_keys: dict = defaultdict(list)


def _is_dropped(uid):
    """Report whether this episode uuid was dropped."""
    with _drop_lock:
        return uid in _dropped_uuids


def _record_or_undo(uid, key, cfg, api_key):
    """Record an uploaded key, or delete it if the episode was dropped.

    Returns True when the caller should bail: the key is already gone from S3.
    """
    with _drop_lock:
        if uid in _dropped_uuids:
            should_delete = True
        else:
            _uploaded_keys[uid].append(key)
            should_delete = False
    if should_delete:
        _delete_keys([key], cfg, api_key)
    return should_delete


# ============================================================================
# Pending spill
# ============================================================================
def _write_manifest_atomic(manifest_path: Path, manifest: dict):
    """Write manifest.json as tmp -> fsync -> rename.

    Survives SIGKILL without leaving a half-written manifest behind.
    """
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(manifest_path)


def _save_pending(files, cfg):
    """Spill files that failed to upload into pending_dir for a later retry."""
    if not files:
        return
    pending_dir = Path(cfg.pending_dir)
    ep_dir = pending_dir / (f"{int(time.time())}_{os.getpid()}_"
                            f"{uuid.uuid4().hex[:UUID_CHARS]}")
    ep_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for key, (data, content_type, episode) in files.items():
        safe_name = key.replace("/", "__")
        (ep_dir / safe_name).write_bytes(data)
        entry = {"content_type": content_type, "file": safe_name}
        if episode:
            entry["episode"] = episode
        manifest[key] = entry

    _write_manifest_atomic(ep_dir / "manifest.json", manifest)
    total_mb = sum(len(d) for d, _, _ in files.values()) / BYTES_PER_MB
    print(f"[dataset] Saved {len(files)} files "
          f"({total_mb:.1f}MB) to {ep_dir.name}")


def _quarantine(ep_dir: Path, reason: str):
    """Rename a bad pending dir aside so it stops being retried forever."""
    new_name = ep_dir.parent / f"corrupt_{int(time.time())}_{ep_dir.name}"
    try:
        ep_dir.rename(new_name)
        print(f"[dataset] Quarantined {ep_dir.name} -> "
              f"{new_name.name} ({reason})")
    except Exception as e:
        print(f"[dataset] Failed to quarantine {ep_dir.name}: {e}")


def _retry_pending(cfg, api_key):
    """Re-upload everything spilled by earlier runs, then clean up."""
    if not api_key:
        return
    pending_dir = Path(cfg.pending_dir)
    if not pending_dir.exists():
        return

    for ep_dir in sorted(pending_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        if ep_dir.name.startswith("corrupt_"):
            continue
        manifest_path = ep_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        # Wide guard: any failure loading this dir is contained and
        # quarantined, so the daemon never crashes on startup.
        try:
            manifest = json.loads(manifest_path.read_text())
            files = {}
            for key, info in manifest.items():
                file_path = ep_dir / info["file"]
                if file_path.exists():
                    files[key] = (file_path.read_bytes(),
                                  info["content_type"], info.get("episode"))
        except Exception as e:
            print(f"[dataset] Failed to load pending {ep_dir.name}: {e}")
            _quarantine(ep_dir, f"manifest load: {e}")
            continue

        if not files:
            shutil.rmtree(ep_dir, ignore_errors=True)
            continue

        print(f"[dataset] Retrying {len(files)} pending files "
              f"from {ep_dir.name}...")
        try:
            uploaded = _upload_all(files, cfg, api_key)
        except Exception as e:
            print(f"[dataset] Upload error during retry "
                  f"of {ep_dir.name}: {e}")
            continue

        if len(uploaded) == len(files):
            shutil.rmtree(ep_dir, ignore_errors=True)
            print(f"[dataset] Retry succeeded, cleaned up {ep_dir.name}")
        else:
            # Forget the ones that landed, so a restart does not re-upload.
            try:
                for key, info in list(manifest.items()):
                    if key in uploaded:
                        (ep_dir / info["file"]).unlink(missing_ok=True)
                        manifest.pop(key)
                _write_manifest_atomic(manifest_path, manifest)
            except Exception as e:
                print(f"[dataset] Failed to update manifest "
                      f"after partial retry: {e}")
            print(f"[dataset] Retry partial: {len(uploaded)}/{len(files)} "
                  f"succeeded, will try rest next restart")


# ============================================================================
# Finalize
# ============================================================================
def _list_remote_episode_indices(prefix, name, cfg, api_key):
    """Find the ep*.npz indices already stored for (prefix, name).

    Paginates, so a dataset past one page still yields the true max. Returns
    a list of ints, or None if the query failed.
    """
    if not api_key:
        return []

    list_prefix = f"datasets/{prefix}/{name}/episodes/"
    indices: list[int] = []
    cont_token = None
    try:
        while True:
            body = {"prefix": list_prefix, "maxKeys": LIST_PAGE_SIZE}
            if cont_token:
                body["continuationToken"] = cont_token
            req = urllib.request.Request(
                f"{cfg.bb_api_url}/v1/downloads/list",
                data=json.dumps(body).encode(),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode())
            for item in payload.get("items", []):
                key = item.get("key", "")
                m = EP_INDEX_RE.search(key)
                if m:
                    indices.append(int(m.group(1)))
            if not payload.get("isTruncated"):
                break
            cont_token = payload.get("nextContinuationToken")
            if not cont_token:
                break
    except Exception as e:
        print(f"[dataset] List query failed for {list_prefix}: {e}")
        return None
    return indices


def _finalize_worker(work_queue, cfg, api_key):
    """Pull episodes off the queue and finalize them, until sent None."""
    while True:
        item = work_queue.get()
        if item is None:
            work_queue.task_done()
            break
        episode, status = item
        try:
            _finalize(episode, cfg, api_key, status)
        except Exception as e:
            print(f"[dataset] Finalize worker error: {e}")
        finally:
            work_queue.task_done()


def _finalize(episode, cfg, api_key, status="completed"):
    """Encode one chunk, upload it, and spill whatever the network refused."""
    idx = episode.ep_idx
    uid = episode.ep_uuid
    chunk = episode.chunk_idx
    prefix = episode.prefix
    run = episode.run_name
    s3_base = f"datasets/{prefix}/{run}"

    if _is_dropped(uid):
        print(f"[dataset] ep{idx:06d}_{uid}_{chunk}: "
              f"dropped pre-finalize, skipping")
        episode.cameras.clear()
        episode.sensors.clear()
        return

    print(f"[dataset] Finalizing ep{idx:06d}_{uid}_{chunk} "
          f"({episode.mem_mb:.0f}MB, {status})...")

    npz_bytes = _build_npz(episode, status)
    npz_key = f"{s3_base}/episodes/ep{idx:06d}_{uid}_{chunk}.npz"
    ep_ref = {"dataset": f"{prefix}/{run}", "idx": idx,
              "uid": uid, "chunk": chunk}

    unsaved: dict = {}

    if api_key:
        if _upload_one(npz_key, npz_bytes, "application/octet-stream",
                       cfg, api_key, ep_ref):
            if _record_or_undo(uid, npz_key, cfg, api_key):
                return  # Dropped mid-upload, key already deleted.
        else:
            unsaved[npz_key] = (npz_bytes, "application/octet-stream", ep_ref)
    else:
        unsaved[npz_key] = (npz_bytes, "application/octet-stream", ep_ref)

    del npz_bytes  # Drop the reference before per-camera encoding.

    for cam in list(episode.cameras.keys()):
        frames = episode.cameras[cam]
        if not frames:
            continue
        if _is_dropped(uid):
            print(f"[dataset]   skip {cam} — episode dropped")
            frames.clear()
            continue
        print(f"[dataset]   encoding {cam}: {len(frames)} frames...")
        mkv_bytes = _build_mkv(frames)  # Destructive: empties `frames`.
        if not mkv_bytes:
            continue
        mkv_key = f"{s3_base}/video/{cam}_ep{idx:06d}_{uid}_{chunk}.mkv"
        if api_key:
            if _upload_one(mkv_key, mkv_bytes, "video/x-matroska",
                           cfg, api_key):
                if _record_or_undo(uid, mkv_key, cfg, api_key):
                    return  # Dropped mid-upload.
            else:
                unsaved[mkv_key] = (mkv_bytes, "video/x-matroska", None)
        else:
            unsaved[mkv_key] = (mkv_bytes, "video/x-matroska", None)

    if unsaved and not _is_dropped(uid):
        print(f"[dataset] {len(unsaved)} file(s) failed, "
              f"saving to disk for retry")
        _save_pending(unsaved, cfg)


# ============================================================================
# Daemon
# ============================================================================
def main():
    """Record on dataset.flag, roll chunks at the cap, upload in the worker."""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    CFG = Config("dataset")
    api_key = CFG.bb_api_key or None
    hostname = socket.gethostname()
    default_name = f"{hostname}_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"[dataset] Robot: {hostname}")
    print(f"[dataset] Default name: {default_name}")
    print(f"[dataset] API key: "
          f"{'present' if api_key else 'NOT FOUND (will save to disk)'}")
    print(f"[dataset] Chunk cap: {CFG.memory_cap_mb}MB "
          f"(logical episodes are unbounded; chunks rolled at this size)")
    print(f"[dataset] Upload retries: {CFG.upload_retries}")
    print(f"[dataset] Pending dir: {CFG.pending_dir}")

    if shutil.which("ffmpeg") is None:
        print("[dataset] FATAL: ffmpeg not found on PATH - refusing to start "
              "(camera video would be silently dropped). Install ffmpeg.")
        sys.exit(1)

    _retry_pending(CFG, api_key)

    # Bounded queue: a full one blocks the loop on a roll or stop-toggle, which
    # is the backpressure point. Watch for "Backpressure" lines on slow wifi.
    finalize_queue: queue.Queue = queue.Queue(maxsize=FINALIZE_QUEUE_MAX)
    finalize_thread = threading.Thread(
        target=_finalize_worker,
        args=(finalize_queue, CFG, api_key),
        daemon=True,
        name="finalize_worker",
    )
    finalize_thread.start()

    def enqueue_finalize(ep, status):
        # Stamp the close time here, in the recording loop, so meta_end says
        # when data stopped and not when the worker got to it.
        ep.end_ns = time.time_ns()
        depth = finalize_queue.qsize()
        if depth >= 1:
            print(f"[dataset] Backpressure: {depth} episode(s) already "
                  f"queued, upload worker is behind — this stop will block "
                  f"until it drains")
        finalize_queue.put((ep, status))

    sensor_readers = {}
    camera_readers = {}

    # keeptime=False: the single Loop trigger below paces us. Per-reader
    # keeptime would oblige a ready() on every reader each iteration, and
    # ready() copies the whole fixed-size record, so idle iterations would
    # memcpy every camera JPEG straight into the trash.
    for name, channel in CFG.sensor_channels.items():
        try:
            sensor_readers[name] = Reader(channel, keeptime=False)
        except Exception as e:
            print(f"[dataset] Skipping sensor {name}: {e}")

    for name, channel in CFG.camera_channels.items():
        try:
            camera_readers[name] = Reader(channel, keeptime=False)
        except Exception as e:
            print(f"[dataset] Skipping camera {name}: {e}")

    flag_reader = Reader("dataset.flag", keeptime=False)

    for r in (list(sensor_readers.values())
              + list(camera_readers.values()) + [flag_reader]):
        r.__enter__()

    loop_trigger = [0]
    Loop.init(loop_trigger)
    Loop.set_ms(LOOP_MS, loop_trigger)

    episode = None
    ep_count = 0
    # Per-(prefix, name) next index, seeded from bb-server on first toggle.
    # A chunk roll does not bump it; only toggle-off does.
    next_idx_for_dataset: dict = {}
    current_prefix = ""
    current_name = ""
    current_text = ""
    loop_count = 0
    stopping = [False]

    def handle_signal(signum, frame):
        stopping[0] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("[dataset] Ready. Waiting for dataset.flag...")

    try:
        while not stopping[0]:
            Loop.keeptime()
            if flag_reader.ready():
                data = flag_reader.data
                toggle = bool(data["toggle_episode"])
                drop = bool(data["drop_episode"])
                prefix = (data["prefix"].decode()
                          if isinstance(data["prefix"], bytes)
                          else str(data["prefix"])).strip("\x00")
                ds_name = (data["name"].decode()
                           if isinstance(data["name"], bytes)
                           else str(data["name"])).strip("\x00")
                # Guarded: an app built against the older schema publishes no
                # `text`, and the Reader adopts the writer's dtype.
                if "text" in data.dtype.names:
                    text = (data["text"].decode()
                            if isinstance(data["text"], bytes)
                            else str(data["text"])).strip("\x00")
                else:
                    text = ""

                if prefix:
                    current_prefix = prefix
                if ds_name:
                    current_name = ds_name
                # Not guarded like prefix/name: empty is a legitimate value,
                # and it must clear text latched from an earlier session.
                current_text = text

                # Each write with toggle=True flips recording state.
                if toggle:
                    if episode is None:
                        name = current_name or default_name
                        ds_key = (current_prefix, name)
                        if ds_key not in next_idx_for_dataset:
                            existing = _list_remote_episode_indices(
                                current_prefix, name, CFG, api_key)
                            if existing is None:
                                start = 0
                                print(f"[dataset] Could not query existing "
                                      f"episodes for {current_prefix}/{name}; "
                                      f"starting at 0 (UUID prevents "
                                      f"collisions)")
                            elif existing:
                                start = max(existing) + 1
                                print(f"[dataset] Found {len(existing)} "
                                      f"existing episodes for "
                                      f"{current_prefix}/{name}, "
                                      f"resuming at ep{start}")
                            else:
                                start = 0
                                print(f"[dataset] No existing episodes for "
                                      f"{current_prefix}/{name}, "
                                      f"starting at ep0")
                            next_idx_for_dataset[ds_key] = start
                        idx = next_idx_for_dataset[ds_key]
                        ep_uuid = uuid.uuid4().hex[:UUID_CHARS]
                        episode = EpisodeBuffer(current_prefix, name,
                                                current_text, idx, ep_uuid, 0,
                                                load_calibration(CFG))
                        print(f"[dataset] Episode {idx} STARTED "
                              f"(datasets/{current_prefix}/{name}, "
                              f"uid={ep_uuid}, chunk=0)")
                    else:
                        print(f"[dataset] Episode {episode.ep_idx} chunk "
                              f"{episode.chunk_idx} STOPPED "
                              f"({episode.mem_mb:.0f}MB), queued for upload")
                        enqueue_finalize(episode, "completed")
                        next_idx_for_dataset[
                            (episode.prefix, episode.run_name)] = (
                                episode.ep_idx + 1)
                        episode = None
                        ep_count += 1

                # Drop discards the active chunk and deletes any chunks of
                # this logical episode already uploaded.
                if drop and episode is not None:
                    uid = episode.ep_uuid
                    print(f"[dataset] Episode {episode.ep_idx} chunk "
                          f"{episode.chunk_idx} DROPPED "
                          f"({episode.mem_mb:.0f}MB active "
                          f"+ any uploaded chunks)")
                    with _drop_lock:
                        _dropped_uuids.add(uid)
                        keys_to_delete = list(_uploaded_keys.pop(uid, []))
                    if keys_to_delete:
                        _delete_keys(keys_to_delete, CFG, api_key)
                    episode = None

            # Deadman, like the arm daemons on ctrl loss: the flag writer is
            # gone with an episode open, so the closing edge never comes.
            if episode is not None and not flag_reader.readable:
                print(f"[dataset] Episode {episode.ep_idx} chunk "
                      f"{episode.chunk_idx} STOPPED (writer gone), "
                      f"queued for upload")
                enqueue_finalize(episode, "completed")
                next_idx_for_dataset[(episode.prefix, episode.run_name)] = (
                    episode.ep_idx + 1)
                episode = None
                ep_count += 1

            if episode is not None:
                for name, reader in sensor_readers.items():
                    if reader.ready():
                        episode.add_sensor(name, reader.data)

                for cam, reader in camera_readers.items():
                    if reader.ready():
                        d = reader.data
                        jpeg_len = int(d["jpeg_len"])
                        if jpeg_len > 0:
                            jpeg_bytes = bytes(d["jpeg"][:jpeg_len])
                            ts_ns = int(d["timestamp"].astype("int64"))
                            episode.add_camera(cam, ts_ns, jpeg_bytes)

                # Roll to the next chunk of the same logical episode: same
                # ep_idx and uuid, chunk_idx + 1, calibration reused.
                if episode.mem_mb >= CFG.memory_cap_mb:
                    print(f"[dataset] Chunk {episode.chunk_idx} full "
                          f"({episode.mem_mb:.0f}MB), rolling to chunk "
                          f"{episode.chunk_idx + 1}")
                    next_chunk = episode.chunk_idx + 1
                    saved_prefix = episode.prefix
                    saved_name = episode.run_name
                    saved_text = episode.text
                    saved_idx = episode.ep_idx
                    saved_uuid = episode.ep_uuid
                    saved_calibration = episode.calibration
                    enqueue_finalize(episode, "completed")
                    episode = EpisodeBuffer(saved_prefix, saved_name,
                                            saved_text, saved_idx, saved_uuid,
                                            next_chunk, saved_calibration)
            else:
                # Idle: keep the small sensor readers current, but do not
                # drain the cameras. ready() copies the whole JPEG buffer, and
                # a stale pre-roll frame is harmless since every frame carries
                # its own timestamp.
                for reader in sensor_readers.values():
                    reader.ready()

            loop_count += 1
            if loop_count % PROGRESS_EVERY == 0 and episode is not None:
                n_sensors = sum(len(v) for v in episode.sensors.values())
                n_frames = sum(len(v) for v in episode.cameras.values())
                print(f"[dataset] Recording ep{episode.ep_idx} "
                      f"chunk{episode.chunk_idx}: "
                      f"{episode.mem_mb:.0f}MB, "
                      f"sensors={n_sensors}, frames={n_frames}")

    finally:
        if episode is not None:
            print(f"[dataset] Shutdown: queueing active chunk "
                  f"{episode.chunk_idx} of episode {episode.ep_idx}")
            enqueue_finalize(episode, "completed")
            ep_count += 1

        print(f"[dataset] Draining finalize queue "
              f"({finalize_queue.qsize()} pending)...")
        finalize_queue.put(None)
        finalize_thread.join(timeout=SHUTDOWN_JOIN_S)
        if finalize_thread.is_alive():
            print("[dataset] Finalize worker did not finish in time, "
                  "exiting anyway")

        for r in (list(sensor_readers.values())
                  + list(camera_readers.values()) + [flag_reader]):
            try:
                r.__exit__(None, None, None)
            except Exception:
                pass

        print(f"[dataset] Stopped. {ep_count} episodes recorded.")


if __name__ == "__main__":
    main()
