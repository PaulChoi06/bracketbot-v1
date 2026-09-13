"""telemetry daemon: batch sensor samples into POSTs to bb-cloud.

One sample per field per tick, one POST every push_interval_s. Loop sets the
cadence; the POST runs on a background thread so it never blocks.
"""

import json
import queue
import socket
import sys
import threading
import urllib.error
import urllib.request

import numpy as np

from bbos import Config, Reader
from bbos.time import Loop

CFG = Config("telemetry")
HOSTNAME = socket.gethostname()


# ============================================================================
# Constants
# ============================================================================
TELEMETRY_PATH = "/v1/telemetry"
USER_AGENT = "bracketbot-telemetry/1.0"
POST_TIMEOUT_S = 30.0
HTTP_OK = range(200, 300)
INFLIGHT_MAX = 1                # Batches in flight; submit() refuses past it.
MS_PER_S = 1000


# ============================================================================
# Sample encoding
# ============================================================================
def _sample_to_rows(stream: str, sample,
                    last_ts_ns: dict[str, int]) -> list[dict]:
    """Walk a structured numpy scalar; emit one row per (field, channel).

    Skips a sample whose timestamp matches the last one emitted for this
    stream, so a paused writer does not re-send stale rows.
    """
    if sample is None:
        return []
    names = sample.dtype.names if hasattr(sample.dtype, "names") else None
    if not names or "timestamp" not in names:
        return []
    ts_field = sample["timestamp"]
    try:
        ns = int(np.asarray(ts_field).astype("int64"))
    except Exception:
        return []
    if ns <= 0 or ns == last_ts_ns.get(stream, 0):
        return []
    ts = np.datetime_as_string(ts_field, unit="us", timezone="UTC")
    rows: list[dict] = []
    for field in names:
        if field == "timestamp":
            continue
        val = np.asarray(sample[field])
        if val.dtype.kind not in ("f", "i", "u", "b"):
            continue
        if val.ndim == 0:
            v = float(val)
            if not np.isfinite(v):
                continue
            rows.append({"ts": ts, "stream": stream, "field": field,
                         "channel": 0, "value": v})
            continue
        flat = val.ravel().astype(np.float64)
        for ch, v in enumerate(flat):
            if not np.isfinite(v):
                continue
            rows.append({"ts": ts, "stream": stream, "field": field,
                         "channel": int(ch), "value": float(v)})
    last_ts_ns[stream] = ns
    return rows


def _snapshot_doc(stream: str, sample,
                  last_hash: dict[str, str]) -> dict | None:
    """Turn a usb.tree-style sample into a snapshots[] entry.

    None if the shape_hash has not moved since the last one we sent.
    """
    try:
        blob = bytes(sample["json"]).rstrip(b"\x00").decode()
        digest = bytes(sample["shape_hash"]).rstrip(b"\x00").decode()
    except (KeyError, ValueError, UnicodeDecodeError):
        return None
    if not blob or digest == last_hash.get(stream):
        return None
    try:
        doc = json.loads(blob)
    except json.JSONDecodeError:
        print(f"[!] {stream} carried unparseable json, skipped")
        return None
    last_hash[stream] = digest
    ts = np.datetime_as_string(sample["timestamp"], unit="us", timezone="UTC")
    return {"ts": str(ts), "kind": CFG.snapshot_streams[stream],
            "shape_hash": digest, "doc": doc}


# ============================================================================
# Upload
# ============================================================================
def _post_metrics(url: str, body: dict, api_key: str,
                  timeout: float = POST_TIMEOUT_S) -> bool:
    """POST one batch. True if the server accepted it."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": USER_AGENT,
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in HTTP_OK
    except urllib.error.HTTPError as e:
        print(f"[!] POST {url} HTTP {e.code}: {e.reason}")
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[!] POST {url} failed: {e}")
        return False


class Uploader(threading.Thread):
    """Background HTTP uploader.

    The main loop hands off one batch at a time via submit(), which never
    blocks. A batch that fails to POST is stashed for the loop to reclaim and
    retry.
    """

    def __init__(self, url: str, api_key: str):
        """Start idle; run() blocks on the queue until the loop submits."""
        super().__init__(daemon=True)
        self.url = url
        self.api_key = api_key
        self.q: queue.Queue = queue.Queue(maxsize=INFLIGHT_MAX)
        self._failed: list[dict] = []
        self._failed_snaps: list[dict] = []
        self._lock = threading.Lock()

    def submit(self, payload: dict) -> bool:
        """Queue one batch. False if another is already in flight."""
        try:
            self.q.put_nowait(payload)
            return True
        except queue.Full:
            return False

    def take_failed(self) -> list[dict]:
        """Hand back rows a prior upload failed to deliver, for a retry."""
        with self._lock:
            rows, self._failed = self._failed, []
        return rows

    def take_failed_snapshots(self) -> list[dict]:
        """Hand back undelivered snapshots, kept out of the row backlog.

        A lost snapshot is not self-healing: the robot emits one only when its
        shape changes, so it must be retried rather than merged and dropped.
        """
        with self._lock:
            snaps, self._failed_snaps = self._failed_snaps, []
        return snaps

    def run(self):
        """POST each submitted batch, stashing whatever the server refuses."""
        while True:
            payload = self.q.get()
            body = {"hostname": HOSTNAME, **payload}
            rows, snaps = payload.get("rows", []), payload.get("snapshots", [])
            what = f"{len(rows)} rows" if rows else f"{len(snaps)} snapshots"
            if _post_metrics(self.url, body, self.api_key):
                print(f"[+] pushed {what}")
            else:
                print(f"[!] push failed, holding {what}")
                with self._lock:
                    self._failed = rows + self._failed
                    self._failed_snaps = snaps + self._failed_snaps


# ============================================================================
# Daemon
# ============================================================================
def main():
    """Sample every tick, push every ticks_per_push, until killed."""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    streams = list(CFG.streams)
    sample_interval_s = float(CFG.sample_interval_s)
    push_interval_s = float(CFG.push_interval_s)
    max_pending = int(CFG.max_pending_rows)
    max_rows_per_push = int(CFG.max_rows_per_push)
    max_snapshots_per_push = int(CFG.max_snapshots_per_push)
    max_pending_snaps = int(CFG.max_pending_snapshots)
    telemetry_url = f"{CFG.bb_api_url.rstrip('/')}{TELEMETRY_PATH}"
    ticks_per_push = max(1, int(round(push_interval_s / sample_interval_s)))

    print(f"[+] telemetry daemon starting hostname={HOSTNAME}")
    print(f"[+] streams={streams}")
    print(f"[+] sample={sample_interval_s}s push={push_interval_s}s "
          f"url={telemetry_url}")

    trigger = [0]
    Loop.init(trigger)
    Loop.set_ms(int(sample_interval_s * MS_PER_S), trigger)

    readers = {name: Reader(name, keeptime=False) for name in streams}
    # sync=True walks the ring one publish at a time; the default jumps to the
    # newest slot and discards the burst in between.
    snap_readers = {name: Reader(name, keeptime=False, sync=True)
                    for name in CFG.snapshot_streams}
    last_hash: dict[str, str] = {}
    last_ts_ns: dict[str, int] = {}
    pending: list[dict] = []
    pending_snaps: list[dict] = []
    snap_turn = True
    ticks = 0

    uploader = Uploader(telemetry_url, CFG.bb_api_key)
    uploader.start()

    try:
        while True:
            Loop.keeptime()

            for name, reader in readers.items():
                if reader.ready():
                    pending.extend(
                        _sample_to_rows(name, reader.data, last_ts_ns))

            if len(pending) > max_pending:
                drop = len(pending) - max_pending
                pending = pending[drop:]
                print(f"[!] pending exceeded {max_pending}, dropped {drop}")

            for name, reader in snap_readers.items():
                # Drain, don't sample: one ready() advances a single slot.
                for _ in range(max_snapshots_per_push):
                    if not reader.ready():
                        break
                    doc = _snapshot_doc(name, reader.data, last_hash)
                    if doc:
                        pending_snaps.append(doc)

            ticks += 1
            if ticks >= ticks_per_push:
                ticks = 0
                reclaimed = uploader.take_failed()
                if reclaimed:
                    pending = reclaimed + pending
                # Re-bound after reclaim: a failing upload must not grow the
                # backlog without limit.
                if len(pending) > max_pending:
                    drop = len(pending) - max_pending
                    pending = pending[drop:]
                    print(f"[!] pending exceeded {max_pending}, "
                          f"dropped {drop} oldest rows")
                pending_snaps = (uploader.take_failed_snapshots()
                                 + pending_snaps)[-max_pending_snaps:]
                # One payload per tick, alternating, so a payload the server
                # keeps rejecting cannot starve the other kind.
                queued = False
                if pending_snaps and (snap_turn or not pending):
                    queued = uploader.submit(
                        {"snapshots": pending_snaps[:max_snapshots_per_push]})
                    if queued:
                        pending_snaps = pending_snaps[max_snapshots_per_push:]
                    snap_turn = False
                if not queued and pending:
                    chunk = pending[:max_rows_per_push]
                    if uploader.submit({"rows": chunk}):
                        pending = pending[len(chunk):]
                    snap_turn = True
    except KeyboardInterrupt:
        print("[+] shutting down")


if __name__ == "__main__":
    main()
