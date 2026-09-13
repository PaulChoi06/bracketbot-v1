"""The daemon's event log: one line per record, to stdout and to a ram ring.

RamLog is the ring, a fixed-size mmap in tmpfs. open_log() points log() at
one; until it runs, log() only prints. Nothing here is USB-specific.
"""

import mmap
import os
import threading
from datetime import datetime

NUL = b"\x00"                   # Unwritten tail; the first is where we resume.
PORT_W = 9                      # Log column widths.
PROBE_W = 12


class RamLog:
    """Fixed-size line log in tmpfs.

    Appends into an mmap; when full, drops the oldest evict_frac and memmoves
    the rest down. Cuts on line boundaries and zero-fills the tail, so the
    first NUL is where a restart resumes.
    """

    def __init__(self, path, size, evict_frac):
        """Map path as a size-byte ring, resuming at the first NUL in it."""
        self._size = int(size)
        self._keep = max(int(self._size * (1.0 - evict_frac)), 1)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size != self._size:
                os.ftruncate(fd, 0)  # A resized cap starts clean.
                os.ftruncate(fd, self._size)
            self._mm = mmap.mmap(fd, self._size)
        finally:
            os.close(fd)
        nul = self._mm.find(NUL)
        self._off = self._size if nul < 0 else nul

    def append(self, line):
        """Add one line, compacting first if it would not fit."""
        b = line.encode("utf-8", "replace")[:self._size - 1] + b"\n"
        if self._off + len(b) > self._size:
            self._evict(len(b))
        self._mm[self._off:self._off + len(b)] = b
        self._off += len(b)

    def _evict(self, need):
        # To the keep target, not just enough for `need`: once per compaction.
        drop = max(self._off + need - self._size, self._off - self._keep)
        nl = self._mm.find(b"\n", drop, self._off)
        drop = self._off if nl < 0 else nl + 1  # Never keep a partial line.
        tail = self._mm[drop:self._off]
        self._mm[0:len(tail)] = tail
        self._mm[len(tail):self._off] = NUL * (self._off - len(tail))
        self._off = len(tail)


_ring = None
# log() is module-level; mmap append is not atomic.
_lock = threading.Lock()


def open_log(path, size, evict_frac):
    """Point log() at a ram ring. Before this, log() only prints."""
    global _ring
    _ring = RamLog(path, size, evict_frac)


def log(port, probe, value, when=None):
    """2026-08-09T16:51:46.702-07:00  1-2.3.2    remove        seq=278901 ...

    when overrides the timestamp, for records that carry the kernel's own.
    """
    when = when or datetime.now().astimezone()
    ts = when.isoformat(timespec="milliseconds")
    line = f"{ts}  {port:<{PORT_W}}  {probe:<{PROBE_W}}  {value}"
    with _lock:
        print(line)
        if _ring is not None:
            _ring.append(line)
