from typing import List
from bbos.registry import Type, run_writer_open_hook
from bbos.time import TimeLog, Loop

import os, json, inspect, contextlib, sys, traceback, ctypes, time
import numpy as np
from pathlib import Path

META_SIZE = 4096

# ── load compiled backend ──

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'ipc_core.so'))

_lib.ipc_writer_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int)]
_lib.ipc_writer_create.restype = ctypes.c_void_p

_lib.ipc_writer_unlink.argtypes = [ctypes.c_char_p]
_lib.ipc_writer_unlink.restype = None

_lib.ipc_munmap.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.ipc_munmap.restype = None

_lib.ipc_seq_inc.argtypes = [ctypes.c_void_p]
_lib.ipc_seq_inc.restype = None

_lib.ipc_seq_read.argtypes = [ctypes.c_void_p]
_lib.ipc_seq_read.restype = ctypes.c_uint32

_lib.ipc_writer_idx.argtypes = [ctypes.c_void_p]
_lib.ipc_writer_idx.restype = ctypes.c_int

_lib.ipc_reader_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int), ctypes.c_char_p,
                                  ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.ipc_reader_open.restype = ctypes.c_void_p

_lib.ipc_check_pid.argtypes = [ctypes.c_int]
_lib.ipc_check_pid.restype = ctypes.c_int

_lib.ipc_inode.argtypes = [ctypes.c_char_p]
_lib.ipc_inode.restype = ctypes.c_ulong

class _ReadResult(ctypes.Structure):
    _fields_ = [('ridx', ctypes.c_int64),
                ('last_seq', ctypes.c_int64),
                ('dropped', ctypes.c_int32)]

_lib.ipc_reader_read.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int64, ctypes.c_int64,
                                  ctypes.c_void_p, ctypes.POINTER(_ReadResult)]
_lib.ipc_reader_read.restype = ctypes.c_int

# ── utilities ──

def pretty_bytes(n: int) -> str:
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def _caller_signature():
    f = inspect.stack()[2]
    return f"{os.path.abspath(f.filename)}:{f.lineno}"

def _encode_header(sig, dtype, period):
    owner = Path(sys.modules['__main__'].__file__)
    owner = owner.parent.name + '/' + owner.name
    return json.dumps({"caller": sig, "dtype": dtype.descr, "period": period, "owner": owner}).encode()

def json_descr_to_dtype(desc):
    fixed = []
    for field in desc:
        if len(field) == 3:
            name, dt, shape = field
            if isinstance(shape, list):
                shape = tuple(shape)
            elif isinstance(shape, int):
                shape = (shape,)
            fixed.append((name, dt, shape))
        else:
            fixed.append(tuple(field))
    return np.dtype(fixed)


class Writer:
    def __init__(self, name, datatype: Type | List[tuple], keeptime=True, buf_ms=0):
        self._typename = datatype._name if isinstance(datatype, Type) else None
        shmtype, period = datatype if isinstance(datatype, tuple) else datatype()
        self.N = max(int(np.ceil(buf_ms / period)), 1) if period is not None else 1
        shmdtype = np.dtype(shmtype)
        self._size = shmdtype.itemsize * self.N + META_SIZE
        if period is not None:
            print(f"Writer {name} using {self.N} buffers, size {pretty_bytes(self._size)}, window {period*self.N}ms", flush=True)
        sig = _caller_signature()
        meta = _encode_header(sig, shmdtype, period)
        assert len(meta) < META_SIZE - 16, "Metadata too large!"
        self._name = name
        self._keeptime = keeptime if period is not None else False

        err = ctypes.c_int()
        self._map = _lib.ipc_writer_create(
            name.encode(), meta, len(meta),
            self._size, self.N, os.getpid(),
            ctypes.byref(err))
        if self._map is None:
            if err.value > 0:
                raise RuntimeError(f"Writer for {name} already exists (pid={err.value})")
            raise RuntimeError(f"Failed to create shared memory for {name}")

        data_buf = (ctypes.c_char * (shmdtype.itemsize * self.N)).from_address(self._map + META_SIZE)
        self._buf = np.ndarray(self.N, dtype=shmdtype, buffer=data_buf)

        if self._keeptime:
            self._trigger = [0]
            Loop.init(self._trigger)
            Loop.set_ms(period, self._trigger)

        if self._typename is not None:
            run_writer_open_hook(self._typename)

    def __enter__(self):
        return self

    def _update(self):
        if self._keeptime:
            return self._trigger[0] == 0
        return True

    @property
    def idx(self):
        return _lib.ipc_writer_idx(self._map)

    def ready(self):
        return self._update()

    @contextlib.contextmanager
    def buf(self):
        should_update = self._update()
        if should_update:
            _lib.ipc_seq_inc(self._map)
        try:
            if should_update:
                self._buf[self.idx]['timestamp'] = np.datetime64(time.time_ns(), 'ns')
            yield self._buf[self.idx] if should_update else np.zeros_like(self._buf[self.idx])
        finally:
            if should_update:
                _lib.ipc_seq_inc(self._map)
        if self._keeptime:
            Loop.keeptime()

    def __setitem__(self, idx, data):
        if self._update():
            _lib.ipc_seq_inc(self._map)
            self._buf[self.idx]['timestamp'] = np.datetime64(time.time_ns(), 'ns')
            self._buf[self.idx][idx] = data
            _lib.ipc_seq_inc(self._map)
        if self._keeptime:
            Loop.keeptime()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Writer {self._name} exited with exception", flush=True)
            traceback.print_exception(exc_type, exc_val, exc_tb)
        try:
            if self._keeptime:
                Loop.remove(self._trigger)
            _lib.ipc_writer_unlink(self._name.encode())
            _lib.ipc_munmap(self._map, self._size)
        except:
            pass
        return True


class Reader:
    def __init__(self, name, keeptime=True, sync=False, aligned_to=None, decimate=1):
        self._name = name
        self._readable = False
        self._sync = sync
        self._decimate = decimate
        self._aligned_to = aligned_to
        self._tlog = TimeLog(name)
        self._data = None
        self._keeptime = keeptime
        self._writer_pid = 0
        self._map = None
        self._map_size = 0
        self._inode = 0
        if keeptime:
            self._trigger = [0]
            Loop.init(self._trigger)

    def __enter__(self):
        return self

    def _update(self):
        if self._keeptime:
            return self._trigger[0] == 0
        return True

    def ready(self):
        if self._readable:
            # Reconnect if the writer PID died, OR the segment was closed+recreated (new inode) by a
            # still-live PID — ipc_check_pid alone can't see the latter, leaving the reader stuck on
            # the deleted segment (e.g. a long-lived daemon that released + reacquired drive.ctrl).
            if not _lib.ipc_check_pid(self._writer_pid) or _lib.ipc_inode(self._name.encode()) != self._inode:
                self._readable = False
                _lib.ipc_munmap(self._map, self._map_size)
                self._map = None
        if not self._readable:
            out_size = ctypes.c_int()
            out_pid = ctypes.c_int()
            out_meta = ctypes.create_string_buffer(META_SIZE)
            out_meta_len = ctypes.c_int()
            err = ctypes.c_int()
            map_ptr = _lib.ipc_reader_open(
                self._name.encode(),
                ctypes.byref(out_size), ctypes.byref(out_pid),
                out_meta, ctypes.byref(out_meta_len), ctypes.byref(err))
            if map_ptr is None:
                if self._keeptime:
                    Loop.keeptime()
                return False
            try:
                self._writer_pid = out_pid.value
                if self._writer_pid > 0 and not _lib.ipc_check_pid(self._writer_pid):
                    _lib.ipc_munmap(map_ptr, out_size.value)
                    if self._keeptime:
                        Loop.keeptime()
                    return False
                lock = json.loads(out_meta.raw[:out_meta_len.value])
                if lock["period"] is None:
                    if self._keeptime:
                        Loop.remove(self._trigger)
                    self._keeptime = False
                shmdtype = np.dtype(json_descr_to_dtype(lock["dtype"]))
                if self._keeptime:
                    self._trigger[0] = 0
                    Loop.set_ms(lock["period"], self._trigger)
                self._readable = True
                self._map = map_ptr
                self._map_size = out_size.value
                self._inode = _lib.ipc_inode(self._name.encode())
                self._dtype = shmdtype
                self._item_size = shmdtype.itemsize
                # byte offset of the 'timestamp' field — for aligned_to reads (it's appended last, not at offset 0)
                self._ts_offset = shmdtype.fields['timestamp'][1] if 'timestamp' in shmdtype.fields else 0
                self._read_buf = np.zeros(1, dtype=shmdtype)
                self._data = np.zeros(1, dtype=shmdtype)[0]
                self._last_seq = -1
            except Exception:
                _lib.ipc_munmap(map_ptr, out_size.value)
                self._readable = False
                if self._keeptime:
                    Loop.keeptime()
                return False
        data = self._read()
        stale = data['timestamp'] == self._data['timestamp']
        self._data = data
        if not stale:
            self._tlog.log()
        if self._keeptime:
            Loop.keeptime()
        return not stale

    def _read(self):
        align_ts = -1
        if self._aligned_to is not None and self._aligned_to.data is not None:
            align_ts = int(self._aligned_to.data['timestamp'].view('i8'))

        result = _ReadResult()
        rc = _lib.ipc_reader_read(
            self._map, self._item_size, self._ts_offset,
            int(self._sync), self._decimate,
            self._last_seq, align_ts,
            self._read_buf.ctypes.data,
            ctypes.byref(result))
        if rc != 0:
            # Writer died mid-publish (seq stuck odd) — the slot is torn, nothing to read.
            # Return the previous data unchanged so ready() reports stale; its pid check
            # disconnects and starts reconnect-polling on the next call.
            return self._data

        self._last_seq = result.last_seq
        if result.dropped > 1 and self._keeptime and os.environ.get("BBOS_VERBOSE"):
            print(f"Reader {self._name} dropped {result.dropped} frames", flush=True)
        return self._read_buf[0].copy()

    @property
    def data(self):
        return self._data

    @property
    def readable(self):
        return self._readable

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
        if self._keeptime:
            Loop.remove(self._trigger)
        if self._readable:
            _lib.ipc_munmap(self._map, self._map_size)
        self._tlog.close()
        return True
