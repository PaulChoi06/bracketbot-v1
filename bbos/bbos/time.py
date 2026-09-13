import math, struct, time, ctypes, os, posix_ipc, mmap
from pathlib import Path
from ctypes import c_long
import sys
WHOAMI = f"{Path(sys.modules['__main__'].__file__).parent.name}__{Path(sys.modules['__main__'].__file__).name[:-3]}"


class timespec(ctypes.Structure):
    _fields_ = [("tv_sec", c_long),
                ("tv_nsec", c_long)]
librt = ctypes.CDLL("librt.so.1", use_errno=True)
CLOCK_MONOTONIC = 1

def ns_sleep(ns: int):
    ts = timespec(ns // 1_000_000_000, ns % 1_000_000_000)
    librt.clock_nanosleep(CLOCK_MONOTONIC, 0, ctypes.byref(ts), None)

class MovingAverage:
    def __init__(self, window_size: int):
        self._window_size: int = window_size
        self._buffer: list[float] = [0.0] * window_size
        self._index: int = 0
        self._count: int = 0
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._max: float = 0.0

    def add(self, new_value: float):
        old_value = self._buffer[self._index]
        self._sum -= old_value
        self._sum_sq -= old_value * old_value
        self._max = max(self._max, new_value)
        self._buffer[self._index] = new_value
        self._sum += new_value
        self._sum_sq += new_value * new_value
        self._index = (self._index + 1) % self._window_size
        self._count = min(self._count + 1, self._window_size)

    def last(self) -> float:
        return self._buffer[self._index]

    def is_reset(self) -> bool:
        return self._index == 0

    def avg(self) -> float:
        if self._count == 0:
            return float('nan')
        return self._sum / self._count

    def max(self) -> float:
        if self._count == 0:
            return float('nan')
        return self._max

    def std(self) -> float:
        if self._count == 0:
            return float('nan')
        if self._count == 1:
            return 0.0
        mean = self.avg()
        variance = (self._sum_sq / self._count) - (mean * mean)
        return math.sqrt(max(0.0, variance))


MAX_TIMELOGS = 128
TIMELOG_HEADER = 64
TIMELOG_SLOT = 128
TIMELOG_SHM_NAME = "bbos_timelogs"
TIMELOG_SIZE = TIMELOG_HEADER + MAX_TIMELOGS * TIMELOG_SLOT
# Slot layout (128 bytes):
#   [0]      active (u8)
#   [1:5]    pid (u32 LE)
#   [5:69]   name (64 bytes, null-padded)
#   [72:80]  avg_ns (i64 LE)
#   [80:88]  std_ns (i64 LE)
#   [88:96]  max_ns (i64 LE)


class TimeLog:
    _shm = None
    _mapfile = None

    @classmethod
    def _ensure_registry(cls):
        if cls._mapfile is not None:
            return
        try:
            cls._shm = posix_ipc.SharedMemory(TIMELOG_SHM_NAME, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL, mode=0o666, size=TIMELOG_SIZE)
            cls._mapfile = mmap.mmap(cls._shm.fd, TIMELOG_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            cls._mapfile.write(b'\x00' * TIMELOG_SIZE)
        except posix_ipc.ExistentialError:
            cls._shm = posix_ipc.SharedMemory(TIMELOG_SHM_NAME)
            cls._mapfile = mmap.mmap(cls._shm.fd, TIMELOG_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        cls._shm.close_fd()
        for i in range(MAX_TIMELOGS):
            o = TIMELOG_HEADER + i * TIMELOG_SLOT
            if cls._mapfile[o] == 1:
                pid = int.from_bytes(bytes(cls._mapfile[o+1:o+5]), 'little')
                try:
                    os.kill(pid, 0)
                except OSError:
                    cls._mapfile[o] = 0

    def __init__(self, name):
        self._ensure_registry()
        self._name = name
        self._buf = MovingAverage(10)
        self._last = -1
        self._slot = -1
        pid_bytes = os.getpid().to_bytes(4, 'little')
        name_bytes = name.encode()[:64].ljust(64, b'\x00')
        for i in range(MAX_TIMELOGS):
            o = TIMELOG_HEADER + i * TIMELOG_SLOT
            if self._mapfile[o] == 0:
                self._slot = i
                self._offset = o
                self._mapfile[o] = 1
                self._mapfile[o+1:o+5] = pid_bytes
                self._mapfile[o+5:o+69] = name_bytes
                struct.pack_into('<qqq', self._mapfile, o + 72, 0, 0, 0)
                break
        if self._slot < 0:
            print(f"[!] TimeLog: no free slots for {name}", flush=True)

    def close(self):
        if self._slot >= 0:
            self._mapfile[self._offset] = 0
            self._slot = -1

    def log(self):
        if self._last < 0:
            self._last = time.monotonic_ns()
        self._buf.add(float(time.monotonic_ns() - self._last))
        self._last = time.monotonic_ns()
        if self._buf.is_reset() and self._slot >= 0:
            struct.pack_into('<qqq', self._mapfile, self._offset + 72,
                             int(self._buf.avg()), int(self._buf.std()), int(self._buf.max()))


class Loop:
    _period = 100 # ms
    _last = -1
    _requested_ms = set()
    _triggers = {}
    _num_calls = 0
    _i = 0
    _manage_period = True
    _lagging = False
    @staticmethod
    def keeptime():
        if Loop._i >= Loop._num_calls - 1:
            Loop._i = 0 
            if Loop._last > 0:
                sleep_for = 1_000_000*Loop._period - (time.monotonic_ns() - Loop._last)
                if sleep_for >= 0:
                    if Loop._manage_period:
                        ns_sleep(sleep_for)
                    Loop._lagging = False
                else:
                    Loop._lagging = True
                    if os.environ.get("BBOS_VERBOSE"): print(f"[-] Loop lagging by {(sleep_for) * 1e-6:.2f}ms", flush=True)
            Loop._last = time.monotonic_ns()
            for trigger, reset in Loop._triggers.values():
                trigger[0] = (trigger[0] + 1) % reset if not Loop._lagging else 0
        else:
            Loop._i += 1
    
    @staticmethod
    def init(trigger):
        Loop._num_calls += 1
        Loop._triggers[hex(id(trigger))] = [trigger,1]
    
    @staticmethod
    def remove(trigger):
        Loop._num_calls -= 1
        Loop._triggers.pop(hex(id(trigger)))

    @staticmethod
    def manage_period(value):
        assert isinstance(value, bool)
        Loop._manage_period = value

    @staticmethod
    def set_ms(ms, trigger):
        assert ms > 0 and isinstance(ms, int)
        trigger_key = hex(id(trigger))
        is_new_trigger = trigger_key not in Loop._triggers
        if not ms in Loop._requested_ms:
            Loop._requested_ms.add(ms)
            print(Loop._requested_ms)
            new_period = math.gcd(*Loop._requested_ms)
            if len(Loop._requested_ms) != 1 and new_period != Loop._period:
                multiplier = Loop._period // new_period
                for t in Loop._triggers:
                    Loop._triggers[t][1] *= multiplier
                print(f"[+] Changed Loop._period from {Loop._period}ms to {new_period}ms")
            Loop._period = new_period
        Loop._triggers[trigger_key] = [trigger, int(ms / Loop._period)]
        if is_new_trigger:
            print(f"[+] Loop._period: {Loop._period}ms")
            print(Loop._triggers)
