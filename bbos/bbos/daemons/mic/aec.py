"""SpeexDSP echo canceller over ctypes, against the system libspeexdsp.so.1.

Feed the near-end mic and the ch2 hardware reference frame-for-frame. Stateful: the
adaptive filter converges over successive frames, so one instance per stream.
"""
import ctypes as C
import numpy as np

# request codes from speex_echo.h / speex_preprocess.h
_ECHO_SET_SAMPLING_RATE = 24
_PREPROCESS_SET_DENOISE = 0
_PREPROCESS_SET_AGC = 1
_PREPROCESS_SET_ECHO_STATE = 24
_PREPROCESS_SET_ECHO_SUPPRESS = 28          # residual echo suppression, dB (<=0)
_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE = 30   # residual suppression during double-talk, dB

# The nix python's loader does not search /lib/aarch64-linux-gnu, so try the absolute path first.
_LIB_CANDIDATES = ("/lib/aarch64-linux-gnu/libspeexdsp.so.1", "libspeexdsp.so.1")


def _load_speex(lib_path=None):
    candidates = [lib_path] if lib_path else list(_LIB_CANDIDATES)
    errors = []
    for cand in candidates:
        try:
            return C.CDLL(cand)
        except OSError as e:
            errors.append(f"{cand}: {e}")
    raise OSError("could not load libspeexdsp; tried " + "; ".join(errors))


class SpeexAEC:
    def __init__(self, frame_size, filter_length, sample_rate,
                 denoise=False, agc=False, echo_suppress=None,
                 echo_suppress_active=None, lib_path=None):
        self.frame_size = int(frame_size)
        self.filter_length = int(filter_length)
        self.sample_rate = int(sample_rate)
        self._lib = _load_speex(lib_path)
        self._bind()

        self._st = self._lib.speex_echo_state_init(self.frame_size, self.filter_length)
        rate = C.c_int(self.sample_rate)
        self._lib.speex_echo_ctl(self._st, _ECHO_SET_SAMPLING_RATE, C.byref(rate))

        self._pp = self._lib.speex_preprocess_state_init(self.frame_size, self.sample_rate)
        self._lib.speex_preprocess_ctl(
            self._pp, _PREPROCESS_SET_ECHO_STATE, C.cast(C.c_void_p(self._st), C.c_void_p))
        self._set_flag(_PREPROCESS_SET_DENOISE, denoise)
        self._set_flag(_PREPROCESS_SET_AGC, agc)
        # Nonlinear residual suppression, the main quality lever; *_active applies during double-talk.
        if echo_suppress is not None:
            self._set_int(_PREPROCESS_SET_ECHO_SUPPRESS, int(echo_suppress))
        if echo_suppress_active is not None:
            self._set_int(_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE, int(echo_suppress_active))

    def _bind(self):
        l = self._lib
        i16p = C.POINTER(C.c_int16)
        l.speex_echo_state_init.restype = C.c_void_p
        l.speex_echo_state_init.argtypes = [C.c_int, C.c_int]
        l.speex_echo_cancellation.argtypes = [C.c_void_p, i16p, i16p, i16p]
        l.speex_echo_ctl.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
        l.speex_echo_state_destroy.argtypes = [C.c_void_p]
        l.speex_preprocess_state_init.restype = C.c_void_p
        l.speex_preprocess_state_init.argtypes = [C.c_int, C.c_int]
        l.speex_preprocess_ctl.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
        l.speex_preprocess_run.argtypes = [C.c_void_p, i16p]
        l.speex_preprocess_state_destroy.argtypes = [C.c_void_p]

    def _set_flag(self, request, enabled):
        v = C.c_int(1 if enabled else 0)
        self._lib.speex_preprocess_ctl(self._pp, request, C.byref(v))

    def _set_int(self, request, value):
        v = C.c_int(int(value))
        self._lib.speex_preprocess_ctl(self._pp, request, C.byref(v))

    def process_frame(self, near, far):
        """Cancel echo from one frame: near/far are int16 arrays of frame_size."""
        near = np.ascontiguousarray(near, dtype=np.int16)
        far = np.ascontiguousarray(far, dtype=np.int16)
        out = np.zeros(self.frame_size, dtype=np.int16)
        i16p = C.POINTER(C.c_int16)
        self._lib.speex_echo_cancellation(
            self._st, near.ctypes.data_as(i16p), far.ctypes.data_as(i16p),
            out.ctypes.data_as(i16p))
        self._lib.speex_preprocess_run(self._pp, out.ctypes.data_as(i16p))
        return out

    def process_chunk(self, near_chunk, far_chunk):
        """Process a multi-frame chunk (length must be a multiple of frame_size)."""
        if len(near_chunk) != len(far_chunk):
            raise ValueError("near and far chunks must be the same length")
        if len(near_chunk) % self.frame_size:
            raise ValueError(
                f"chunk length {len(near_chunk)} not a multiple of frame_size {self.frame_size}")
        out = np.empty(len(near_chunk), dtype=np.int16)
        for i in range(0, len(near_chunk), self.frame_size):
            sl = slice(i, i + self.frame_size)
            out[sl] = self.process_frame(near_chunk[sl], far_chunk[sl])
        return out

    def close(self):
        if getattr(self, "_pp", None):
            self._lib.speex_preprocess_state_destroy(self._pp)
            self._pp = None
        if getattr(self, "_st", None):
            self._lib.speex_echo_state_destroy(self._st)
            self._st = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
