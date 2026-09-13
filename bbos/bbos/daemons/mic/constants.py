from bbos import register, realtime

import numpy as np


@register
class mic:
    sample_rate: int = 16_000
    channels: int = 1
    chunk_ms: int = 100
    chunk_size: int = sample_rate // 1000 * chunk_ms
    pipeline: str = "raw_aec"             # "raw_aec" = Speex AEC on the ch2 reference, "raw" = gain only
    raw_gain_db: float = 24.0
    aec_denoise: bool = False             # True reaches ~34 dB ERLE but destroys near speech during double-talk
    aec_echo_suppress: int = -55          # residual suppression, more negative is more aggressive
    aec_echo_suppress_active: int = -35   # during double-talk, gentler so barge-in survives


@realtime(ms=mic.chunk_ms)
def mic_audio():
    """Mono PCM from the near-end mic, at mic.sample_rate."""
    return [
        ("audio", np.int16, (mic.chunk_size, mic.channels)),
    ]


@realtime(ms=mic.chunk_ms)
def mic_ref_level():
    """Level of the hardware echo reference: robot echo can only exist while this is hot. -240 = silent."""
    return [
        ("dbfs", np.float32),
    ]
