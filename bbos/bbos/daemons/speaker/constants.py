from bbos import register, realtime, state
import numpy as np


@register
class speaker:
    sample_rate: int = 16_000
    channels: int = 1
    chunk_ms: int = 100
    chunk_size: int = sample_rate // 1000 * chunk_ms
    volume: float = 0.6
    pulse_volume: str = "100%"
    comp_enabled: bool = True


@realtime(ms=speaker.chunk_ms)
def speaker_audio():
    """Mono PCM to play on the head board, at speaker.sample_rate."""
    return [
        ("audio", np.int16, (speaker.chunk_size, speaker.channels)),
    ]


@state
def speaker_volume():
    """Live playback gain, 0..1, and mute."""
    return [
        ("gain", np.float32),
        ("mute", np.bool_),
    ]
