"""Mode announcements over the speaker.

Phrases are pre-rendered in wavs/ at the speaker topic's format (16 kHz mono s16). The writer is
handed in, not opened: bbos allows one per topic and the daemon already holds it for the browser
relay. Safe to share, since the relay stops once a Quest connects and only a Quest changes mode.

Regenerate on a Mac:
    say -v Samantha -r 170 -o wavs/<name>.wav --data-format=LEI16@16000 "<text>"
"""
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np
from bbos import Config

CFG = Config("speaker")
WAV_DIR = Path(__file__).parent / "wavs"

# The speaker daemon forwards one chunk per loop into pacat's 100ms buffer, so writing at bare
# realtime starves it: write ahead, lead in with silence, and open the writer with buf_ms >= 400.
PACE = 0.90                     # Fraction of realtime to write at.
LEAD_IN_S = 0.4
EXIT_JOIN_S = 5.0               # Cap the teardown wait for the queue to drain.


class Sound:
    """say() queues and returns; a worker plays one at a time, and writer=None makes it a no-op."""

    def __init__(self, writer):
        self._w = writer
        self._q = queue.Queue()
        self._thread = None
        if writer is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def close(self):
        """Stop the worker. Safe to call more than once."""
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=EXIT_JOIN_S)
            self._thread = None

    def say(self, text):
        if self._thread is not None:
            self._q.put(text)

    def _load(self, text):
        """wavs/<text>.wav, lead-in prepended and padded out to whole chunks."""
        path = WAV_DIR / ("%s.wav" % text.lower().replace(" ", "_"))
        with wave.open(str(path)) as f:
            fmt = (f.getframerate(), f.getnchannels(), f.getsampwidth())
            if fmt != (CFG.sample_rate, CFG.channels, 2):
                raise ValueError("%s is %s, need (%d, %d, 2)"
                                 % (path.name, fmt, CFG.sample_rate,
                                    CFG.channels))
            audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        chunk = CFG.chunk_size
        lead = np.zeros(int(LEAD_IN_S * CFG.sample_rate), dtype=np.int16)
        tail = np.zeros(-(len(lead) + len(audio)) % chunk, dtype=np.int16)
        return np.concatenate([lead, audio, tail])

    def _run(self):
        chunk = CFG.chunk_size
        period = PACE * chunk / CFG.sample_rate
        for text in iter(self._q.get, None):
            try:
                audio = self._load(text)
                due = time.monotonic()
                for i in range(0, len(audio), chunk):
                    with self._w.buf() as b:
                        b["audio"] = audio[i:i + chunk].reshape(
                            -1, CFG.channels)
                    due += period
                    time.sleep(max(0.0, due - time.monotonic()))
            except Exception as e:
                print("  [sound] %r failed: %s" % (text, e), flush=True)
