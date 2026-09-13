"""Detect the wake phrase and publish ``wakeword.state``.

Runs the three ONNX models (melspectrogram -> speech embedding -> classifier)
directly with onnxruntime. This replicates openwakeword's streaming pipeline
(verified score-identical) without importing openwakeword, whose package
__init__ drags in scipy/scikit-learn (+86MB RSS) that inference never uses.

All complete 80 ms frames in each 100 ms mic chunk are processed with a single
melspectrogram call, so per-frame overhead is paid once per chunk.
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from bbos import Reader, Type, Writer

FRAME_SAMPLES = 1280   # 80 ms at 16 kHz
MEL_CONTEXT = 480      # extra trailing samples fed to melspec for filter context
MEL_WINDOW = 76        # mel frames per embedding window (~775 ms)
MEL_STEP = 8           # mel frames between embedding windows (80 ms)
MEL_MAX = 970          # ~10 s of mel history
FEAT_MAX = 120         # ~10 s of embedding history
THRESHOLD = 0.72
REFRACTORY_S = 3.0
PUBLISH_S = 1.0
POLL_S = 0.05
WARMUP_FRAMES = 8


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    cpu = ["CPUExecutionProvider"]
    models = Path(__file__).parent / "models"
    melspec = ort.InferenceSession(str(models / "melspectrogram.onnx"), opts, providers=cpu)
    embedding = ort.InferenceSession(str(models / "embedding_model.onnx"), opts, providers=cpu)
    classifier = ort.InferenceSession(str(models / "hey_bracket_bot_progressive_ensemble_v1.onnx"), opts, providers=cpu)
    cls_input = classifier.get_inputs()[0].name
    n_feature_frames = classifier.get_inputs()[0].shape[1]

    def mel_of(samples_i16):
        x = samples_i16.astype(np.float32)[None, :]
        return melspec.run(None, {"input": x})[0].squeeze() / 10 + 2

    def embed(mel_window):
        x = mel_window.astype(np.float32)[None, :, :, None]
        return embedding.run(None, {"input_1": x})[0].squeeze()

    # Prime the feature buffer from noise, exactly like openwakeword does, so
    # the classifier's 16-frame input window is full from the first real frame.
    seed_spec = mel_of(np.random.randint(-1000, 1000, 16000 * 4).astype(np.int16))
    seed_windows = [seed_spec[i:i + MEL_WINDOW] for i in range(0, seed_spec.shape[0], MEL_STEP)
                    if seed_spec[i:i + MEL_WINDOW].shape[0] == MEL_WINDOW]
    seed_batch = np.expand_dims(np.array(seed_windows), axis=-1).astype(np.float32)
    feat_buf = embedding.run(None, {"input_1": seed_batch})[0].squeeze()

    mel_buf = np.ones((MEL_WINDOW, 32), dtype=np.float32)
    tail = np.zeros(MEL_CONTEXT, dtype=np.int16)
    pending = np.zeros(0, dtype=np.int16)

    def process(new_audio):
        """Consume complete frames from new_audio; return scores of new frames."""
        nonlocal mel_buf, feat_buf, tail, pending
        pending = np.concatenate((pending, new_audio))
        n = (pending.size // FRAME_SAMPLES) * FRAME_SAMPLES
        if n == 0:
            return []
        seg = np.concatenate((tail, pending[:n]))
        tail = seg[-MEL_CONTEXT:]
        pending = pending[n:]
        mel_buf = np.vstack((mel_buf, mel_of(seg)))[-MEL_MAX:]
        n_frames = n // FRAME_SAMPLES
        for i in range(n_frames - 1, -1, -1):
            ndx = -MEL_STEP * i if i != 0 else len(mel_buf)
            window = mel_buf[-MEL_WINDOW + ndx:ndx]
            if window.shape[0] == MEL_WINDOW:
                feat_buf = np.vstack((feat_buf, embed(window)))
        feat_buf = feat_buf[-FEAT_MAX:]
        scores = []
        for i in range(n_frames - 1, -1, -1):
            s = -n_feature_frames - i
            e = s + n_feature_frames if s + n_feature_frames != 0 else len(feat_buf)
            feats = feat_buf[s:e][None, :].astype(np.float32)
            scores.append(float(classifier.run(None, {cls_input: feats})[0][0][0]))
        return scores

    silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    for _ in range(WARMUP_FRAMES):
        process(silence)

    print(f"[+] wakeword model=hey_bracket_bot_progressive_ensemble_v1 threshold={THRESHOLD:.3f}")

    active = False
    last_detection = float("-inf")
    next_publish = time.monotonic() + PUBLISH_S

    with (
        Reader("mic.audio", keeptime=False, sync=True) as microphone,
        Writer("wakeword.state", Type("wakeword_state"), keeptime=False) as state,
    ):
        while True:
            if microphone.ready():
                samples = microphone.data["audio"].reshape(-1).copy()
                for score in process(samples):
                    now = time.monotonic()
                    if score >= THRESHOLD and now - last_detection >= REFRACTORY_S:
                        active = True
                        last_detection = now
                        next_publish = now  # publish the detection immediately
                        print(f"[+] detected score={score:.3f}")
            else:
                time.sleep(POLL_S)

            now = time.monotonic()
            if now >= next_publish:
                with state.buf() as data:
                    data["active"] = active
                active = False
                next_publish = now + PUBLISH_S


if __name__ == "__main__":
    main()
