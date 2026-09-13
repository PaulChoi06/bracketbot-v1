# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "soxr",
#   "bbos",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///

"""Capture: board 5ch@48k -> mic ch + echo-ref ch -> 16k -> Speex AEC -> mic.audio.

Capture ch2 is an electrical loopback of the amplifier feed, so the AEC far-end reference
arrives in our own stream; mic.ref_level publishes its level for echo-gating consumers.
"""

from bbos import Writer, Type, Config
import numpy as np
import soxr

import fcntl
import os
import shutil
import struct
import subprocess
import sys
import termios
import threading
import time

from aec import SpeexAEC

CFG = Config("mic")
CHUNK = CFG.chunk_size
BBOS_RATE = CFG.sample_rate
CAP_RATE = 48_000
CAP_CH = 5
MIC_CH = 0                      # near-end acoustic mic column
REF_CH = 2                      # hardware echo reference (electrical loopback)
READ_FRAMES = CAP_RATE // 1000 * CFG.chunk_ms
READ_BYTES = READ_FRAMES * CAP_CH * 2   # int16 interleaved
ZERO_RECOVER_READS = max(1, 4000 // CFG.chunk_ms)      # respawn capture after ~4s of pure silence

ACOUSTIC_COLS = (0, 1, 3, 4)    # real mic columns, watched by the dead-capture guard
AEC_FRAME = 160                 # 10 ms at 16 kHz
AEC_FILTER_LEN = 4800           # 300 ms adaptive filter tail
STATUS_PERIOD_S = 10
SUSPEND_REASSERT_S = 30
STATUS_LOOPS = max(1, STATUS_PERIOD_S * 1000 // CFG.chunk_ms)
TOKENS = ("speakerphone", "bracketbot")   # substrings of the head board's USB product string

# PA can lose the source while the USB card is still in ALSA, and never re-detect on its own.
DEVICE_RECOVER_TRIES = 5      # discovery attempts before exiting for a manager respawn
DEVICE_RECOVER_WAIT_S = 2.0   # settle time after each recovery nudge

CATCHUP_MAX_CHUNKS = 50       # max extra chunks per iteration (5 s of backlog)
CATCHUP_LOG_CHUNKS = 5        # log when a catch-up this size happens


def _matches(text):
    t = text.lower()
    return any(tok in t for tok in TOKENS)


def _candidate_pulse_servers():
    seen = set()
    cur = os.environ.get("PULSE_SERVER")
    if cur:
        seen.add(cur)
        yield cur
    cands = []
    for base in ("/tmp", "/run/user"):
        for root, _, files in os.walk(base):
            if "native" in files and "pulse" in root.lower():
                p = os.path.join(root, "native")
                try:
                    cands.append((os.path.getmtime(p), f"unix:{p}"))
                except OSError:
                    pass
    for _, server in sorted(cands, reverse=True):
        if server not in seen:
            seen.add(server)
            yield server


def _server_has_device(env):
    """True only if this server exposes the capture source, not just the sink."""
    src = _find_source(env)
    return bool(src and "multichannel" in src.lower())


def _pulse_env():
    """Pin PULSE_SERVER to the server hosting the board; several run on this box."""
    for server in _candidate_pulse_servers():
        env = {**os.environ, "PULSE_SERVER": server}
        if _server_has_device(env):
            os.environ["PULSE_SERVER"] = server
            print(f"[pa] server: {server}", flush=True)
            return env
    default_env = os.environ.copy()
    if _server_has_device(default_env):
        print("[pa] server: inherited default (device reachable, PULSE_SERVER unpinned)",
              flush=True)
        return default_env
    print("[pa] WARNING: audio device not found on any PulseAudio server", flush=True)
    return default_env


def _suspend_module_loaded(env):
    try:
        r = subprocess.run(["pactl", "list", "modules", "short"],
                           capture_output=True, text=True, timeout=3, env=env)
    except Exception:
        return False
    return any("module-suspend-on-idle" in ln for ln in r.stdout.splitlines())


def _all_pulse_envs(base_env):
    """base_env plus an env for every other reachable PA server, since any of them may own the card."""
    envs = [base_env]
    seen = {base_env.get("PULSE_SERVER")}
    if None not in seen:  # the bare inherited default may be its own server
        envs.append({k: v for k, v in base_env.items() if k != "PULSE_SERVER"})
        seen.add(None)
    for server in _candidate_pulse_servers():
        if server and server not in seen:
            seen.add(server)
            envs.append({**base_env, "PULSE_SERVER": server})
    return envs


def _ensure_suspend_unloaded(base_env, reason, sweep=False):
    """Unload module-suspend-on-idle: a suspended USB card streams pure silence."""
    acted = False
    for env in (_all_pulse_envs(base_env) if sweep else [base_env]):
        try:
            if _suspend_module_loaded(env):
                subprocess.run(["pactl", "unload-module", "module-suspend-on-idle"],
                               capture_output=True, text=True, timeout=3, env=env)
                print(f"[pa] unloaded module-suspend-on-idle ({reason}) on "
                      f"{env.get('PULSE_SERVER') or 'default'}", flush=True)
                acted = True
        except Exception as e:
            print(f"[pa] suspend re-check failed: {e!r}", flush=True)
    return acted


def _find_source(env):
    """The board's 5-channel multichannel capture source."""
    r = subprocess.run(["pactl", "list", "sources", "short"],
                       capture_output=True, text=True, timeout=3, env=env)
    fallback = None
    for line in r.stdout.splitlines():
        if _matches(line) and ".monitor" not in line:
            name = line.split("\t")[1]
            if "multichannel" in name.lower():
                return name
            fallback = fallback or name
    return fallback


def _cleanup_stale():
    """Kill an orphaned parecord/pacat from a previous instance still holding the board."""
    me = os.getpid()
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        try:
            cmd = open(f"/proc/{name}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        stale = ("parecord" in cmd or "pacat" in cmd) and _matches(cmd)
        if stale:
            try:
                os.kill(int(name), 15)
                print(f"[cleanup] killed stale child {name}", flush=True)
            except ProcessLookupError:
                pass


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _avail_bytes(stream) -> int:
    """Bytes currently readable on a pipe without blocking (FIONREAD)."""
    try:
        return struct.unpack("i", fcntl.ioctl(stream.fileno(), termios.FIONREAD,
                                              b"\x00\x00\x00\x00"))[0]
    except Exception:
        return 0


def _spawn_rec(env, source):
    """Start (or restart) the 5-channel capture recorder."""
    parecord = shutil.which("parecord") or "parecord"
    return subprocess.Popen(
        [parecord, f"--device={source}", f"--channels={CAP_CH}",
         f"--rate={CAP_RATE}", "--format=s16le", "--raw",
         f"--latency-msec={CFG.chunk_ms}"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, bufsize=0)


def _recover_pa_device(env, attempt):
    """Re-expose the board after a USB re-enumeration: wait, reload module-udev-detect, restart PA."""
    if attempt == 0:
        return
    if attempt == 1:
        print("[recover] reloading module-udev-detect", flush=True)
        for e in _all_pulse_envs(env):
            subprocess.run(["pactl", "unload-module", "module-udev-detect"],
                           capture_output=True, timeout=5, env=e)
            subprocess.run(["pactl", "load-module", "module-udev-detect"],
                           capture_output=True, timeout=5, env=e)
        return
    pulseaudio = shutil.which("pulseaudio")
    if not pulseaudio:
        return
    print("[recover] restarting PulseAudio", flush=True)
    subprocess.run([pulseaudio, "-k"], capture_output=True, timeout=5)
    subprocess.run([pulseaudio, "--start"], capture_output=True, timeout=5)


def main():
    _cleanup_stale()
    env = _pulse_env()
    _ensure_suspend_unloaded(env, "startup", sweep=True)
    source = _find_source(env)
    attempt = 0
    while not source and attempt < DEVICE_RECOVER_TRIES:
        print(f"[!] capture source not found -- recovery "
              f"{attempt + 1}/{DEVICE_RECOVER_TRIES}", flush=True)
        _recover_pa_device(env, attempt)
        time.sleep(DEVICE_RECOVER_WAIT_S)
        env = _pulse_env()
        _ensure_suspend_unloaded(env, "recover", sweep=True)
        source = _find_source(env)
        attempt += 1
    if not source:
        print(f"[!] capture source not found after {DEVICE_RECOVER_TRIES} recovery "
              "attempts; exiting for respawn", flush=True)
        sys.exit(1)
    print(f"[pa] source: {source}", flush=True)

    rec = _spawn_rec(env, source)
    rs_mic = soxr.ResampleStream(CAP_RATE, BBOS_RATE, 1, dtype=np.float32)
    rs_ref = soxr.ResampleStream(CAP_RATE, BBOS_RATE, 1, dtype=np.float32)
    pipeline = str(CFG.pipeline).strip().lower()
    raw_mode = pipeline == "raw"
    raw_aec = pipeline == "raw_aec"
    raw_gain = float(10.0 ** (float(CFG.raw_gain_db) / 20.0))
    aec = None
    print(f"[mic] pipeline={pipeline.upper()} ch{MIC_CH} "
          f"{'AEC then ' if raw_aec else ''}+{float(CFG.raw_gain_db):.0f}dB "
          "(ref_level still published)", flush=True)
    if not raw_mode:
        aec = SpeexAEC(AEC_FRAME, AEC_FILTER_LEN, BBOS_RATE,
                       denoise=bool(CFG.aec_denoise),
                       echo_suppress=CFG.aec_echo_suppress,
                       echo_suppress_active=CFG.aec_echo_suppress_active)
        print(f"[aec] Speex AEC active (mic ch{MIC_CH}, ref ch{REF_CH}, "
              f"filter={AEC_FILTER_LEN}, denoise={bool(CFG.aec_denoise)}, "
              f"suppress={CFG.aec_echo_suppress}dB active={CFG.aec_echo_suppress_active}dB)", flush=True)

    near_buf = np.empty(0, dtype=np.int16)
    ref_buf = np.empty(0, dtype=np.int16)
    zero_reads = 0
    loops = 0
    # The loop is paced by the capture stream, so loop<10/s means capture is starving.
    rate_t0 = time.monotonic()

    # Own thread: a 50-200 ms pactl call inline would stall this capture-paced loop.
    def _upkeep_loop():
        while True:
            time.sleep(SUSPEND_REASSERT_S)
            try:
                _ensure_suspend_unloaded(env, "re-assert")
            except Exception:
                pass
    threading.Thread(target=_upkeep_loop, name="pa-upkeep", daemon=True).start()

    mic_type = Type("mic_audio")
    # Deep rings: a consumer that briefly stalls must not lose chunks to 1-buffer overwrite.
    w_mic = Writer("mic.audio", mic_type, keeptime=False, buf_ms=400)
    w_ref = Writer("mic.ref_level", Type("mic_ref_level"), keeptime=False, buf_ms=400)

    with w_mic, w_ref:
        while True:
            raw = _read_exact(rec.stdout, READ_BYTES)
            if raw is None:
                print("[!] capture stream ended", flush=True)
                break

            # Drain to live: PA's record buffer holds seconds and never self-drains, so one
            # chunk per iteration turns any slow iteration into permanent mic-content delay.
            raws = [raw]
            while (len(raws) < CATCHUP_MAX_CHUNKS
                   and _avail_bytes(rec.stdout) >= READ_BYTES):
                nxt = _read_exact(rec.stdout, READ_BYTES)
                if nxt is None:
                    break
                raws.append(nxt)
            if len(raws) > CATCHUP_LOG_CHUNKS:
                print(f"[cap] capture backlog {len(raws) * CFG.chunk_ms / 1000:.1f}s "
                      "— draining to live", flush=True)

            respawned = False
            for raw in raws:
                x = np.frombuffer(raw, dtype=np.int16).reshape(-1, CAP_CH)

                # PA hands out pure silence for a suspended card with no error, so an all-zero
                # run means unsuspend and respawn rather than trust the stream.
                mic_block = x[:, list(ACOUSTIC_COLS)]

                if mic_block.any():
                    zero_reads = 0
                else:
                    zero_reads += 1
                    if zero_reads >= ZERO_RECOVER_READS:
                        print(f"[!] mic silent for ~{zero_reads * CFG.chunk_ms}ms — "
                              "respawning capture", flush=True)
                        subprocess.run(["pactl", "suspend-source", source, "0"],
                                       capture_output=True, env=env)
                        try:
                            rec.terminate()
                            rec.wait(timeout=2)
                        except Exception:
                            rec.kill()
                        rec = _spawn_rec(env, source)
                        near_buf = np.empty(0, dtype=np.int16)
                        ref_buf = np.empty(0, dtype=np.int16)
                        zero_reads = 0
                        respawned = True
                        break   # remaining chunks came from the dead recorder

                # Feed the AEC at unity: gain is applied post-AEC so it never disturbs the adaptive filter.
                mic_f = x[:, MIC_CH].astype(np.float32) / 32768.0
                ref_f = x[:, REF_CH].astype(np.float32) / 32768.0
                mic_rs = rs_mic.resample_chunk(mic_f)
                ref_rs = rs_ref.resample_chunk(ref_f)
                if raw_mode:
                    mic_rs = mic_rs * raw_gain
                mic16 = (np.clip(mic_rs, -1, 1) * 32767).astype(np.int16)
                ref16 = (np.clip(ref_rs, -1, 1) * 32767).astype(np.int16)
                near_buf = np.concatenate([near_buf, mic16])
                ref_buf = np.concatenate([ref_buf, ref16])

                while len(near_buf) >= CHUNK and len(ref_buf) >= CHUNK:
                    near_chunk = near_buf[:CHUNK]
                    ref_chunk = ref_buf[:CHUNK]
                    near_buf = near_buf[CHUNK:]
                    ref_buf = ref_buf[CHUNK:]
                    _rr = float(np.sqrt(np.mean((ref_chunk.astype(np.float64) / 32768.0) ** 2)))
                    if raw_mode:
                        out_i = near_chunk
                    else:
                        out = aec.process_chunk(near_chunk, ref_chunk)
                        if raw_aec:
                            out_f = out.astype(np.float32) * raw_gain
                        else:
                            out_f = out.astype(np.float32)
                        out_i = np.clip(out_f, -32768, 32767).astype(np.int16)
                    with w_mic.buf() as b:
                        b["audio"] = out_i.reshape(-1, CFG.channels)
                    with w_ref.buf() as b:
                        b["dbfs"] = 20.0 * np.log10(_rr) if _rr > 0 else -240.0
            del respawned

            loops += 1
            if loops % STATUS_LOOPS == 0:
                _rt = time.monotonic()
                _dt = max(1e-6, _rt - rate_t0)
                print(f"[rate] loop={STATUS_LOOPS / _dt:.2f}/s", flush=True)
                rate_t0 = _rt

    try:
        rec.terminate()
        rec.wait(timeout=3)
    except Exception:
        rec.kill()


if __name__ == "__main__":
    main()
