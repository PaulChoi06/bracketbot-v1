"""Playback: speaker.audio -> gain -> compressor -> upsample -> pacat."""
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import soxr

from bbos import Config, Reader

CFG = Config("speaker")

RATE = CFG.sample_rate
CHANNELS = CFG.channels
CHUNK_MS = CFG.chunk_ms
DEVICE_RATE = 48_000
TOKENS = ("speakerphone", "bracketbot")   # substrings of the head board's USB product string
COMP_THRESHOLD_DBFS = -18.0
COMP_RATIO = 3.0
GOV_POLL_S = 2.0
GOV_PA_LIMIT_MS = 400.0
GOV_SILENT_RMS = 0.003
SUSPEND_REASSERT_S = 30
STATUS_LOOPS = max(1, 10 * 1000 // CHUNK_MS)
RECOVER_TRIES = 6
RECOVER_WAIT_S = 2.0


class SmoothGain:
    """Glides gain across a chunk so a level change cannot click."""

    def __init__(self, initial=1.0):
        self._g = float(initial)

    def apply(self, x, target):
        x = np.asarray(x, dtype=np.float32)
        target = float(target)
        if x.size == 0:
            self._g = target
            return x
        y = x * np.linspace(self._g, target, x.size, dtype=np.float32)
        self._g = target
        return y


class PeakCompressor:
    """Shaves playback peaks, which set the barge gate's echo bar. Slow release: instant pumped."""

    def __init__(self, threshold_dbfs, ratio, frame_s=0.1, attack_s=0.010, release_s=0.300):
        self._thr = float(threshold_dbfs)
        self._ratio = float(ratio)
        self._att = min(1.0, frame_s / max(attack_s, 1e-3))
        self._rel = min(1.0, frame_s / max(release_s, 1e-3))
        self._g = 1.0

    def apply(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
        over = 20.0 * np.log10(rms) - self._thr
        target = 1.0 if over <= 0 else 10.0 ** (-(over - over / self._ratio) / 20.0)
        prev = self._g
        step = self._att if target < self._g else self._rel
        self._g += (target - self._g) * step
        return x * np.linspace(prev, self._g, x.size, dtype=np.float32)


def _matches(text):
    t = text.lower()
    return any(tok in t for tok in TOKENS)


def _candidate_servers():
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


def _find_sink(env):
    r = subprocess.run(["pactl", "list", "sinks", "short"],
                       capture_output=True, text=True, timeout=3, env=env)
    for line in r.stdout.splitlines():
        if _matches(line):
            return line.split("\t")[1]
    return None


def _pulse_env():
    """Pin PULSE_SERVER to whichever server exposes the sink; several run here."""
    for server in _candidate_servers():
        env = {**os.environ, "PULSE_SERVER": server}
        if _find_sink(env):
            os.environ["PULSE_SERVER"] = server
            print(f"[pa] server: {server}", flush=True)
            return env
    default_env = os.environ.copy()
    if _find_sink(default_env):
        print("[pa] server: inherited default", flush=True)
        return default_env
    print("[pa] WARNING: sink not found on any PulseAudio server", flush=True)
    return default_env


def _all_envs(base_env):
    envs = [base_env]
    seen = {base_env.get("PULSE_SERVER")}
    if None not in seen:
        envs.append({k: v for k, v in base_env.items() if k != "PULSE_SERVER"})
        seen.add(None)
    for server in _candidate_servers():
        if server and server not in seen:
            seen.add(server)
            envs.append({**base_env, "PULSE_SERVER": server})
    return envs


def _ensure_suspend_unloaded(base_env, reason, sweep=False):
    """A suspended USB card stops accepting playback, so keep the module off."""
    for env in (_all_envs(base_env) if sweep else [base_env]):
        try:
            r = subprocess.run(["pactl", "list", "modules", "short"],
                               capture_output=True, text=True, timeout=3, env=env)
            if any("module-suspend-on-idle" in ln for ln in r.stdout.splitlines()):
                subprocess.run(["pactl", "unload-module", "module-suspend-on-idle"],
                               capture_output=True, text=True, timeout=3, env=env)
                print(f"[pa] unloaded module-suspend-on-idle ({reason}) on "
                      f"{env.get('PULSE_SERVER') or 'default'}", flush=True)
        except Exception as e:
            print(f"[pa] suspend re-check failed: {e!r}", flush=True)


def _recover(env, attempt):
    """Nudge PulseAudio to re-expose the sink after a USB re-enumeration."""
    if attempt == 0:
        return
    if attempt == 1:
        print("[recover] reloading module-udev-detect", flush=True)
        for e in _all_envs(env):
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


def _cleanup_stale():
    """Kill an orphaned pacat from a previous instance still holding the sink."""
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if "pacat" in cmd and _matches(cmd):
            try:
                os.kill(int(pid), 15)
                print(f"[cleanup] killed stale pacat {pid}", flush=True)
            except ProcessLookupError:
                pass


def _pacat_latency_ms(env):
    """pacat's sink-input latency in ms, or 0.0 if unavailable."""
    try:
        r = subprocess.run(["pactl", "list", "sink-inputs"],
                           capture_output=True, text=True, timeout=3, env=env)
        in_pacat = False
        buf_us = sink_us = 0.0
        for ln in r.stdout.splitlines():
            t = ln.strip()
            if t.startswith("Sink Input #"):
                in_pacat = False
                buf_us = sink_us = 0.0
            elif t.startswith("Buffer Latency:"):
                buf_us = float(t.split(":")[1].strip().split()[0])
            elif t.startswith("Sink Latency:"):
                sink_us = float(t.split(":")[1].strip().split()[0])
            elif 'application.name = "pacat"' in t:
                in_pacat = True
            if in_pacat and buf_us + sink_us > 0:
                return (buf_us + sink_us) / 1000.0
        return 0.0
    except Exception:
        return 0.0


def main():
    _cleanup_stale()
    env = _pulse_env()
    _ensure_suspend_unloaded(env, "startup", sweep=True)
    sink = _find_sink(env)
    attempt = 0
    while not sink and attempt < RECOVER_TRIES:
        print(f"[!] sink not found -- recovery {attempt + 1}/{RECOVER_TRIES}", flush=True)
        _recover(env, attempt)
        time.sleep(RECOVER_WAIT_S)
        env = _pulse_env()
        _ensure_suspend_unloaded(env, "recover", sweep=True)
        sink = _find_sink(env)
        attempt += 1
    if not sink:
        print(f"[!] sink not found after {RECOVER_TRIES} attempts; exiting for respawn",
              flush=True)
        sys.exit(1)
    print(f"[pa] sink: {sink}", flush=True)
    subprocess.run(["pactl", "set-sink-mute", sink, "0"], capture_output=True, timeout=3, env=env)
    subprocess.run(["pactl", "set-sink-volume", sink, CFG.pulse_volume],
                   capture_output=True, timeout=3, env=env)

    pacat = shutil.which("pacat") or "pacat"
    play = subprocess.Popen(
        [pacat, f"--device={sink}", f"--channels={CHANNELS}", f"--rate={DEVICE_RATE}",
         "--format=s16le", "--raw", f"--latency-msec={CHUNK_MS}"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, bufsize=0)

    rs = soxr.ResampleStream(RATE, DEVICE_RATE, 1, dtype=np.float32)
    gain = SmoothGain(float(CFG.volume))
    comp = PeakCompressor(COMP_THRESHOLD_DBFS, COMP_RATIO) if CFG.comp_enabled else None
    print(f"[spk] {RATE}->{DEVICE_RATE}Hz gain={CFG.volume} "
          f"comp={'ON' if comp else 'off'}", flush=True)

    vol, muted = float(CFG.volume), False
    loops, fwd = 0, 0
    rate_t0 = time.monotonic()

    # pactl costs 50-200 ms, so the governor samples it off this paced loop.
    gov = {"pa_ms": 0.0, "skip": 0}

    def _gov_loop():
        while True:
            gov["pa_ms"] = _pacat_latency_ms(env)
            time.sleep(GOV_POLL_S)
    threading.Thread(target=_gov_loop, name="pa-gov", daemon=True).start()

    def _upkeep_loop():
        while True:
            time.sleep(SUSPEND_REASSERT_S)
            try:
                _ensure_suspend_unloaded(env, "re-assert")
            except Exception:
                pass
    threading.Thread(target=_upkeep_loop, name="pa-upkeep", daemon=True).start()

    r_audio = Reader("speaker.audio")
    r_vol = Reader("speaker.volume")

    with r_audio, r_vol:
        while True:
            if r_vol.ready():
                vol = float(r_vol.data["gain"])
                muted = bool(r_vol.data["mute"])

            # bbos Readers pace the loop: exactly ONE ready() per iteration.
            if r_audio.ready():
                fwd += 1
                spk = r_audio.data["audio"].reshape(-1).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(spk * spk)) + 1e-12)
                if gov["pa_ms"] > GOV_PA_LIMIT_MS and rms < GOV_SILENT_RMS:
                    gov["skip"] += 1
                else:
                    spk = np.clip(gain.apply(spk, 0.0 if muted else vol), -1.0, 1.0)
                    if comp is not None:
                        spk = comp.apply(spk)
                    out = (rs.resample_chunk(spk) * 32767).astype(np.int16)
                    if out.size:
                        try:
                            play.stdin.write(out.tobytes())
                        except (BrokenPipeError, ValueError):
                            print("[!] playback pipe closed", flush=True)
                            break

            loops += 1
            if loops % STATUS_LOOPS == 0:
                dt = max(1e-6, time.monotonic() - rate_t0)
                print(f"[rate] loop={STATUS_LOOPS / dt:.2f}/s fwd={fwd / dt:.2f}/s "
                      f"pa={gov['pa_ms']:.0f}ms gov_skip={gov['skip']}", flush=True)
                rate_t0 = time.monotonic()
                fwd = 0
                gov["skip"] = 0

    try:
        play.terminate()
        play.wait(timeout=3)
    except Exception:
        play.kill()


if __name__ == "__main__":
    main()
