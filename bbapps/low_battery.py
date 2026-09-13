# /// script
# dependencies = [
#   "bbos",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""
Low Battery Alert — monitors voltage, plays a random quip every 15s when low.
"""
import subprocess
import random
import time
from pathlib import Path
from bbos import Reader

LOW_BATTERY_THRESHOLD = 17.4
ALERT_INTERVAL = 15

WAVS_DIR = Path(__file__).parent / "play_sound" / "wavs"
PLAY_WAV = Path(__file__).parent / "play_sound" / "main.py"
APP_NAME = Path(__file__).stem  # "low_battery"

def play_random_alert():
    wavs = sorted(WAVS_DIR.glob(f"{APP_NAME}_*.wav"))
    if not wavs:
        print(f"[LOW_BATTERY] No {APP_NAME}_*.wav files found in {WAVS_DIR}")
        return
    chosen = random.choice(wavs)
    rel_name = "wavs/" + chosen.stem
    print(f"[LOW_BATTERY] Playing: {chosen.name}")
    subprocess.Popen(
        ["uv", "run", "--script", str(PLAY_WAV), "--", rel_name],
        cwd=str(PLAY_WAV.parent),
    )

def main():
    print(f"[LOW_BATTERY] Monitoring battery voltage (threshold: {LOW_BATTERY_THRESHOLD}V)")
    print(f"[LOW_BATTERY] Alert every {ALERT_INTERVAL}s when low")

    alert_active = False
    last_alert_time = 0

    with Reader("drive.status", sync=True) as r_status:
        while True:
            if r_status.ready():
                voltage = float(r_status.data["voltage"])
                now = time.monotonic()

                if voltage < LOW_BATTERY_THRESHOLD:
                    if not alert_active or now - last_alert_time >= ALERT_INTERVAL:
                        if not alert_active:
                            print(f"[LOW_BATTERY] Battery low! {voltage:.2f}V")
                        alert_active = True
                        play_random_alert()
                        last_alert_time = now
                else:
                    if alert_active:
                        print(f"[LOW_BATTERY] Battery OK: {voltage:.2f}V")
                        alert_active = False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[LOW_BATTERY] Stopped")
