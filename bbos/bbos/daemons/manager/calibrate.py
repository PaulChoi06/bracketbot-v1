#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pwd

CURRENT_USER = pwd.getpwuid(os.getuid()).pw_name


def daemon_roots():
    """Return configured and conventional daemon roots in preference order."""
    configured = os.environ.get("BBOS_DAEMONS_PATH")
    if configured:
        yield Path(configured).expanduser()

    # New robot layout, followed by the pre-bbcore checkout layout.
    yield Path.home() / "bbcore" / "bbos" / "bbos" / "daemons"
    yield Path.home() / "bbos" / "bbos" / "daemons"

    # Makes the dispatcher usable directly from a source checkout.
    yield Path(__file__).resolve().parents[1]


def resolve_calibration(requested_name):
    """Find the requested daemon's calibration entry point."""
    seen = set()
    for root in daemon_roots():
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        daemon_dir = root / requested_name
        calibrate_py = daemon_dir / "calibrate.py"
        if calibrate_py.is_file():
            return daemon_dir, calibrate_py
    return None


def main():
    if len(sys.argv) != 2:
        print("Usage: calibrate <daemon-name>")
        return 1

    requested_name = sys.argv[1]
    resolved = resolve_calibration(requested_name)
    if resolved is None:
        print(f"[calibrate] No calibrate.py found for daemon '{requested_name}'")
        return 1

    daemon_dir, calibrate_py = resolved

    os.chdir(daemon_dir)
    os.environ["PATH"] = f"/home/{CURRENT_USER}/.nix-profile/bin:/home/{CURRENT_USER}/.local/bin:/usr/bin:/bin"
    os.environ["DEVENV_TUI"] = "false"
    cmd = [
        "devenv",
        "shell",
        "--quiet",
        "--no-tui",
        "--no-reload",
        "--",
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "python",
        calibrate_py.name,
        requested_name,
    ]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
