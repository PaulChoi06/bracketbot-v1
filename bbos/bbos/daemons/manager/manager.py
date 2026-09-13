#!/usr/bin/env python3
import json, os, pwd, subprocess, sys, time, signal
from multiprocessing import Process
from pathlib import Path
import threading

CURRENT_USER = pwd.getpwuid(os.getuid()).pw_name
ENV_KEYS = {
    "PATH",
    "LD_LIBRARY_PATH",
    "CPATH",
    "LIBRARY_PATH",
    "PKG_CONFIG_PATH",
    "NIX_CFLAGS_COMPILE",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "UV_PYTHON",
    "UV_PYTHON_DOWNLOADS",
    "UV_LINK_MODE",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PREVIEW",
    "CUDA_PATH",
    "CUDA_HOME",
    "CUDACXX",
    "CUDAHOSTCXX",
    "TORCH_CUDA_ARCH_LIST",
    "FORCE_CUDA",
    "CUDA_VISIBLE_DEVICES",
    "LDFLAGS",
    "CFLAGS",
}
ENV_PREFIXES = ("CUDA_", "NIX_", "UV_", "TORCH_")

class ManagedProc:

    def __init__(self, name: str, cwd: Path):
        self.name = name
        self.cwd = cwd
        self.proc = None
        self.last_start = 0.0

    def _launch(self):
        # Give each daemon (uv plus its Python child) its own process group so
        # stopping the wrapper cannot leave an orphaned writer behind.
        os.setsid()
        os.chdir(self.cwd)
        log_path = f"/dev/shm/{self.name}.log"
        env = os.environ.copy()
        env.update(self._cached_env())
        cmd = [
            "uv", "run", "--frozen", "--no-sync",
            "python", "daemon.py", self.name,
        ]
        with open(log_path, "wb", buffering=0) as log_fd:
            os.dup2(log_fd.fileno(), 1)
            os.dup2(log_fd.fileno(), 2)
            os.write(1, f"now running daemon: {self.name}\n".encode())
            os.execvpe(cmd[0], cmd, env)

    def _env_dir(self):
        return self.cwd / ".devenv"

    def _env_cache_path(self):
        return self._env_dir() / "bbos-env.json"

    def _env_stamp_path(self):
        return self._env_dir() / "bbos-env.stamp"

    def _env_inputs(self):
        inputs = [
            self.cwd / "devenv.nix",
            self.cwd / "devenv.lock",
            self.cwd / "pyproject.toml",
            self.cwd / "uv.lock",
        ]
        inputs.extend(sorted(self.cwd.parent.glob("*.nix")))
        return [path for path in inputs if path.exists()]

    def _env_cache_stale(self):
        cache_path = self._env_cache_path()
        stamp_path = self._env_stamp_path()
        python_path = self.cwd / ".venv/bin/python"
        if not cache_path.exists() or not stamp_path.exists() or not python_path.exists():
            return True
        stamp_mtime = stamp_path.stat().st_mtime
        return any(path.stat().st_mtime > stamp_mtime for path in self._env_inputs())

    def _env_cache_available(self):
        return self._env_cache_path().exists() and (self.cwd / ".venv/bin/python").exists()

    def _cached_env(self):
        with open(self._env_cache_path()) as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}

    def _refresh_env_cache(self):
        self._env_dir().mkdir(exist_ok=True)
        capture = r'''
import json
import os

keys = set(os.environ.get("BBOS_ENV_KEYS", "").split(":"))
prefixes = tuple(filter(None, os.environ.get("BBOS_ENV_PREFIXES", "").split(":")))
env = {
    key: value
    for key, value in os.environ.items()
    if key in keys or key.startswith(prefixes)
}
tmp = ".devenv/bbos-env.json.tmp"
with open(tmp, "w") as f:
    json.dump(env, f, sort_keys=True)
os.replace(tmp, ".devenv/bbos-env.json")
'''
        env = os.environ.copy()
        env.update({
            "PATH": f"/home/{CURRENT_USER}/.nix-profile/bin:/home/{CURRENT_USER}/.local/bin:/usr/bin:/bin",
            "DEVENV_TUI": "false",
            "BBOS_ENV_KEYS": ":".join(sorted(ENV_KEYS)),
            "BBOS_ENV_PREFIXES": ":".join(ENV_PREFIXES),
        })
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
            "-c",
            capture,
        ]
        subprocess.run(cmd, cwd=self.cwd, env=env, check=True)
        self._env_stamp_path().touch()

    def _ensure_env_cache(self):
        if self._env_cache_stale():
            print(f"[manager] {self.name}: refreshing devenv cache")
            try:
                self._refresh_env_cache()
            except subprocess.CalledProcessError as exc:
                if self._env_cache_available():
                    print(
                        f"[manager] {self.name}: devenv cache refresh failed "
                        f"({exc.returncode}); using previous cached env"
                    )
                    return
                print(
                    f"[manager] {self.name}: devenv cache refresh failed "
                    f"({exc.returncode}); no previous cached env available"
                )
                raise

    def _wait_for_ready(self, timeout=10.0):
        log_path = Path(f"/dev/shm/{self.name}.log")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not log_path.exists():
                time.sleep(0.1)
                continue
            with open(log_path, "rb") as f:
                lines = f.readlines()[-10:]
                if any(b"now running daemon" in l for l in lines):
                    self.ready = True
                    print(f"[manager] {self.name} is now running:")
                    return
            time.sleep(0.2)
        print(
            f"[manager] {self.name} did not signal readiness within {timeout} sec"
        )

    def _cleanup_devenv_shells(self):
        devenv_dir = self.cwd / ".devenv"
        if not devenv_dir.is_dir():
            return
        for path in devenv_dir.glob("shell-*.sh"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[manager] {self.name}: could not remove {path.name}: {exc}")

    def has_daemon(self):
        return (self.cwd / "daemon.py").exists()

    def is_stopped(self):
        return (self.cwd / ".stopped").exists()

    def is_disabled(self):
        return (self.cwd / ".disabled").exists()

    def can_run(self):
        return self.has_daemon() and not self.is_stopped() and not self.is_disabled()

    def start(self):
        if self.proc or time.monotonic() - self.last_start < 1.0:
            return
        if not self.can_run():
            if not self.has_daemon():
                print(f"[manager] skipping {self.name}: daemon.py not found")
            elif self.is_disabled():
                print(f"[manager] skipping {self.name}: disabled")
            elif self.is_stopped():
                print(f"[manager] skipping {self.name}: stopped")
            return
        self._cleanup_devenv_shells()
        try:
            self._ensure_env_cache()
        except subprocess.CalledProcessError:
            return
        self._cleanup_devenv_shells()
        self.proc = Process(target=self._launch, name=self.name)
        self.proc.start()
        self.last_start = time.monotonic()
        print(f"[manager] initializing {self.name}... (pid={self.proc.pid})")
        self._wait_for_ready()
        self._cleanup_devenv_shells()

    @staticmethod
    def _group_exists(pgid):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _wait_for_group_exit(self, pgid, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Reap the tracked wrapper so a dead leader does not make its
            # process group appear alive while descendants are being checked.
            if self.proc and self.proc.pid == pgid:
                self.proc.join(timeout=0)
            if not self._group_exists(pgid):
                return True
            time.sleep(0.05)
        return not self._group_exists(pgid)

    def _stop_group(self, pgid, sig=signal.SIGINT):
        for group_signal, timeout in (
                (sig, 5.0),
                (signal.SIGTERM, 2.0),
                (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pgid, group_signal)
            except ProcessLookupError:
                return
            except PermissionError as exc:
                print(f"[manager] {self.name}: cannot signal process group "
                      f"{pgid}: {exc}")
                return
            if self._wait_for_group_exit(pgid, timeout):
                return
        print(f"[manager] {self.name}: process group {pgid} survived SIGKILL")

    def stop(self, sig=signal.SIGINT):
        if not self.proc:
            return
        pgid = self.proc.pid
        self._stop_group(pgid, sig)
        self.proc.join(timeout=0)
        self.proc = None

    def check_alive(self):
        if not self.can_run():
            if self.proc:
                print(f"[manager] {self.name}: no longer runnable, stopping")
                self.stop()
            return
        if self.proc and self.proc.exitcode is not None:
            print(f"[manager] {self.name} exited ({self.proc.exitcode}), restarting")
            # uv can exit before its Python child. Kill any descendants still
            # occupying the old process group before starting a replacement.
            self._stop_group(self.proc.pid)
            self.proc.join(timeout=0)
            self.proc = None
            self.start()
        elif not self.proc:
            self.start()


def discover_daemons(root: Path):
    for p in root.glob("*/devenv.nix"):
        yield ManagedProc(p.parent.name, p.parent)


def main():
    if len(sys.argv) < 2:
        print("Usage: manager <daemons_dir> [only]")
        sys.exit(1)
    args = type('Args', (), {'daemons_dir': sys.argv[1], 'only': sys.argv[2:]})()
    procs = list(discover_daemons(Path(args.daemons_dir)))
    print(args.daemons_dir)
    if args.only:
        procs = [proc for proc in procs if proc.name in args.only]
    print(f"Managing daemons: {', '.join(p.name for p in procs)}")

    running = True
    def shutdown(signum, _):
        nonlocal running
        print(f"\n[manager] signal {signum}, shutting down…")
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    
    def start_proc(proc):
        proc.start()
    
    threads = []
    for p in procs:
        thread = threading.Thread(target=start_proc, args=(p,))
        thread.start()
        threads.append(thread)
    
    # Wait for all startup threads to complete
    for thread in threads:
        thread.join()

    while running:
        for p in procs:
            p.check_alive()
        time.sleep(1.0)

    for p in procs:
        p.stop()
    print("[manager] done")


if __name__ == "__main__":
    main()
