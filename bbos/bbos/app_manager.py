import os, json, signal, time, subprocess, multiprocessing as mp
from pathlib import Path
from typing import Dict, List
from multiprocessing import Process
import multiprocessing as mp
import pwd

# Configuration
RATE_LIMIT_INTERVAL: float = 2.0  # seconds between app starts
PROCESS_STOP_TIMEOUT: float = 5.0  
CURRENT_USER = pwd.getpwuid(os.getuid()).pw_name

LOCK_DIR = Path("/dev/shm")
APP_DIRS = [Path.home() / "bbapps"]   # filesystem fallback for app discovery

def get_lock_path(app): return LOCK_DIR / f"app-{app}_lock"

def get_registry_path(): return LOCK_DIR / "app-manager_lock"

def stop_app(app):
    lock = get_lock_path(app)
    if lock.exists():
        lock.unlink()
    else:
        print(f"[app-manager] {app} is not running!")
        return False
    return True

def _write_lock(lock: Path, content: str):
    """Atomically write `content` into `lock` (temp file + rename on same fs)."""
    tmp = lock.with_name(f"{lock.name}.tmp.{os.getpid()}")
    tmp.write_text(content)
    os.replace(tmp, lock)

def start_app(name, args=""):
    lock = get_lock_path(name)
    if lock.exists():
        print(f"[app-manager] {name} is already running!")
        return False
    _write_lock(lock, args)
    return True

def read_app_args(name) -> str:
    """Read the args an app was started with from its lock file ('' if none)."""
    try:
        return get_lock_path(name).read_text().strip()
    except FileNotFoundError:
        return ""

def _app_path(name) -> str:
    """Absolute path of the app's entry file, from the registry ('' if unknown)."""
    try:
        with open(get_registry_path(), "r") as fd:
            return json.load(fd).get(name, "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""

def _pgrep_alive(name) -> bool:
    """True if a process for this app is still running (matched by its entry path)."""
    path = _app_path(name)
    if not path:
        return False
    return subprocess.run(["pgrep", "-f", path],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0

def _wait_for_exit(name, *, alive=None, timeout=20.0, poll=0.2,
                   clock=time.monotonic, sleep=time.sleep) -> bool:
    """True if the app's process exits within `timeout` (injectables for tests)."""
    alive = alive or (lambda: _pgrep_alive(name))
    deadline = clock() + timeout
    while clock() < deadline:
        if not alive():
            return True
        sleep(poll)
    return not alive()

def restart_app(name, args=None, *, wait=None):
    """Stop, wait for real exit, then restart; reuses stored args unless given.
    Returns False (left stopped) if it doesn't exit in time, never double-running."""
    lock = get_lock_path(name)
    if not lock.exists():
        print(f"[app-manager] {name} is not running!")
        return False
    preserved = lock.read_text() if args is None else args
    lock.unlink()
    wait = wait or (lambda: _wait_for_exit(name))
    if not wait():
        print(f"[app-manager] {name} did not exit in time; left stopped")
        return False
    _write_lock(get_lock_path(name), preserved)
    return True

def is_known_app(name) -> bool:
    """True if `name` is an app the manager knows about (registry or on disk)."""
    try:
        with open(get_registry_path()) as fd:
            if name in json.load(fd):
                return True
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    for d in APP_DIRS:
        if (Path(d) / f"{name}.py").exists() or (Path(d) / name / "main.py").exists():
            return True
    return False

def cli_app_command(verb, name, args="") -> int:
    """App-side entry for the start/stop/restart shell commands; returns exit code."""
    if not is_known_app(name):
        print(f"[app-manager] unknown app: {name}")
        return 1
    if verb == "start":
        if start_app(name, args):
            print(f"[app-manager] Started app: {name}")
            return 0
        return 1
    if verb == "stop":
        if not stop_app(name):
            return 1
        # Lock removal only *requests* the stop; the app_manager sweep does the
        # actual kill. Verify the process really exited so the message never
        # overstates (fix #1 guarantees a SIGKILL, so this normally returns fast).
        if _wait_for_exit(name):
            print(f"[app-manager] Stopped app: {name}")
            return 0
        print(f"[app-manager] {name}: lock removed but process still alive after "
              f"timeout — investigate")
        return 1
    if verb == "restart":
        if restart_app(name):
            print(f"[app-manager] Restarted app: {name}")
            return 0
        return 1
    print(f"[app-manager] unknown verb: {verb}")
    return 2

def get_status(exclude: List[str] = []) -> Dict:
    """Get complete status of all apps and lock files"""
    with open(get_registry_path(), "r") as fd:
        app_paths = json.load(fd)
    app_status = {}
    for app in app_paths:
        if app in exclude:
            continue
        app_status[app] = get_lock_path(app).exists()
    return app_status

class AppManager:
    def __init__(self, app_dirs: List[Path] | Path):
        self.app_dirs = app_dirs if isinstance(app_dirs, list) else [app_dirs]
        mp.set_start_method("fork", force=True)
        self.processes: Dict[str, Process] = {}
        self.last_start: Dict[str, float] = {}
        self.app_paths: Dict[str, Path] = {}
        self.autostart: List[str] = []
        self.app_args: Dict[str, str] = {}
        self.get_available_apps()

    def get_available_apps(self) -> List[str]:
        """Get list of available apps """
        def is_autostart(app_name: str) -> bool:
            for app_dir in self.app_dirs:
                autostart_file = app_dir / ".autostart"
                if autostart_file.exists():
                    for line in autostart_file.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(maxsplit=1)
                        name = parts[0]
                        if name == app_name:
                            if len(parts) > 1:
                                self.app_args[app_name] = parts[1]
                            return True
            return False
        apps = []
        for app_dir in self.app_dirs:
            # Check for .py files
            for app_file in app_dir.glob("*.py"):
                app_name = app_file.stem
                apps.append(app_name)
                self.app_paths[app_name] = app_file.absolute()
                if is_autostart(app_name):
                    self.autostart.append(app_name)
            
            # Check for folders with main.py
            for folder in app_dir.iterdir():
                app_name = folder.name
                if folder.is_dir():
                    main_file = folder / "main.py"
                    if main_file.exists():
                        apps.append(app_name)
                        self.app_paths[app_name] = main_file.absolute()
                        if is_autostart(app_name):
                            self.autostart.append(app_name)
        with open(get_registry_path(), "w") as fd:
            json.dump({k: str(v.absolute()) for k, v in self.app_paths.items()}, fd)
        return apps
    
    def is_app_running(self, app: str) -> bool:
        print(f"is app {app} alive: ", self.processes[app].is_alive() if app in self.processes else None, flush=True)
        return app in self.processes and self.processes[app].is_alive()

   # ── dashboard/launch side ─────────────────────────────────────────────
    def _launch_app(self, app):
        os.setsid()                                    # ① new session = new PGID
        os.chdir(self.app_paths[app].parent)
        os.environ["PATH"] += f"/home/{CURRENT_USER}/.local/bin"
        log_fd = open(LOCK_DIR / f"app-{app}.log", "wb", 0)
        os.dup2(log_fd.fileno(), 1)
        os.dup2(log_fd.fileno(), 2)
        venv_path = str(self.app_paths[app].parent / ".venv")
        args = read_app_args(app)
        if Path(venv_path).exists():
            cmd = f"source {venv_path}/bin/activate && exec python {self.app_paths[app]}"
            if args:
                cmd += f" {args}"
            os.execvp("bash", ["bash", "-c", cmd])
        else:
            if args:
                os.execvp("uv", ["uv", "run", str(self.app_paths[app])] + args.split())
            else:
                os.execvp("uv", ["uv", "run", str(self.app_paths[app])])

    # ── dashboard/stop side ───────────────────────────────────────────────
    @staticmethod
    def _signal_pg(pid, sig):
        """Signal the whole process group; ignore an already-dead group."""
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _reap_within(pid, timeout, poll=0.1):
        """Non-blocking wait for `pid` to die and be reaped, up to `timeout`.

        Must NOT use Process.join(timeout): app children os.execvp() into uv/python,
        which closes the multiprocessing sentinel pipe on exec; join() then sees a
        false "exited" signal and falls through to a *blocking* os.waitpid(pid, 0)
        that ignores the timeout — so a SIGTERM-ignoring app freezes the whole
        reconcile loop. Polling waitpid(WNOHANG) sidesteps that. True once pid gone."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    return True                              # reaped
            except ChildProcessError:
                return True                                  # already reaped elsewhere
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)

    def _stop_app(self, app, timeout=PROCESS_STOP_TIMEOUT, reap=None):
        """Kill the app's process group and untrack it. Does NOT remove the lock —
        the caller owns lock lifecycle, so a concurrent restart that has already
        recreated the lock is not clobbered (the stop/restart race)."""
        reap = reap or self._reap_within
        if self.is_app_running(app):
            pid = self.processes[app].pid
            self._signal_pg(pid, signal.SIGTERM)             # ② SIGTERM the whole group
            if not reap(pid, timeout):
                self._signal_pg(pid, signal.SIGKILL)         # escalate → real SIGKILL
                if not reap(pid, 2.0):
                    # Survived SIGKILL (e.g. uninterruptible sleep): keep it tracked
                    # so the next sweep re-attempts. Don't untrack-and-forget.
                    print(f"[app-manager] {app} survived SIGKILL; will retry", flush=True)
                    return False
        self.processes.pop(app, None)
        return True


    def _start_app(self, app_name: str) -> bool:
        """Start an app"""
        if self.is_app_running(app_name):
            return False  # Already running
        
        # Rate limiting: don't start too frequently
        if app_name in self.last_start and time.time() - self.last_start[app_name] < RATE_LIMIT_INTERVAL:
            return False
        
        try:
            ctx  = mp.get_context("fork")         # optional – keeps code explicit
            proc = ctx.Process(target=self._launch_app, args=(app_name,),
                              name=app_name)
            print(f"[app-manager] Starting {app_name}")
            proc.start()
            self.processes[app_name] = proc
            self.last_start[app_name] = time.time()
            print(f"[app-manager] Started app: {app_name} (pid={proc.pid})")
            return True
            
        except Exception as e:
            print(f"[app-manager] Failed to start {app_name}: {e}")
            return False
    
    def _seed_autostart_locks(self):
        """Create/refresh lock files for autostart apps with their .autostart args.
        Overwrites a stale lock so the app never relaunches with the wrong args."""
        for app in self.autostart:
            print(f"[app-manager] Starting autostart app: {app}")
            _write_lock(get_lock_path(app), self.app_args.get(app, ""))

    def start(self):
        self._seed_autostart_locks()
        try:
            while True:
                for app in self.get_available_apps():
                    if not get_lock_path(app).exists() and app in self.processes:
                        print(f"[app-manager] Detected external delete of {app}_lock. Terminating.")
                        self._stop_app(app)
                    if get_lock_path(app).exists() and app not in self.processes:
                        print(f"[app-manager] Detected external creation of {app}_lock. Starting.")
                        self._start_app(app)
                    if app in self.processes and not self.processes[app].is_alive():
                        print(f"[app-manager] Detected external termination of {app}. Terminating.")
                        self._stop_app(app)
                        stop_app(app)   # crash: drop the stale lock so it isn't relaunched
                    time.sleep(0.05)
        except KeyboardInterrupt:
            self.stop_all()

    def stop_all(self):
        # snapshot + pop so a process that ever survives SIGKILL can't infinite-loop,
        # and use the real _stop_app (there is no stop_app method).
        for app in list(self.processes):
            self._stop_app(app)
            self.processes.pop(app, None)
    
    def __del__(self):
        self.stop_all()