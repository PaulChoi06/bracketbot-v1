import importlib.util, sys
from pathlib import Path
import copy
from inspect import isclass
from collections import defaultdict, deque
import types
import ast

# PEP 562
_symbols = {
    "Type": "bbos.registry",
    "Config": "bbos.registry",
    "register": "bbos.registry",
    "realtime": "bbos.registry",
    "state": "bbos.registry",
    "on_writer_open": "bbos.registry",
    "all_configs": "bbos.registry",
    "all_types": "bbos.registry",
    "Writer": "bbos.ipc",
    "Reader": "bbos.ipc",
    "AppManager": "bbos.app_manager",
}

try:
    m = sys.modules.get("__main__")
    if m is not None and not hasattr(m, "__file__"):
        m.__file__ = str((Path.cwd() / "-c").resolve())
except Exception:
    pass

__all__ = list(_symbols.keys())

def __dir__():
    return list(globals().keys()) + list(_symbols.keys())

_collected = False
def __getattr__(name):
    global _collected
    if name in _symbols:
        if name in ("Config", "Type", "all_configs", "all_types") and not _collected:
            _collected = True
            _collect_daemon_constants()
        mod = importlib.import_module(_symbols[name])
        val = getattr(mod, name)
        globals()[name] = val  # cache for next time
        return val
    raise AttributeError(f"module 'bbos' has no attribute '{name}'")

def _collect_daemon_constants():
    base = Path(__file__).parent / "daemons"
    if not base.is_dir():
        return

    paths = {}
    config_to_module = {}   # "camera" -> "bbos.daemons.camera.constants"
    deps = defaultdict(set)

    def scan_constants(pth):
        src = pth.read_text()
        tree = ast.parse(src, filename=str(pth))
        configs = set()
        deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "register":
                        configs.add(node.name)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "Config":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        deps.add(node.args[0].value)
        return configs, deps

    for sub in base.iterdir():
        pth = sub / "constants.py"
        if not pth.is_file():
            continue
        mod_name = f"{__name__}.daemons.{sub.name}.constants"
        paths[mod_name] = pth
        configs, cur_deps = scan_constants(pth)
        for config in configs:
            config_to_module[config] = mod_name
        deps[mod_name] = cur_deps

    # topo sort modules by deps
    normalize_deps = {k: {config_to_module[d] for d in deps[k]} for k in deps}
    order = _topo_sort(normalize_deps)
    # second pass: real execution in order
    for mod_name in order:
        spec = importlib.util.spec_from_file_location(mod_name, paths[mod_name])
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[!] Error loading {mod_name}: {e}")

def _topo_sort(deptree):
    indeg = defaultdict(int)
    adj = defaultdict(set)
    for src, tgts in deptree.items():
        for t in tgts:
            adj[t].add(src)
            indeg[src] += 1

    for k in deptree:
        indeg.setdefault(k, 0)

    q = deque([m for m, d in indeg.items() if d == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(indeg):
        raise ValueError("Cycle detected in config dependencies")
    return order