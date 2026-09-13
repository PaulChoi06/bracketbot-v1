# registry.py
from typing import List
from inspect import isclass
from threading import Lock
import numpy as np
import difflib

_periods: dict[str, float] = {}
_types: dict[str, callable] = {}  # functions & callables
_config: dict[str, type] = {}  # classes
_writer_open_hooks: dict[str, callable] = {}  # writer-open callbacks
_lock = Lock()

# --- Configs ---------------------------------------------------------------
def register(robj):
    """Decorator that drtypes functions into _types and classes into _config."""

    def deco(obj):
        key = obj.__name__
        store = _config if isclass(obj) else _types  # branch once

        with _lock:
            if key in store:
                raise ValueError(
                    f"‘{key}’ already registered in {store is _config and 'config' or 'types'}"
                )
            store[key] = obj
        return obj  # object remains intact

    return deco(robj)

# --- Types ---------------------------------------------------------------
def state(robj):
    """Decorator that registers a type for a state variable within a state machine"""
    def deco(obj):
        key = obj.__name__
        assert not isclass(obj), "realtime decorator must be used on a type function"

        with _lock:
            if key in _types:
                raise ValueError(
                    f"‘{key}’ already registered in {_types is _config and 'config' or 'types'}"
                )
            _types[key] = obj
        return obj  # object remains intact
    return deco(robj)

def realtime(ms: int):
    """Decorator that registers a type that updates at a given period in milliseconds."""
    def deco(obj):
        key = obj.__name__
        assert not isclass(obj), "realtime decorator must be used on a type function"

        with _lock:
            if key in _types:
                raise ValueError(
                    f"‘{key}’ already registered in {_types is _config and 'config' or 'types'}"
                )
            _types[key] = obj
            _periods[key] = ms
        return obj  # object remains intact
    return deco

# --- writer-open hooks -----------------------------------------------------
def on_writer_open(name: str):
    """Decorator that registers a callback run when a Writer for `name` opens its channel."""
    def deco(fn):
        _writer_open_hooks[name] = fn
        return fn
    return deco

def run_writer_open_hook(name: str):
    fn = _writer_open_hooks.get(name)
    if fn is not None:
        fn()

# --- helpers ---------------------------------------------------------------
class Type:
    def __init__(self, name: str):
        self._name = name
        if not self._name in _types:
            raise ValueError(f"Type {self._name} not found! Maybe you meant one of: {difflib.get_close_matches(self._name, _types.keys())}")

    @property
    def dtype(self):
        return _types[self._name]()

    def __call__(self, *args, **kwargs):
        return _types[self._name](*args, **kwargs)+[("timestamp", 'datetime64[ns]')], _periods[self._name] if self._name in _periods else None


class Config:
    def __init__(self, name: str):
        if not name in _config:
            raise ValueError(f"Config {name} not found! Maybe you meant one of: {difflib.get_close_matches(name, _config.keys())}")
        else:
            cfg = _config[name]
        for k, v in cfg.__dict__.items():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, v)
        self._name = name


def all_types():
    return {k: v() + [("timestamp", 'datetime64[ns]')] for k, v in _types.items()}


def all_configs():
    return _config
