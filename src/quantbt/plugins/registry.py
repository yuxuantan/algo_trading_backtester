from __future__ import annotations

from typing import Callable, Dict

ENTRY_PLUGINS: Dict[str, Callable] = {}
EXIT_PLUGINS: Dict[str, Callable] = {}
SIZING_PLUGINS: Dict[str, Callable] = {}


def register_entry(name: str):
    def _decorator(fn: Callable):
        ENTRY_PLUGINS[name] = fn
        return fn

    return _decorator


def register_exit(name: str):
    def _decorator(fn: Callable):
        EXIT_PLUGINS[name] = fn
        return fn

    return _decorator


def register_sizing(name: str):
    def _decorator(fn: Callable):
        SIZING_PLUGINS[name] = fn
        return fn

    return _decorator


def get_entry(name: str):
    if name not in ENTRY_PLUGINS:
        raise KeyError(f"entry plugin not found: {name}")
    return ENTRY_PLUGINS[name]


def get_exit(name: str):
    if name not in EXIT_PLUGINS:
        raise KeyError(f"exit plugin not found: {name}")
    return EXIT_PLUGINS[name]


def get_sizing(name: str):
    if name not in SIZING_PLUGINS:
        raise KeyError(f"sizing plugin not found: {name}")
    return SIZING_PLUGINS[name]


def load_default_plugins() -> None:
    # Entries
    from quantbt.plugins.entries import sma_cross  # noqa: F401
    from quantbt.plugins.entries import random_entry  # noqa: F401
    from quantbt.plugins.entries import donchian_breakout  # noqa: F401

    # Exits
    from quantbt.plugins.exits import atr_brackets  # noqa: F401
    from quantbt.plugins.exits import time_exit  # noqa: F401
    from quantbt.plugins.exits import random_time_exit  # noqa: F401

    # Sizing
    from quantbt.plugins.sizing import fixed_risk  # noqa: F401
    from quantbt.plugins.sizing import fixed_units  # noqa: F401
