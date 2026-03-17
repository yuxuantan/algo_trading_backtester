from __future__ import annotations

import importlib
import warnings
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


def _safe_import_default_plugin(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        warnings.warn(
            f"Skipping unavailable default plugin module '{module_name}': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def load_default_plugins() -> None:
    module_names = [
        # Entries
        "quantbt.plugins.entries.sma_cross",
        "quantbt.plugins.entries.random_entry",
        "quantbt.plugins.entries.donchian_breakout",
        "quantbt.plugins.entries.interequity_liqsweep_entry",
        "quantbt.plugins.entries.interequity_liqsweepb_entry",
        # Exits
        "quantbt.plugins.exits.atr_brackets",
        "quantbt.plugins.exits.time_exit",
        "quantbt.plugins.exits.random_time_exit",
        "quantbt.plugins.exits.interequity_liqsweep_exit",
        "quantbt.plugins.exits.interequity_liqsweepb_exit",
        # Sizing
        "quantbt.plugins.sizing.fixed_risk",
        "quantbt.plugins.sizing.fixed_units",
    ]
    for module_name in module_names:
        _safe_import_default_plugin(module_name)
