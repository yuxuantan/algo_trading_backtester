from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SOURCE_PATH = Path(__file__).with_name("IE2026-03 LiqSweep A.py")
_MODULE_NAME = f"{__name__}__impl"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load strategy source from {_SOURCE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_MODULE, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
