from __future__ import annotations

import math
from typing import Any


def build_param_space_values(spec: dict[str, Any]) -> list[int] | list[float]:
    integer = bool(spec.get("integer", False))
    raw_end = spec.get("end", spec.get("stop"))
    if not (all(name in spec for name in ("start", "step")) and raw_end is not None):
        return []

    start = float(spec["start"])
    stop = float(raw_end)
    step = float(spec["step"])
    if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step) or step <= 0:
        raise ValueError(f"invalid numeric param-space range: {spec!r}")

    values: list[int] | list[float] = []
    current = start
    tolerance = max(abs(step) / 1000.0, 1e-12)
    while current <= stop + tolerance:
        values.append(int(round(current)) if integer else round(current, 12))
        current += step
    return values


def build_param_space_from_limited_test(strategy_cfg: dict[str, Any]) -> dict[str, list[int] | list[float]]:
    limited_cfg = strategy_cfg.get("limited_test", {}) if isinstance(strategy_cfg, dict) else {}
    param_space: dict[str, list[int] | list[float]] = {}

    for section_name in ("entry", "exit"):
        section_cfg = limited_cfg.get(section_name, {}) if isinstance(limited_cfg, dict) else {}
        raw_optimizable = section_cfg.get("optimizable", {}) if isinstance(section_cfg, dict) else {}
        if isinstance(raw_optimizable, list):
            raw_optimizable = {str(key): {} for key in raw_optimizable}
        if not isinstance(raw_optimizable, dict):
            continue

        for raw_key, raw_spec in raw_optimizable.items():
            key = str(raw_key or "").strip()
            if not key or key in param_space or not isinstance(raw_spec, dict):
                continue
            values = build_param_space_values(raw_spec)
            if values:
                param_space[key] = values

    return param_space


def min_max_rr_constraint(params: dict[str, Any]) -> bool:
    if "min_rr" in params and "max_rr" in params:
        return float(params["max_rr"]) > float(params["min_rr"])
    return True
