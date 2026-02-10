from __future__ import annotations

from quantbt.plugins.registry import register_sizing


@register_sizing("fixed_units")
def size_fn(*, cfg, equity, side, entry_open, exit_spec, entry=None, params=None):
    params = params or {}
    units = params.get("units", 0)
    try:
        units = float(units)
    except (TypeError, ValueError):
        return None
    if units <= 0:
        return None
    return units
