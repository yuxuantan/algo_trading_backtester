from __future__ import annotations

import numpy as np

from quantbt.plugins.registry import register_sizing


@register_sizing("fixed_risk")
def size_fn(*, cfg, equity, side, entry_open, exit_spec, entry=None, params=None):
    params = params or {}
    risk_pct = float(params.get("risk_pct", cfg.risk_pct))
    risk_dollars = equity * risk_pct

    stop_dist = float(exit_spec.get("stop_dist", 0.0)) if exit_spec else 0.0
    if stop_dist > 0:
        units = risk_dollars / stop_dist
    else:
        units = risk_dollars / (10 * cfg.pip_size)
    if not np.isfinite(units) or units <= 0:
        return None
    return float(units)
