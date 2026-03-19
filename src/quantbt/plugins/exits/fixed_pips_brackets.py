from __future__ import annotations

import math

from quantbt.plugins.registry import register_exit


@register_exit("fixed_pips_brackets")
def build_exit(side: str, entry_open: float, prev_low: float, prev_high: float, params: dict, entry=None):
    del prev_low, prev_high, entry

    rr = float(params["rr"])
    stop_pips = params.get("stop_pips")
    if stop_pips is None:
        raise ValueError("fixed_pips_brackets requires stop_pips")

    pip_size = float(params.get("pip_size", 0.0001))
    stop_dist = float(stop_pips) * pip_size
    if not (math.isfinite(stop_dist) and stop_dist > 0):
        return None

    if side == "long":
        sl = entry_open - stop_dist
        tp = entry_open + rr * stop_dist
        return {"sl": sl, "tp": tp, "stop_dist": stop_dist}

    if side == "short":
        sl = entry_open + stop_dist
        tp = entry_open - rr * stop_dist
        return {"sl": sl, "tp": tp, "stop_dist": stop_dist}

    raise ValueError("side must be long/short")
