from __future__ import annotations

import math

from quantbt.plugins.registry import register_exit


@register_exit("atr_brackets")
def build_exit(side: str, entry_open: float, prev_low: float, prev_high: float, params: dict, entry=None):
    if entry is None:
        return None
    atr = float(entry.get("atr", float("nan")))
    if not math.isfinite(atr) or atr <= 0:
        return None

    rr = float(params["rr"])
    sldist = params.get("sldist_atr_mult", params.get("sldist_atr"))
    if sldist is None:
        raise ValueError("atr_brackets requires sldist_atr_mult")
    stop_dist = float(sldist) * atr
    if stop_dist <= 0:
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


build_exit.requires_atr = True
