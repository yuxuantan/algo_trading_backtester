from __future__ import annotations

import math
import pandas as pd

from quantbt.plugins.entries.interequity_liqsweep_entry import get_cached_bracket
from quantbt.plugins.registry import register_exit


@register_exit("interequity_liqsweep_exit")
def build_exit(side: str, entry_open: float, prev_low: float, prev_high: float, params: dict, entry=None):
    """
    InterEquity LiqSweep exit plugin.

    Priority:
    1) If the paired entry plugin has a cached strategy-native bracket for this
       entry_time+side, use it.
    2) Fallback to a simple bracket derived from prev candle structure.
    """
    if entry is not None:
        entry_time = entry.get("entry_time")
        if entry_time is not None:
            cached = get_cached_bracket(pd.Timestamp(entry_time), side)
            if cached is not None:
                return cached

    rr = float(params.get("rr", params.get("min_rr", 2.0)))
    if rr <= 0:
        return None

    sl_buffer_pips = float(params.get("sl_buffer_pips", 1.0))
    pip_size = float(params.get("pip_size", 0.0001))
    sl_buffer = sl_buffer_pips * pip_size

    if side == "long":
        sl = float(prev_low) - sl_buffer
        stop_dist = float(entry_open) - sl
        if not math.isfinite(stop_dist) or stop_dist <= 0:
            return None
        tp = float(entry_open) + rr * stop_dist
        return {"sl": sl, "tp": tp, "stop_dist": stop_dist}

    if side == "short":
        sl = float(prev_high) + sl_buffer
        stop_dist = sl - float(entry_open)
        if not math.isfinite(stop_dist) or stop_dist <= 0:
            return None
        tp = float(entry_open) - rr * stop_dist
        return {"sl": sl, "tp": tp, "stop_dist": stop_dist}

    raise ValueError("side must be long/short")

