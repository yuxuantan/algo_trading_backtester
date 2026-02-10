from __future__ import annotations

import random

from quantbt.plugins.registry import register_exit


@register_exit("random_time_exit")
def build_exit(side: str, entry_open: float, prev_low: float, prev_high: float, params: dict, entry=None):
    hold_values = params.get("hold_bars_values")
    if not hold_values:
        raise ValueError("random_time_exit requires hold_bars_values")

    rng = params.get("rng")
    if rng is None:
        seed = int(params.get("seed", 7))
        rng = random.Random(seed)
        params["rng"] = rng

    hb = rng.choice(hold_values)
    return {"hold_bars": int(hb)}
