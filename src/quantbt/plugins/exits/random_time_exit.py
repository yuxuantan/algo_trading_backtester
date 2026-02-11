from __future__ import annotations

import math
import random

from quantbt.plugins.registry import register_exit


def _entry_rng(*, seed: int, entry: dict | None) -> random.Random:
    if entry is None:
        return random.Random(seed)
    entry_i = int(entry.get("entry_i", 0))
    mixed_seed = (int(seed) * 1_000_003) ^ entry_i
    return random.Random(mixed_seed)


def _sample_hold_bars_from_avg(*, avg_hold_bars: float, rng: random.Random) -> int:
    if not math.isfinite(avg_hold_bars) or avg_hold_bars <= 0:
        raise ValueError("avg_hold_bars must be a positive finite number")
    low = max(1, int(math.floor(avg_hold_bars)))
    high = max(low, int(math.ceil(avg_hold_bars)))
    if low == high:
        return low
    p_high = avg_hold_bars - low
    return high if rng.random() < p_high else low


@register_exit("monkey_exit")
@register_exit("random_time_exit")
def build_exit(side: str, entry_open: float, prev_low: float, prev_high: float, params: dict, entry=None):
    seed = int(params.get("seed", 7))
    rng = _entry_rng(seed=seed, entry=entry)

    hold_values = params.get("hold_bars_values")
    if hold_values:
        vals = [int(v) for v in hold_values]
        vals = [v for v in vals if v > 0]
        if not vals:
            raise ValueError("hold_bars_values must contain at least one positive integer")
        hb = rng.choice(vals)
        return {"hold_bars": int(hb)}

    avg_hold_bars = params.get("avg_hold_bars")
    if avg_hold_bars is not None:
        hb = _sample_hold_bars_from_avg(avg_hold_bars=float(avg_hold_bars), rng=rng)
        min_hb = int(params.get("min_hold_bars", 1))
        max_hb_raw = params.get("max_hold_bars")
        if max_hb_raw is not None:
            hb = min(hb, int(max_hb_raw))
        hb = max(hb, min_hb)
        if hb <= 0:
            raise ValueError("effective hold_bars must be > 0")
        return {"hold_bars": int(hb)}

    hold_bars = params.get("hold_bars")
    if hold_bars is not None:
        hb = int(hold_bars)
        if hb <= 0:
            raise ValueError("hold_bars must be > 0")
        return {"hold_bars": hb}

    raise ValueError("random_time_exit/monkey_exit requires one of: hold_bars_values, avg_hold_bars, hold_bars")


def _validate(params: dict) -> bool:
    try:
        if "hold_bars_values" in params and params.get("hold_bars_values"):
            vals = [int(v) for v in params["hold_bars_values"]]
            if not vals or any(v <= 0 for v in vals):
                return False
            return True

        if "avg_hold_bars" in params and params.get("avg_hold_bars") is not None:
            avg = float(params["avg_hold_bars"])
            if not math.isfinite(avg) or avg <= 0:
                return False
            if "min_hold_bars" in params and int(params["min_hold_bars"]) <= 0:
                return False
            if "max_hold_bars" in params and int(params["max_hold_bars"]) <= 0:
                return False
            return True

        if "hold_bars" in params and params.get("hold_bars") is not None:
            if int(params["hold_bars"]) <= 0:
                return False
            return True

    except Exception:
        return False
    return False


build_exit.validate = _validate
