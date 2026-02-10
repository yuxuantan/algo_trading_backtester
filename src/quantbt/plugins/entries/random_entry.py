from __future__ import annotations

import random
import pandas as pd

from quantbt.plugins.registry import register_entry


@register_entry("random")
def signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    prob = float(params.get("prob", 0.01))
    seed = int(params.get("seed", 7))
    min_bars_between = int(params.get("min_bars_between", 0))
    side = params.get("side", "both")

    rng = random.Random(seed)
    idx = df.index.to_list()

    long_sig = pd.Series(False, index=idx)
    short_sig = pd.Series(False, index=idx)

    last_i = -10**9
    for i in range(len(idx) - 1):
        if i - last_i <= min_bars_between:
            continue
        if rng.random() >= prob:
            continue

        if side == "both":
            long_side = rng.random() < 0.5
        elif side == "long":
            long_side = True
        elif side == "short":
            long_side = False
        else:
            raise ValueError("side must be long/short/both")

        if long_side:
            long_sig.iloc[i] = True
        else:
            short_sig.iloc[i] = True
        last_i = i

    out = pd.DataFrame(index=idx)
    out["long_entry"] = long_sig
    out["short_entry"] = short_sig
    return out
