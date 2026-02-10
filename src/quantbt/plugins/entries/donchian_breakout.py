from __future__ import annotations

import pandas as pd

from quantbt.plugins.registry import register_entry


@register_entry("donchian_breakout")
def signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    lookback = int(params.get("lookback", 20))
    if lookback <= 0:
        raise ValueError("lookback must be > 0")

    high = df["high"].astype(float)
    low = df["low"].astype(float)

    highest = high.rolling(lookback).max()
    lowest = low.rolling(lookback).min()

    long_sig = high > highest.shift(1)
    short_sig = low < lowest.shift(1)

    out = pd.DataFrame(index=df.index)
    out["long_entry"] = long_sig.fillna(False)
    out["short_entry"] = short_sig.fillna(False)
    return out
