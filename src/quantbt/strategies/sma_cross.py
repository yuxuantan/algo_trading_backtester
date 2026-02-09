from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class SMACrossParams:
    fast: int = 50
    slow: int = 200
    rr: float = 2.0
    sl_buffer_pips: float = 1.0
    pip_size: float = 0.0001

def compute_features(df: pd.DataFrame, p: SMACrossParams) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(p.fast).mean()
    out["sma_slow"] = out["close"].rolling(p.slow).mean()
    out = out.dropna(subset=["sma_fast", "sma_slow"])
    return out

def compute_signals(df_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_feat.copy()
    out["bull_cross"] = (out["sma_fast"] > out["sma_slow"]) & (out["sma_fast"].shift(1) <= out["sma_slow"].shift(1))
    out["bear_cross"] = (out["sma_fast"] < out["sma_slow"]) & (out["sma_fast"].shift(1) >= out["sma_slow"].shift(1))
    return out

def build_brackets_from_signal(
    side: str,
    entry_open: float,
    prev_low: float,
    prev_high: float,
    p: SMACrossParams,
):
    """
    Returns (sl, tp, stop_dist) in price units, or None if invalid.
    side: "long" or "short"
    """
    sl_buffer = p.sl_buffer_pips * p.pip_size

    if side == "long":
        sl = prev_low - sl_buffer
        stop_dist = entry_open - sl
        if stop_dist <= 0:
            return None
        tp = entry_open + p.rr * stop_dist
        return sl, tp, stop_dist

    if side == "short":
        sl = prev_high + sl_buffer
        stop_dist = sl - entry_open
        if stop_dist <= 0:
            return None
        tp = entry_open - p.rr * stop_dist
        return sl, tp, stop_dist

    raise ValueError("side must be 'long' or 'short'")

# ---- Standard interface for universal runner ----
Params = SMACrossParams
