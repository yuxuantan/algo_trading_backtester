from __future__ import annotations

import pandas as pd


def prepare_ohlc_features(
    df: pd.DataFrame,
    *,
    required_columns: tuple[str, ...] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    out = df.copy().sort_index()
    for column in required_columns:
        if column not in out.columns:
            raise ValueError(f"Missing required OHLC column: {column}")
        out[column] = out[column].astype(float)
    return out


def inert_cross_signals(df_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_feat.copy()
    out["bull_cross"] = False
    out["bear_cross"] = False
    return out


def prev_bar_rr_brackets(
    *,
    side: str,
    entry_open: float,
    prev_low: float,
    prev_high: float,
    sl_buffer_pips: float,
    pip_size: float,
    rr: float,
):
    sl_buffer = float(sl_buffer_pips) * float(pip_size)
    rr = max(float(rr), 0.1)

    if side == "long":
        sl = float(prev_low) - sl_buffer
        stop_dist = float(entry_open) - sl
        if stop_dist <= 0:
            return None
        tp = float(entry_open) + rr * stop_dist
        return sl, tp, stop_dist

    if side == "short":
        sl = float(prev_high) + sl_buffer
        stop_dist = sl - float(entry_open)
        if stop_dist <= 0:
            return None
        tp = float(entry_open) - rr * stop_dist
        return sl, tp, stop_dist

    raise ValueError("side must be 'long' or 'short'")
