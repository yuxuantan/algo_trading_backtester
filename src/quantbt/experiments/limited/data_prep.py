from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def load_price_frame(data_path: str | Path, *, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if ts_col not in df.columns:
        raise ValueError(f"ts_col '{ts_col}' not in data")
    df[ts_col] = pd.to_datetime(df[ts_col])
    return df.set_index(ts_col)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def build_signal_frame(
    df: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    atr_series: pd.Series | None,
) -> pd.DataFrame:
    df_sig = df.loc[combined.index].copy()
    df_sig["long_entry"] = combined["long_entry"]
    df_sig["short_entry"] = combined["short_entry"]
    if atr_series is not None:
        df_sig["atr"] = atr_series.reindex(df_sig.index)
    return df_sig


def iter_entries_from_signals(df_sig: pd.DataFrame, *, use_atr: bool):
    idx = df_sig.index.to_list()
    for i in range(len(idx) - 1):
        t = idx[i]
        t_next = idx[i + 1]
        if bool(df_sig.at[t, "long_entry"]):
            e = {
                "entry_i": i + 1,
                "entry_time": t_next,
                "side": "long",
                "entry_open": float(df_sig.at[t_next, "open"]),
                "prev_low": float(df_sig.at[t, "low"]),
                "prev_high": float(df_sig.at[t, "high"]),
            }
        elif bool(df_sig.at[t, "short_entry"]):
            e = {
                "entry_i": i + 1,
                "entry_time": t_next,
                "side": "short",
                "entry_open": float(df_sig.at[t_next, "open"]),
                "prev_low": float(df_sig.at[t, "low"]),
                "prev_high": float(df_sig.at[t, "high"]),
            }
        else:
            continue

        if use_atr:
            atr = float(df_sig.at[t_next, "atr"])
            if not math.isfinite(atr) or atr <= 0:
                continue
            e["atr"] = atr

        yield e
