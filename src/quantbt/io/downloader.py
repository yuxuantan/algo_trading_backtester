from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD
from quantbt.io.datasets import compute_dataset_meta_from_df, write_dataset_meta

INTERVAL_MAP = {
    "1H": dukascopy_python.INTERVAL_HOUR_1,
    "4H": dukascopy_python.INTERVAL_HOUR_4,
    "1D": dukascopy_python.INTERVAL_DAY_1,
}

def _resolve_time_col(df: pd.DataFrame) -> str:
    # common candidates from various libs
    candidates = [
        "timestamp", "time", "datetime", "date", "Date", "Time",
        "Timestamp", "DATE", "DATETIME"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # sometimes time is in index already
    if df.index.name and df.index.name.lower() in {"time", "date", "datetime", "timestamp"}:
        return "__index__"
    raise KeyError(f"Could not find a timestamp column. Columns: {list(df.columns)} | index.name={df.index.name}")

def _resolve_ohlc_cols(df: pd.DataFrame) -> dict:
    # map many possible names -> standard
    mapping = {}
    col_map = {
        "open":  ["open", "Open", "OPEN", "o", "O"],
        "high":  ["high", "High", "HIGH", "h", "H"],
        "low":   ["low", "Low", "LOW", "l", "L"],
        "close": ["close", "Close", "CLOSE", "c", "C"],
        "volume":["volume", "Volume", "VOLUME", "vol", "Vol"],
    }
    for std, options in col_map.items():
        for c in options:
            if c in df.columns:
                mapping[c] = std
                break
    # require OHLC at minimum
    required = {"open", "high", "low", "close"}
    have = set(mapping.values())
    if not required.issubset(have):
        raise KeyError(f"Missing OHLC columns. Found columns: {list(df.columns)} | mapped: {mapping}")
    return mapping

def download_dukascopy_fx(
    *,
    symbol: str = "EURUSD",
    timeframe: str = "1H",
    start: datetime,
    end: datetime,
    offer_side=dukascopy_python.OFFER_SIDE_BID,
    save_path: str | Path | None = None,
) -> pd.DataFrame:

    if symbol != "EURUSD":
        raise NotImplementedError("Only EURUSD wired for now")

    if timeframe not in INTERVAL_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    interval = INTERVAL_MAP[timeframe]

    df = dukascopy_python.fetch(
        INSTRUMENT_FX_MAJORS_EUR_USD,
        interval,
        offer_side,
        start,
        end,
    )

    # ---- normalize schema robustly ----
    time_col = _resolve_time_col(df)

    if time_col == "__index__":
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df.index)
    else:
        df = df.copy()
        df = df.rename(columns={time_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    ohlc_map = _resolve_ohlc_cols(df)
    df = df.rename(columns=ohlc_map)

    # standard output: timestamp index + open/high/low/close (+ volume if present)
    keep = ["timestamp", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    df = df[keep].set_index("timestamp").sort_index()

    # enforce numeric types
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)

        meta = compute_dataset_meta_from_df(
            csv_path=save_path,
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source="dukascopy_python",
            extra={"offer_side": str(offer_side)}
        )
        write_dataset_meta(save_path, meta)


    return df
