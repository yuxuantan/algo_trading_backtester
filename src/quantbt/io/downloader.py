from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dk_instruments

from quantbt.io.datasets import compute_dataset_meta_from_df, write_dataset_meta
from quantbt.io.naming import dataset_filename  # should build {symbol}_{tf}_{start}_{end}_{source}.{ext}

INTERVAL_MAP = {
    "1H": dukascopy_python.INTERVAL_HOUR_1,
    "4H": dukascopy_python.INTERVAL_HOUR_4,
    "1D": dukascopy_python.INTERVAL_DAY_1,
}


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("_", "").replace("-", "").upper()


@lru_cache(maxsize=1)
def _fx_symbol_to_instrument() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, value in vars(dk_instruments).items():
        if not name.startswith("INSTRUMENT_FX_"):
            continue
        if not isinstance(value, str):
            continue
        symbol = _normalize_symbol(value)
        # Keep first if duplicates appear.
        mapping.setdefault(symbol, value)
    if not mapping:
        raise RuntimeError("No INSTRUMENT_FX_* symbols found in dukascopy_python.instruments")
    return dict(sorted(mapping.items()))


def list_available_dukascopy_fx_symbols() -> list[str]:
    return list(_fx_symbol_to_instrument().keys())


def resolve_dukascopy_fx_instrument(symbol: str) -> str:
    symbol_norm = _normalize_symbol(symbol)
    mapping = _fx_symbol_to_instrument()
    instrument = mapping.get(symbol_norm)
    if instrument is None:
        example = ", ".join(list(mapping.keys())[:20])
        raise ValueError(
            f"Unsupported Dukascopy FX symbol: {symbol}. "
            f"Use one of {len(mapping)} symbols (first 20): {example}"
        )
    return instrument

def _resolve_time_col(df: pd.DataFrame) -> str:
    candidates = [
        "timestamp", "time", "datetime", "date", "Date", "Time",
        "Timestamp", "DATE", "DATETIME"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    if df.index.name and df.index.name.lower() in {"time", "date", "datetime", "timestamp"}:
        return "__index__"
    raise KeyError(f"Could not find a timestamp column. Columns: {list(df.columns)} | index.name={df.index.name}")

def _resolve_ohlc_cols(df: pd.DataFrame) -> dict:
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
    save_dir: str | Path | None = None,  # 👈 NEW (preferred)
    file_ext: str = "csv",               # "csv" (default) or "parquet"
) -> pd.DataFrame:
    """
    If save_path is provided, writes exactly there.
    Else if save_dir is provided, auto-names the file using dataset_filename(...).
    """

    if timeframe not in INTERVAL_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if save_path is not None and save_dir is not None:
        raise ValueError("Provide only one of save_path or save_dir (not both).")

    symbol = _normalize_symbol(symbol)
    instrument = resolve_dukascopy_fx_instrument(symbol)
    interval = INTERVAL_MAP[timeframe]

    df = dukascopy_python.fetch(
        instrument,
        interval,
        offer_side,
        start,
        end,
    )

    # ---- normalize schema robustly ----
    time_col = _resolve_time_col(df)

    df = df.copy()
    if time_col == "__index__":
        df["timestamp"] = pd.to_datetime(df.index)
    else:
        df = df.rename(columns={time_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    ohlc_map = _resolve_ohlc_cols(df)
    df = df.rename(columns=ohlc_map)

    keep = ["timestamp", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    df = df[keep].set_index("timestamp").sort_index()

    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # ---- determine output path (auto naming) ----
    out_path: Path | None = None
    if save_path is not None:
        out_path = Path(save_path)
    elif save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fname = dataset_filename(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            source="dukascopy_python",
            ext=file_ext,
        )
        out_path = save_dir / fname

    # ---- write file + meta if requested ----
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if file_ext.lower() == "csv":
            df.to_csv(out_path)
        elif file_ext.lower() == "parquet":
            df.to_parquet(out_path)
        else:
            raise ValueError("file_ext must be 'csv' or 'parquet'")

        meta = compute_dataset_meta_from_df(
            csv_path=out_path,   # name kept for backward compat in your helper
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source="dukascopy_python",
            extra={
                "offer_side": str(offer_side),
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
            }
        )
        write_dataset_meta(out_path, meta)

    return df
