from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dk_instruments

from quantbt.io.datasets import compute_dataset_meta_from_df, write_dataset_meta
from quantbt.io.naming import dataset_filename  # should build {symbol}_{tf}_{start}_{end}_{source}.{ext}


_UNIT_SECONDS = {
    "S": 1,
    "M": 60,
    "H": 3600,
    "D": 86400,
    "W": 7 * 86400,
    "MO": 30 * 86400,  # Approximation for sorting only.
}


def _parse_timeframe_key(value: str) -> tuple[int, str] | None:
    m = re.fullmatch(r"(\d+)(MO|S|M|H|D|W)", value)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    return n, unit


def _timeframe_sort_key(value: str):
    parsed = _parse_timeframe_key(value)
    if parsed is None:
        return (1, value)
    n, unit = parsed
    return (0, n * _UNIT_SECONDS[unit], n, unit)


def _interval_name_to_timeframe(attr_name: str) -> str | None:
    if not attr_name.startswith("INTERVAL_"):
        return None
    token = attr_name[len("INTERVAL_") :].upper()

    patterns: list[tuple[str, str]] = [
        (r"^(?:SEC|SECS|SECOND|SECONDS)_(\d+)$", "S"),
        (r"^(\d+)_(?:SEC|SECS|SECOND|SECONDS)$", "S"),
        (r"^(?:MIN|MINS|MINUTE|MINUTES)_(\d+)$", "M"),
        (r"^(\d+)_(?:MIN|MINS|MINUTE|MINUTES)$", "M"),
        (r"^(?:HOUR|HOURS)_(\d+)$", "H"),
        (r"^(\d+)_(?:HOUR|HOURS)$", "H"),
        (r"^(?:DAY|DAYS)_(\d+)$", "D"),
        (r"^(\d+)_(?:DAY|DAYS)$", "D"),
        (r"^(?:WEEK|WEEKS)_(\d+)$", "W"),
        (r"^(\d+)_(?:WEEK|WEEKS)$", "W"),
        (r"^(?:MONTH|MONTHS)_(\d+)$", "MO"),
        (r"^(\d+)_(?:MONTH|MONTHS)$", "MO"),
    ]
    for pattern, suffix in patterns:
        m = re.fullmatch(pattern, token)
        if m:
            return f"{int(m.group(1))}{suffix}"

    singular = {
        "SEC": "1S",
        "SECOND": "1S",
        "MIN": "1M",
        "MINUTE": "1M",
        "HOUR": "1H",
        "DAY": "1D",
        "WEEK": "1W",
        "MONTH": "1MO",
    }
    if token in singular:
        return singular[token]
    return token


def _build_interval_map() -> dict[str, object]:
    mapping: dict[str, object] = {}
    for name, value in vars(dukascopy_python).items():
        if not name.startswith("INTERVAL_"):
            continue
        if value is None or callable(value):
            continue
        key = _interval_name_to_timeframe(name)
        if key is None:
            continue
        mapping.setdefault(key, value)
    if not mapping:
        raise RuntimeError("No supported INTERVAL_* constants found in dukascopy_python")
    return dict(sorted(mapping.items(), key=lambda kv: _timeframe_sort_key(kv[0])))


INTERVAL_MAP = _build_interval_map()


def list_supported_dukascopy_timeframes() -> list[str]:
    return list(INTERVAL_MAP.keys())


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta | None:
    parsed = _parse_timeframe_key(timeframe.upper())
    if parsed is None:
        return None
    n, unit = parsed
    if unit == "MO":
        return None  # Month length is variable; skip strict completeness checks.
    seconds = n * _UNIT_SECONDS[unit]
    return pd.Timedelta(seconds=seconds)


def _to_index_tz(ts: datetime | pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if idx.tz is None:
        if out.tzinfo is not None:
            out = out.tz_convert("UTC").tz_localize(None)
        return out
    if out.tzinfo is None:
        return out.tz_localize(idx.tz)
    return out.tz_convert(idx.tz)


def _align_start_to_reference(start_ts: pd.Timestamp, reference_ts: pd.Timestamp, step: pd.Timedelta) -> pd.Timestamp:
    remainder = (start_ts - reference_ts) % step
    if remainder == pd.Timedelta(0):
        return start_ts
    return start_ts + (step - remainder)


def _slot_of_week_ns(idx: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(
        idx.dayofweek.astype("int64") * 24 * 3600 * 1_000_000_000
        + idx.hour.astype("int64") * 3600 * 1_000_000_000
        + idx.minute.astype("int64") * 60 * 1_000_000_000
        + idx.second.astype("int64") * 1_000_000_000
        + idx.nanosecond.astype("int64")
    )


def _collapse_missing_ranges(
    missing: pd.DatetimeIndex, step: pd.Timedelta
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    if len(missing) == 0:
        return []
    out: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = missing[0]
    prev = missing[0]
    count = 1
    for ts in missing[1:]:
        if (ts - prev) == step:
            prev = ts
            count += 1
            continue
        out.append((start, prev, count))
        start = ts
        prev = ts
        count = 1
    out.append((start, prev, count))
    return out


def _significant_gap_threshold(step: pd.Timedelta) -> pd.Timedelta:
    if step < pd.Timedelta(days=1):
        return pd.Timedelta(days=3)   # ignore normal weekend closure / short holidays
    if step == pd.Timedelta(days=1):
        return pd.Timedelta(days=7)
    return pd.Timedelta(days=30)


def _format_missing_ranges(
    ranges: list[tuple[pd.Timestamp, pd.Timestamp, int]],
    step: pd.Timedelta,
    *,
    max_items: int = 8,
) -> str:
    if not ranges:
        return ""
    parts: list[str] = []
    for start, end, count in ranges[:max_items]:
        if start == end:
            parts.append(f"{start.isoformat()} ({count} bar)")
        else:
            parts.append(f"{start.isoformat()} -> {end.isoformat()} ({count} bars)")
    if len(ranges) > max_items:
        parts.append(f"... +{len(ranges) - max_items} more ranges")
    return "; ".join(parts)


def _validate_requested_coverage(
    *,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> None:
    step = _timeframe_to_timedelta(timeframe)
    if step is None:
        return
    idx = pd.DatetimeIndex(df.index).sort_values().unique()
    if idx.empty:
        raise ValueError("Downloaded dataset has no timestamps.")

    req_start = _to_index_tz(start, idx)
    req_end = _to_index_tz(end, idx)
    if req_start >= req_end:
        raise ValueError(f"Invalid request window: start={req_start}, end={req_end}")

    expected_start = _align_start_to_reference(req_start, idx[0], step)
    if expected_start >= req_end:
        return

    expected = pd.date_range(start=expected_start, end=req_end, freq=step, inclusive="left", tz=idx.tz)
    if expected.empty:
        return

    # Restrict expected timestamps to observed weekly trading slots to avoid weekend/session false positives.
    observed_slots = _slot_of_week_ns(idx)
    expected_slots = _slot_of_week_ns(expected)
    expected = expected[pd.Index(expected_slots).isin(set(observed_slots))]
    if expected.empty:
        return

    actual_window = idx[(idx >= req_start) & (idx < req_end)]
    actual_matching = pd.Index(actual_window).intersection(expected)
    missing = expected.difference(actual_matching)
    if missing.empty:
        return

    expected_count = len(expected)
    actual_count = len(actual_matching)
    coverage = actual_count / expected_count if expected_count else 1.0

    missing_ranges = _collapse_missing_ranges(missing, step)
    gap_threshold = _significant_gap_threshold(step)
    significant_ranges = [
        (s, e, c)
        for s, e, c in missing_ranges
        if (e - s + step) >= gap_threshold
    ]

    # Fail on materially incomplete coverage or long missing spans.
    if coverage >= 0.98 and not significant_ranges:
        return

    detail = _format_missing_ranges(significant_ranges or missing_ranges, step)
    raise ValueError(
        "Incomplete Dukascopy dataset coverage for "
        f"{symbol} {timeframe} in [{req_start.isoformat()} .. {req_end.isoformat()}]. "
        f"Expected {expected_count} bars, got {actual_count} (coverage={coverage:.2%}). "
        f"Missing ranges: {detail}"
    )


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

    timeframe = str(timeframe).upper()
    if timeframe not in INTERVAL_MAP:
        supported = ", ".join(INTERVAL_MAP.keys())
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {supported}")

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

    if df is None or len(df) == 0:
        raise ValueError(
            "No data returned from Dukascopy for "
            f"{symbol} {timeframe} in [{start.date()} .. {end.date()}]. "
            "The requested range/timeframe may be unavailable. "
            "Try a newer date range or a higher timeframe."
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

    if df.empty:
        raise ValueError(
            "Dukascopy returned an empty dataset after normalization for "
            f"{symbol} {timeframe} in [{start.date()} .. {end.date()}]. "
            "The requested range/timeframe may be unavailable."
        )

    _validate_requested_coverage(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )

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
