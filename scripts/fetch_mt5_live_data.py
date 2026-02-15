from __future__ import annotations

import argparse
from datetime import datetime
import re
from pathlib import Path
from typing import Any

import pandas as pd

from quantbt.io.datasets import compute_dataset_meta_from_df, write_dataset_meta
from quantbt.io.naming import dataset_filename


class MT5Client:
    def __init__(self, *, backend: str, api: Any):
        self.backend = backend
        self.api = api

    def __getattr__(self, name: str) -> Any:
        return getattr(self.api, name)

    def initialize(self, **kwargs) -> bool:
        if self.backend == "native":
            return bool(self.api.initialize(**kwargs))

        # siliconmetatrader5 usually connects on object construction.
        init_fn = getattr(self.api, "initialize", None)
        if callable(init_fn):
            try:
                return bool(init_fn())
            except TypeError:
                return bool(init_fn)
        return True

    def shutdown(self) -> None:
        shut_fn = getattr(self.api, "shutdown", None)
        if callable(shut_fn):
            shut_fn()


def _last_error(mt5: MT5Client) -> tuple[int, str]:
    fn = getattr(mt5, "last_error", None)
    if callable(fn):
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) == 2:
                return int(out[0]), str(out[1])
            return -1, str(out)
        except Exception as e:
            return -1, str(e)
    return -1, "last_error unavailable"


def _load_mt5(provider: str, *, host: str, port: int) -> MT5Client:
    provider = str(provider).lower().strip()
    native_err: Exception | None = None
    silicon_err: Exception | None = None

    if provider in {"auto", "native"}:
        try:
            import MetaTrader5 as mt5_mod  # type: ignore

            return MT5Client(backend="native", api=mt5_mod)
        except Exception as e:
            native_err = e
            if provider == "native":
                raise SystemExit(
                    "Failed to import MetaTrader5 native package.\n"
                    "On Windows install with:\n"
                    "  pip install MetaTrader5\n"
                    "  pip install -e '.[mt5]'\n"
                    f"Import error: {e}"
                ) from e

    if provider in {"auto", "silicon"}:
        try:
            from siliconmetatrader5 import MetaTrader5 as SiliconMT5  # type: ignore

            return MT5Client(backend="silicon", api=SiliconMT5(host=host, port=port))
        except Exception as e:
            silicon_err = e
            if provider == "silicon":
                raise SystemExit(
                    "Failed to import/use siliconmetatrader5.\n"
                    "Install on macOS with:\n"
                    "  pip install siliconmetatrader5\n"
                    "  pip install -e '.[mt5-mac]'\n"
                    f"Import/runtime error: {e}"
                ) from e

    raise SystemExit(
        "Could not load any MT5 backend.\n"
        f"Native MetaTrader5 error: {native_err}\n"
        f"siliconmetatrader5 error: {silicon_err}\n"
        "Install one backend:\n"
        "  Windows: pip install -e '.[mt5]'\n"
        "  macOS (Apple Silicon): pip install -e '.[mt5-mac]'"
    )


def _build_timeframe_map(mt5: MT5Client) -> dict[str, int]:
    # Keep a conservative set of widely available MT5 timeframes.
    labels = [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M10",
        "M12",
        "M15",
        "M20",
        "M30",
        "H1",
        "H2",
        "H3",
        "H4",
        "H6",
        "H8",
        "H12",
        "D1",
        "W1",
        "MN1",
    ]
    out: dict[str, int] = {}
    for label in labels:
        attr = f"TIMEFRAME_{label}"
        value = getattr(mt5, attr, None)
        if value is not None:
            out[label] = int(value)
    if not out:
        raise SystemExit("No MT5 timeframes discovered from selected backend.")
    return out


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("_", "").replace("-", "").upper()


def _as_utc_timestamp_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="s", utc=True, errors="coerce")


def _parse_utc_datetime(value: str, *, field_name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as e:
        raise SystemExit(
            f"Invalid {field_name}: {value!r}. "
            "Use formats like YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ."
        ) from e
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _normalize_rates_payload(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise SystemExit("Empty MT5 payload.")
    if "time" not in raw.columns:
        raise SystemExit("MT5 rates payload missing 'time' column.")

    raw = raw.copy()
    raw["timestamp"] = _as_utc_timestamp_col(raw["time"])
    raw = raw.dropna(subset=["timestamp"])
    raw = raw.drop_duplicates(subset=["timestamp"], keep="first")
    raw = raw.set_index("timestamp").sort_index()

    out = raw[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]].copy()
    out = out.rename(columns={"tick_volume": "volume"})
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["spread"] = pd.to_numeric(out["spread"], errors="coerce")
    out["real_volume"] = pd.to_numeric(out["real_volume"], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        raise SystemExit("All fetched bars were invalid after normalization.")
    return out


def _request_rates_chunk(
    mt5: MT5Client,
    *,
    symbol: str,
    timeframe_mt5: int,
    pos: int,
    req: int,
    request_cap: int,
    progress_every: int,
    chunk_idx: int,
) -> tuple[pd.DataFrame | None, int, tuple[int, str]]:
    first_req = min(int(req), int(request_cap))
    candidates = [first_req] + [x for x in (5000, 2000, 1000, 500, 200, 100, 50, 10) if x < first_req]
    last_err: tuple[int, str] = (-1, "unknown")
    if progress_every > 0 and req > request_cap:
        print(
            f"Chunk {chunk_idx}: requested count {req} capped to {request_cap} per MT5 call.",
            flush=True,
        )
    for c in candidates:
        if progress_every > 0:
            print(
                f"Chunk {chunk_idx}: requesting pos={pos}, count={c}...",
                flush=True,
            )
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, pos, c)
        if rates is not None and len(rates) > 0:
            if progress_every > 0:
                print(
                    f"Chunk {chunk_idx}: received {len(rates)} rows (requested {c}).",
                    flush=True,
                )
            return pd.DataFrame(rates), c, last_err
        last_err = _last_error(mt5)
        if progress_every > 0:
            code, msg = last_err
            print(
                f"Chunk {chunk_idx}: request failed for count={c} last_error=({code}) {msg}",
                flush=True,
            )
    return None, 0, last_err


_UNIT_SECONDS = {"M": 60, "H": 3600, "D": 86400, "W": 7 * 86400}


def _timeframe_to_timedelta_mt5(timeframe: str) -> pd.Timedelta | None:
    tf = timeframe.upper().strip()
    if tf.startswith("MN"):
        return None  # month length is variable
    m = re.fullmatch(r"([MHDW])(\d+)", tf)
    if not m:
        return None
    unit = m.group(1)
    n = int(m.group(2))
    return pd.Timedelta(seconds=n * _UNIT_SECONDS[unit])


def _to_index_tz(ts: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp:
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
        return pd.Timedelta(days=3)
    if step == pd.Timedelta(days=1):
        return pd.Timedelta(days=7)
    return pd.Timedelta(days=30)


def _format_missing_ranges(
    ranges: list[tuple[pd.Timestamp, pd.Timestamp, int]],
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


def _validate_requested_coverage_mt5(
    *,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    step = _timeframe_to_timedelta_mt5(timeframe)
    if step is None:
        return

    idx = pd.DatetimeIndex(df.index).sort_values().unique()
    if idx.empty:
        raise ValueError("Fetched MT5 dataset has no timestamps.")

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

    if coverage >= 0.98 and not significant_ranges:
        return

    detail = _format_missing_ranges(significant_ranges or missing_ranges)
    raise ValueError(
        "Incomplete MT5 dataset coverage for "
        f"{symbol} {timeframe} in [{req_start.isoformat()} .. {req_end.isoformat()}). "
        f"Expected {expected_count} bars, got {actual_count} (coverage={coverage:.2%}). "
        f"Missing ranges: {detail}"
    )


def _connect_mt5(mt5: MT5Client, args) -> None:
    init_kwargs: dict = {}
    if mt5.backend == "native" and args.terminal_path:
        init_kwargs["path"] = args.terminal_path
    if mt5.backend == "native" and args.login is not None:
        init_kwargs["login"] = int(args.login)
    if mt5.backend == "native" and args.password:
        init_kwargs["password"] = args.password
    if mt5.backend == "native" and args.server:
        init_kwargs["server"] = args.server

    ok = mt5.initialize(**init_kwargs)
    if not ok:
        code, msg = _last_error(mt5)
        raise SystemExit(f"MT5 initialize failed ({mt5.backend}): ({code}) {msg}")

    # Some backends support explicit login after initialize/attach.
    if args.login is not None:
        login_fn = getattr(mt5, "login", None)
        if callable(login_fn):
            kwargs: dict[str, Any] = {"login": int(args.login)}
            if args.password:
                kwargs["password"] = args.password
            if args.server:
                kwargs["server"] = args.server
            try:
                logged = bool(login_fn(**kwargs))
            except TypeError:
                # Fallback positional invocation for wrappers with different signatures.
                logged = bool(login_fn(int(args.login), args.password or None, args.server or None))
            if not logged:
                code, msg = _last_error(mt5)
                raise SystemExit(f"MT5 login failed ({mt5.backend}): ({code}) {msg}")


def _fetch_rates_by_bars(
    mt5: MT5Client,
    *,
    symbol: str,
    timeframe_mt5: int,
    bars: int,
    batch_size: int,
    progress_every: int,
    request_cap: int,
) -> pd.DataFrame:
    remaining = int(bars)
    pos = 0
    raw_parts: list[pd.DataFrame] = []
    last_err: tuple[int, str] = (-1, "unknown")
    oldest_fetched: pd.Timestamp | None = None
    newest_fetched: pd.Timestamp | None = None
    batch_idx = 0

    while remaining > 0:
        batch_idx += 1
        req = min(int(batch_size), remaining)
        part, used_count, last_err = _request_rates_chunk(
            mt5,
            symbol=symbol,
            timeframe_mt5=timeframe_mt5,
            pos=pos,
            req=req,
            request_cap=request_cap,
            progress_every=progress_every,
            chunk_idx=batch_idx,
        )
        if part is None:
            if pos == 0:
                code, msg = last_err
                raise SystemExit(
                    f"MT5 returned no rates for {symbol}. last_error=({code}) {msg}. "
                    "Try smaller --bars (e.g. 500) or verify exact broker symbol name (e.g. EURUSD, EURUSD., EURUSDm)."
                )
            break

        raw_parts.append(part)
        got_n = len(part)
        pos += got_n
        remaining -= got_n
        # Do not stop on short chunks; some bridges impose per-call caps and still
        # allow further paging with larger pos offsets.

        chunk_ts = _as_utc_timestamp_col(part["time"])
        if not chunk_ts.empty:
            chunk_oldest = chunk_ts.min()
            chunk_newest = chunk_ts.max()
            if oldest_fetched is None or chunk_oldest < oldest_fetched:
                oldest_fetched = chunk_oldest
            if newest_fetched is None or chunk_newest > newest_fetched:
                newest_fetched = chunk_newest

        fetched_rows = int(bars) - remaining
        should_report = progress_every > 0 and ((batch_idx % progress_every == 0) or remaining <= 0)
        if should_report:
            pct = (100.0 * fetched_rows / int(bars)) if int(bars) > 0 else 100.0
            print(
                f"Fetch progress (bars mode): batches={batch_idx}, rows={fetched_rows}/{int(bars)} ({pct:.2f}%), "
                f"oldest={oldest_fetched if oldest_fetched is not None else 'n/a'}, "
                f"newest={newest_fetched if newest_fetched is not None else 'n/a'}",
                flush=True,
            )

    if not raw_parts:
        code, msg = last_err
        raise SystemExit(f"MT5 returned no rates for {symbol}. last_error=({code}) {msg}")
    raw = pd.concat(raw_parts, ignore_index=True)
    return _normalize_rates_payload(raw)


def _fetch_rates_by_date_range(
    mt5: MT5Client,
    *,
    symbol: str,
    timeframe_mt5: int,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    batch_size: int,
    max_backfill_batches: int,
    progress_every: int,
    request_cap: int,
) -> pd.DataFrame:
    if start_utc >= end_utc:
        raise SystemExit("--start must be earlier than --end")

    pos = 0
    raw_parts: list[pd.DataFrame] = []
    last_err: tuple[int, str] = (-1, "unknown")
    reached_start = False
    oldest_fetched: pd.Timestamp | None = None

    for i in range(int(max_backfill_batches)):
        part, used_count, last_err = _request_rates_chunk(
            mt5,
            symbol=symbol,
            timeframe_mt5=timeframe_mt5,
            pos=pos,
            req=int(batch_size),
            request_cap=request_cap,
            progress_every=progress_every,
            chunk_idx=i + 1,
        )
        if part is None:
            if pos == 0:
                code, msg = last_err
                raise SystemExit(
                    f"MT5 returned no rates for {symbol}. last_error=({code}) {msg}. "
                    "Verify exact broker symbol name and MT5 login state."
                )
            break

        raw_parts.append(part)
        got_n = len(part)
        pos += got_n

        chunk_ts = _as_utc_timestamp_col(part["time"])
        oldest_in_chunk = chunk_ts.min() if not chunk_ts.empty else pd.NaT
        if pd.notna(oldest_in_chunk) and oldest_in_chunk <= start_utc:
            reached_start = True
            break

        if pd.notna(oldest_in_chunk):
            if oldest_fetched is None or oldest_in_chunk < oldest_fetched:
                oldest_fetched = oldest_in_chunk

        if progress_every > 0 and ((i + 1) % progress_every == 0):
            approx_rows = sum(len(x) for x in raw_parts)
            if oldest_fetched is None:
                coverage_pct = 0.0
            else:
                total_span = (end_utc - start_utc).total_seconds()
                covered_span = (end_utc - oldest_fetched).total_seconds()
                coverage_pct = 100.0 if total_span <= 0 else max(0.0, min(100.0, 100.0 * covered_span / total_span))
            print(
                f"Backfill progress: batches={i + 1}, rows={approx_rows}, "
                f"oldest={oldest_fetched if oldest_fetched is not None else 'n/a'}, "
                f"time-coverage~{coverage_pct:.2f}%"
                ,
                flush=True,
            )

    if not raw_parts:
        code, msg = last_err
        raise SystemExit(f"MT5 returned no rates for {symbol}. last_error=({code}) {msg}")

    raw = pd.concat(raw_parts, ignore_index=True)
    all_rates = _normalize_rates_payload(raw)
    df = all_rates[(all_rates.index >= start_utc) & (all_rates.index < end_utc)].copy()
    if df.empty:
        avail_start = all_rates.index.min()
        avail_end = all_rates.index.max()
        if avail_start >= end_utc:
            reason = (
                f"Available history starts at {avail_start.isoformat()}, "
                f"which is after requested end {end_utc.isoformat()}."
            )
        elif avail_end < start_utc:
            reason = (
                f"Available history ends at {avail_end.isoformat()}, "
                f"which is before requested start {start_utc.isoformat()}."
            )
        else:
            reason = (
                f"Available fetched range is [{avail_start.isoformat()} .. {avail_end.isoformat()}], "
                "but no bars matched requested window boundaries."
            )
        raise SystemExit(
            f"No bars available in requested window [{start_utc.isoformat()} .. {end_utc.isoformat()}) "
            f"for {symbol}. {reason}"
        )

    if not reached_start:
        oldest = all_rates.index.min()
        print(
            "WARNING: Could not backfill fully to requested start. "
            f"Earliest fetched bar is {oldest}. "
            f"Try increasing --max-backfill-batches (currently {max_backfill_batches}) "
            "and ensure MT5 history depth is expanded in terminal settings. "
            "Completeness validation will determine pass/fail."
        )
    return df


def _save_output(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    save_dir: Path,
    file_ext: str,
    extra: dict,
    requested_start: datetime | None = None,
    requested_end: datetime | None = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    start_ts = requested_start or df.index.min().to_pydatetime()
    end_ts = requested_end or df.index.max().to_pydatetime()
    source = "mt5_ftmo_python"

    out_name = dataset_filename(
        symbol=symbol,
        timeframe=timeframe,
        start=start_ts,
        end=end_ts,
        source=source,
        ext=file_ext,
    )
    out_path = save_dir / out_name
    if file_ext == "csv":
        df.to_csv(out_path, index_label="timestamp")
    else:
        df.to_parquet(out_path)

    meta = compute_dataset_meta_from_df(
        csv_path=out_path,
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        extra=extra,
    )
    write_dataset_meta(out_path, meta)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="POC: fetch live/historical bars from MT5 terminal (e.g., FTMO account).")
    parser.add_argument("--list-timeframes", action="store_true", help="List supported MT5 timeframe labels and exit.")
    parser.add_argument("--list-symbols", action="store_true", help="List symbols exposed by connected MT5 terminal and exit.")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "native", "silicon"],
        help="MT5 backend: native MetaTrader5 (Windows), siliconmetatrader5 (macOS), or auto-detect.",
    )
    parser.add_argument("--host", default="localhost", help="siliconmetatrader5 host (default: localhost).")
    parser.add_argument("--port", type=int, default=8001, help="siliconmetatrader5 port (default: 8001).")
    parser.add_argument("--symbol", default="EURUSD", help="Instrument symbol. Example: EURUSD")
    parser.add_argument("--timeframe", default="M5", help="MT5 timeframe label, e.g. M1, M5, H1, D1.")
    parser.add_argument(
        "--start",
        default="",
        help="UTC start datetime/date (inclusive), e.g. 2025-01-01 or 2025-01-01T00:00:00Z.",
    )
    parser.add_argument(
        "--end",
        default="",
        help="UTC end datetime/date (exclusive). If omitted with --start, defaults to now UTC.",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=0,
        help="Fallback mode: number of most recent bars to fetch (used only when --start is omitted).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Per-request chunk size for copy_rates_from_pos (smaller is safer on RPC bridges).",
    )
    parser.add_argument(
        "--request-cap",
        type=int,
        default=5000,
        help="Hard cap for single copy_rates_from_pos request size to avoid bridge stalls/invalid params.",
    )
    parser.add_argument(
        "--max-backfill-batches",
        type=int,
        default=15000,
        help="Maximum chunk iterations when backfilling for --start/--end mode.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N fetched batches (set 1 for every batch, 0 to disable).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Do not fail when completeness check finds missing ranges; print warning instead.",
    )
    parser.add_argument("--login", type=int, default=None, help="MT5 account login (optional if terminal already logged in).")
    parser.add_argument("--password", default="", help="MT5 account password (optional).")
    parser.add_argument("--server", default="", help="MT5 server name, e.g. FTMO-ServerX (optional).")
    parser.add_argument("--terminal-path", default="", help="Path to terminal executable (optional).")
    parser.add_argument("--save-dir", default="data/processed", help="Directory for output dataset.")
    parser.add_argument("--file-ext", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--no-save", action="store_true", help="Do not write dataset file; only print summary.")
    args = parser.parse_args()

    mt5 = _load_mt5(args.provider, host=args.host, port=int(args.port))
    timeframe_map = _build_timeframe_map(mt5)

    if args.list_timeframes:
        for tf in sorted(timeframe_map.keys()):
            print(tf)
        mt5.shutdown()
        return

    if args.list_symbols:
        try:
            _connect_mt5(mt5, args)
            symbols = mt5.symbols_get()
            if symbols is None:
                code, msg = _last_error(mt5)
                raise SystemExit(f"symbols_get failed: ({code}) {msg}")
            names = sorted(
                {
                    str(getattr(s, "name", "")).strip().upper()
                    for s in symbols
                    if str(getattr(s, "name", "")).strip()
                }
            )
            for name in names:
                print(name)
        finally:
            mt5.shutdown()
        return

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.request_cap <= 0:
        raise SystemExit("--request-cap must be > 0")
    if args.max_backfill_batches <= 0:
        raise SystemExit("--max-backfill-batches must be > 0")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")

    symbol = _normalize_symbol(args.symbol)
    timeframe = str(args.timeframe).upper()
    if timeframe not in timeframe_map:
        supported = ", ".join(sorted(timeframe_map.keys()))
        raise SystemExit(f"Unsupported --timeframe {timeframe!r}. Supported: {supported}")

    use_date_mode = bool(str(args.start).strip())
    if use_date_mode:
        start_utc = _parse_utc_datetime(args.start, field_name="--start")
        end_utc = (
            _parse_utc_datetime(args.end, field_name="--end")
            if str(args.end).strip()
            else pd.Timestamp.now(tz="UTC")
        )
        if start_utc >= end_utc:
            raise SystemExit("--start must be earlier than --end")
    else:
        start_utc = None
        end_utc = None
        if args.bars <= 0:
            raise SystemExit("Provide --start (and optional --end) or use --bars > 0.")

    try:
        print(f"Connecting MT5 backend={args.provider} host={args.host} port={args.port}...", flush=True)
        _connect_mt5(mt5, args)
        print("MT5 connected.", flush=True)

        print(f"Selecting symbol {symbol}...", flush=True)
        if not mt5.symbol_select(symbol, True):
            code, msg = _last_error(mt5)
            raise SystemExit(f"symbol_select failed for {symbol}: ({code}) {msg}")
        print(f"Symbol {symbol} selected.", flush=True)

        timeframe_mt5 = timeframe_map[timeframe]
        if use_date_mode and start_utc is not None and end_utc is not None:
            print(
                f"Starting date-range fetch: timeframe={timeframe}, window=[{start_utc} .. {end_utc}), "
                f"batch_size={args.batch_size}, request_cap={args.request_cap}, "
                f"max_batches={args.max_backfill_batches}",
                flush=True,
            )
            df = _fetch_rates_by_date_range(
                mt5,
                symbol=symbol,
                timeframe_mt5=timeframe_mt5,
                start_utc=start_utc,
                end_utc=end_utc,
                batch_size=int(args.batch_size),
                max_backfill_batches=int(args.max_backfill_batches),
                progress_every=int(args.progress_every),
                request_cap=int(args.request_cap),
            )
            try:
                _validate_requested_coverage_mt5(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_utc,
                    end=end_utc,
                )
            except ValueError as e:
                if args.allow_incomplete:
                    print(f"WARNING: {e}")
                else:
                    raise SystemExit(str(e)) from e
        else:
            print(
                f"Starting bars fetch: timeframe={timeframe}, bars={args.bars}, "
                f"batch_size={args.batch_size}, request_cap={args.request_cap}",
                flush=True,
            )
            df = _fetch_rates_by_bars(
                mt5,
                symbol=symbol,
                timeframe_mt5=timeframe_mt5,
                bars=int(args.bars),
                batch_size=int(args.batch_size),
                progress_every=int(args.progress_every),
                request_cap=int(args.request_cap),
            )

        account = mt5.account_info()
        terminal = mt5.terminal_info()
        tick = mt5.symbol_info_tick(symbol)

        print(f"Fetched {len(df)} bars for {symbol} {timeframe}")
        print(f"Backend: {mt5.backend}")
        if use_date_mode and start_utc is not None and end_utc is not None:
            print(f"Requested UTC window: [{start_utc} .. {end_utc})")
        print(f"Range UTC: {df.index.min()} -> {df.index.max()}")
        if account is not None:
            print(f"Account: login={account.login} server={account.server} company={account.company}")
        if terminal is not None:
            print(f"Terminal: {terminal.name} build={terminal.build}")
        if tick is not None:
            tick_ts = datetime.utcfromtimestamp(int(tick.time))
            print(f"Latest tick UTC: {tick_ts} bid={tick.bid} ask={tick.ask}")

        print("\nLast 5 bars:")
        print(df.tail(5).to_string())

        if not args.no_save:
            extra = {
                "provider": "mt5_terminal",
                "backend": mt5.backend,
                "requested_bars": int(args.bars) if not use_date_mode else None,
                "requested_start": start_utc.isoformat() if start_utc is not None else None,
                "requested_end": end_utc.isoformat() if end_utc is not None else None,
                "timeframe": timeframe,
                "symbol": symbol,
                "server": getattr(account, "server", None) if account is not None else None,
                "login": getattr(account, "login", None) if account is not None else None,
                "terminal": getattr(terminal, "name", None) if terminal is not None else None,
                "terminal_build": getattr(terminal, "build", None) if terminal is not None else None,
            }
            out_path = _save_output(
                df,
                symbol=symbol,
                timeframe=timeframe,
                save_dir=Path(args.save_dir),
                file_ext=args.file_ext,
                extra=extra,
                requested_start=start_utc.to_pydatetime() if start_utc is not None else None,
                requested_end=end_utc.to_pydatetime() if end_utc is not None else None,
            )
            print(f"\nSaved dataset: {out_path}")
            print(f"Metadata: {out_path.with_suffix(out_path.suffix + '.meta.json')}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
