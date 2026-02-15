from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from quantbt.io.naming import dataset_filename


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _normalize_symbol(value: str) -> str:
    return value.replace("/", "").replace("_", "").replace("-", "").upper()


def _run_and_echo(cmd: list[str]) -> int:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return int(proc.wait())


def _load_dukascopy_helpers():
    try:
        from quantbt.io.downloader import (  # pylint: disable=import-outside-toplevel
            download_dukascopy_fx,
            list_available_dukascopy_fx_symbols,
            list_supported_dukascopy_timeframes,
        )
    except Exception as e:
        raise SystemExit(
            "Dukascopy provider is unavailable because dukascopy dependencies are missing.\n"
            "Install with: pip install dukascopy-python\n"
            f"Import error: {e}"
        ) from e
    return download_dukascopy_fx, list_available_dukascopy_fx_symbols, list_supported_dukascopy_timeframes


def _mt5_fetch_script_path() -> Path:
    return Path(__file__).resolve().parent / "fetch_mt5_live_data.py"


def _build_mt5_base_cmd(args: argparse.Namespace) -> list[str]:
    fetch_script = _mt5_fetch_script_path()
    if not fetch_script.exists():
        raise SystemExit(f"Missing script: {fetch_script}")

    cmd = [
        sys.executable,
        str(fetch_script),
        "--provider",
        args.mt5_backend,
        "--host",
        args.mt5_host,
        "--port",
        str(int(args.mt5_port)),
    ]
    if args.mt5_login is not None:
        cmd.extend(["--login", str(int(args.mt5_login))])
    if args.mt5_password:
        cmd.extend(["--password", args.mt5_password])
    if args.mt5_server:
        cmd.extend(["--server", args.mt5_server])
    if args.mt5_terminal_path:
        cmd.extend(["--terminal-path", args.mt5_terminal_path])
    return cmd


def _handle_mt5_list_timeframes(args: argparse.Namespace) -> None:
    cmd = _build_mt5_base_cmd(args) + ["--list-timeframes"]
    rc = _run_and_echo(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _handle_mt5_list_symbols(args: argparse.Namespace) -> None:
    cmd = _build_mt5_base_cmd(args) + ["--list-symbols"]
    rc = _run_and_echo(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _handle_mt5_download(args: argparse.Namespace) -> None:
    symbol = _normalize_symbol(args.symbol)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if start >= end:
        raise ValueError("--start must be earlier than --end")

    cmd = _build_mt5_base_cmd(args) + [
        "--symbol",
        symbol,
        "--timeframe",
        str(args.timeframe).upper(),
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--save-dir",
        str(args.save_dir),
        "--file-ext",
        args.file_ext,
        "--batch-size",
        str(int(args.mt5_batch_size)),
        "--request-cap",
        str(int(args.mt5_request_cap)),
        "--max-backfill-batches",
        str(int(args.mt5_max_backfill_batches)),
        "--progress-every",
        str(int(args.mt5_progress_every)),
    ]
    if args.mt5_allow_incomplete:
        cmd.append("--allow-incomplete")

    rc = _run_and_echo(cmd)
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified market data downloader (Dukascopy or MT5/FTMO).")
    parser.add_argument(
        "--provider",
        default="dukascopy",
        choices=["dukascopy", "mt5_ftmo"],
        help="Data provider backend.",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="Print available symbols and exit.",
    )
    parser.add_argument(
        "--list-timeframes",
        action="store_true",
        help="Print available timeframes and exit.",
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Instrument symbol (e.g., EURUSD). Use --list-symbols to inspect options.",
    )
    parser.add_argument(
        "--timeframe",
        default="1H",
        type=str.upper,
        help="Timeframe label. Dukascopy examples: 1H/5M. MT5 examples: M5/H1.",
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2013-01-01", help="End date (YYYY-MM-DD).")
    parser.add_argument("--save-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--file-ext", default="csv", choices=["csv", "parquet"])

    # MT5-specific options (used when --provider mt5_ftmo)
    parser.add_argument("--mt5-backend", default="auto", choices=["auto", "native", "silicon"])
    parser.add_argument("--mt5-host", default="localhost")
    parser.add_argument("--mt5-port", type=int, default=8001)
    parser.add_argument("--mt5-login", type=int, default=None)
    parser.add_argument("--mt5-password", default="")
    parser.add_argument("--mt5-server", default="")
    parser.add_argument("--mt5-terminal-path", default="")
    parser.add_argument("--mt5-batch-size", type=int, default=1000)
    parser.add_argument("--mt5-request-cap", type=int, default=5000)
    parser.add_argument("--mt5-max-backfill-batches", type=int, default=15000)
    parser.add_argument("--mt5-progress-every", type=int, default=25)
    parser.add_argument("--mt5-allow-incomplete", action="store_true")
    args = parser.parse_args()

    provider = str(args.provider).strip().lower()

    if args.list_symbols:
        if provider == "dukascopy":
            _, list_available_dukascopy_fx_symbols, _ = _load_dukascopy_helpers()
            available_symbols = list_available_dukascopy_fx_symbols()
            for sym in available_symbols:
                print(sym)
            return
        _handle_mt5_list_symbols(args)
        return

    if args.list_timeframes:
        if provider == "dukascopy":
            _, _, list_supported_dukascopy_timeframes = _load_dukascopy_helpers()
            timeframe_choices = list_supported_dukascopy_timeframes()
            for tf in timeframe_choices:
                print(tf)
            return
        _handle_mt5_list_timeframes(args)
        return

    if provider == "mt5_ftmo":
        _handle_mt5_download(args)
        return

    download_dukascopy_fx, list_available_dukascopy_fx_symbols, list_supported_dukascopy_timeframes = _load_dukascopy_helpers()
    available_symbols = list_available_dukascopy_fx_symbols()
    timeframe_choices = list_supported_dukascopy_timeframes()

    symbol = _normalize_symbol(args.symbol)
    dukascopy_tf = str(args.timeframe).upper()
    if dukascopy_tf not in set(timeframe_choices):
        supported = ", ".join(timeframe_choices)
        raise ValueError(f"Unsupported Dukascopy timeframe: {dukascopy_tf}. Supported: {supported}")

    if symbol not in set(available_symbols):
        preview = ", ".join(available_symbols[:20])
        raise ValueError(
            f"--symbol {args.symbol!r} is not available. "
            f"Known symbols ({len(available_symbols)} total, first 20): {preview}"
        )

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if start >= end:
        raise ValueError("--start must be earlier than --end")

    save_dir = Path(args.save_dir)
    try:
        df = download_dukascopy_fx(
            symbol=symbol,
            timeframe=dukascopy_tf,
            start=start,
            end=end,
            save_dir=save_dir,
            file_ext=args.file_ext,
        )
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from e

    out_name = dataset_filename(
        symbol=symbol,
        timeframe=dukascopy_tf,
        start=start,
        end=end,
        source="dukascopy_python",
        ext=args.file_ext,
    )
    out_path = save_dir / out_name
    print(f"Saved {len(df)} rows to: {out_path}")
    print(f"Metadata: {out_path.with_suffix(out_path.suffix + '.meta.json')}")


if __name__ == "__main__":
    main()
