from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from quantbt.io.downloader import (
    download_dukascopy_fx,
    list_available_dukascopy_fx_symbols,
)
from quantbt.io.naming import dataset_filename


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _normalize_symbol(value: str) -> str:
    return value.replace("/", "").replace("_", "").replace("-", "").upper()


def main() -> None:
    available_symbols = list_available_dukascopy_fx_symbols()
    parser = argparse.ArgumentParser(description="Download and store Dukascopy FX OHLC data.")
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="Print available Dukascopy FX symbols and exit.",
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Instrument symbol (e.g., EURUSD). Use --list-symbols to inspect options.",
    )
    parser.add_argument("--timeframe", default="1H", choices=["1H", "4H", "1D"])
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2013-01-01", help="End date (YYYY-MM-DD).")
    parser.add_argument("--save-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--file-ext", default="csv", choices=["csv", "parquet"])
    args = parser.parse_args()

    if args.list_symbols:
        for sym in available_symbols:
            print(sym)
        return

    symbol = _normalize_symbol(args.symbol)
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
    df = download_dukascopy_fx(
        symbol=symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
        save_dir=save_dir,
        file_ext=args.file_ext,
    )

    out_name = dataset_filename(
        symbol=symbol,
        timeframe=args.timeframe,
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
