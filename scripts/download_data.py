from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from quantbt.io.downloader import download_dukascopy_fx
from quantbt.io.naming import dataset_filename


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and store Dukascopy FX OHLC data.")
    parser.add_argument("--symbol", default="EURUSD", help="Instrument symbol. Currently only EURUSD is supported.")
    parser.add_argument("--timeframe", default="1H", choices=["1H", "4H", "1D"])
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2013-01-01", help="End date (YYYY-MM-DD).")
    parser.add_argument("--save-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--file-ext", default="csv", choices=["csv", "parquet"])
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if start >= end:
        raise ValueError("--start must be earlier than --end")

    save_dir = Path(args.save_dir)
    df = download_dukascopy_fx(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
        save_dir=save_dir,
        file_ext=args.file_ext,
    )

    out_name = dataset_filename(
        symbol=args.symbol,
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
