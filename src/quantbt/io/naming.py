from datetime import datetime
from pathlib import Path

def dataset_filename(
    *,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    source: str,
    ext: str = "parquet",
) -> str:
    sym = symbol.lower()
    tf = timeframe.lower()
    s = start.strftime("%Y%m%d")
    e = end.strftime("%Y%m%d")
    return f"{sym}_{tf}_{s}_{e}_{source}.{ext}"
