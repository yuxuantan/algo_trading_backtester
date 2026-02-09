from datetime import datetime
from quantbt.io.downloader import download_dukascopy_fx

download_dukascopy_fx(
    symbol="EURUSD",
    timeframe="1H",
    start=datetime(2010, 1, 1),
    end=datetime.now(),
    save_dir="data/processed",
    file_ext="csv",   # or "parquet"
)