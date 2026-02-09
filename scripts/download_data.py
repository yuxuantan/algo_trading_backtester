from datetime import datetime
from quantbt.io.downloader import download_dukascopy_fx

df = download_dukascopy_fx(
    symbol="EURUSD",
    timeframe="1H",
    start=datetime(2010, 1, 1),
    end=datetime.now(),
    save_path="data/processed/eurusd_1h.csv",
)

print(df.head())
print(f"Saved {len(df)} rows to data/processed/eurusd_1h.csv")
