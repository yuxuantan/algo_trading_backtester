import pandas as pd

REQUIRED_COLS = ["open", "high", "low", "close"]

def load_ohlc_csv(
    path: str,
    ts_col: str = "timestamp",
    tz: str | None = None,
) -> pd.DataFrame:
    """
    Loads OHLC CSV into a DataFrame indexed by timestamp.
    Expected cols: timestamp, open, high, low, close (plus any extras).
    """
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found. Columns: {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col])
    if tz is not None:
        # interpret naive timestamps as tz (optional)
        df[ts_col] = df[ts_col].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

    df = df.set_index(ts_col).sort_index()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}. Columns: {list(df.columns)}")

    for c in REQUIRED_COLS:
        df[c] = df[c].astype(float)

    return df
