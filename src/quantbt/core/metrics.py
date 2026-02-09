import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def profit_factor(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return np.nan
    gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    return float(gains / losses) if losses > 0 else np.inf
