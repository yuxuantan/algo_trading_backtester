from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class SMACrossTestStratParams:
    fast: int = 50
    slow: int = 200
    rr: float = 2.0
    sl_buffer_pips: float = 1.0
    pip_size: float = 0.0001

    def __post_init__(self):
        object.__setattr__(self, "fast", int(self.fast))
        object.__setattr__(self, "slow", int(self.slow))
        object.__setattr__(self, "rr", float(self.rr))
        object.__setattr__(self, "sl_buffer_pips", float(self.sl_buffer_pips))
        object.__setattr__(self, "pip_size", float(self.pip_size))

        if self.fast <= 0 or self.slow <= 0:
            raise ValueError(f"fast/slow must be > 0. Got fast={self.fast}, slow={self.slow}")
        if self.slow <= self.fast:
            raise ValueError(f"slow must be > fast. Got fast={self.fast}, slow={self.slow}")


def compute_features(df: pd.DataFrame, p: SMACrossTestStratParams) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(int(p.fast)).mean()
    out["sma_slow"] = out["close"].rolling(int(p.slow)).mean()
    out = out.dropna(subset=["sma_fast", "sma_slow"])
    return out

def compute_signals(df_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_feat.copy()
    out["bull_cross"] = (out["sma_fast"] > out["sma_slow"]) & (out["sma_fast"].shift(1) <= out["sma_slow"].shift(1))
    out["bear_cross"] = (out["sma_fast"] < out["sma_slow"]) & (out["sma_fast"].shift(1) >= out["sma_slow"].shift(1))
    return out

def build_brackets_from_signal(
    side: str,
    entry_open: float,
    prev_low: float,
    prev_high: float,
    p: SMACrossTestStratParams,
):
    """
    Returns (sl, tp, stop_dist) in price units, or None if invalid.
    side: "long" or "short"
    """
    sl_buffer = p.sl_buffer_pips * p.pip_size

    if side == "long":
        sl = prev_low - sl_buffer
        stop_dist = entry_open - sl
        if stop_dist <= 0:
            return None
        tp = entry_open + p.rr * stop_dist
        return sl, tp, stop_dist

    if side == "short":
        sl = prev_high + sl_buffer
        stop_dist = sl - entry_open
        if stop_dist <= 0:
            return None
        tp = entry_open - p.rr * stop_dist
        return sl, tp, stop_dist

    raise ValueError("side must be 'long' or 'short'")


def iter_entries(df: pd.DataFrame):
    """
    Expects df has bull_cross / bear_cross computed on bar i,
    and we enter on bar i+1 open.
    """
    idx = df.index.to_list()
    for i in range(len(df) - 1):
        t = idx[i]
        t_next = idx[i + 1]

        if bool(df.at[t, "bull_cross"]):
            yield {
                "entry_i": i + 1,
                "entry_time": t_next,
                "side": "long",
                "entry_open": float(df.at[t_next, "open"]),
                "prev_low": float(df.at[t, "low"]),
                "prev_high": float(df.at[t, "high"]),
            }
        elif bool(df.at[t, "bear_cross"]):
            yield {
                "entry_i": i + 1,
                "entry_time": t_next,
                "side": "short",
                "entry_open": float(df.at[t_next, "open"]),
                "prev_low": float(df.at[t, "low"]),
                "prev_high": float(df.at[t, "high"]),
            }


# ---- Standard interface for universal runner ----
Params = SMACrossTestStratParams
SMACrossParams = SMACrossTestStratParams

# ---- Plugin-based strategy configuration ----
STRATEGY = {
    "entry": {
        "mode": "all",
        "rules": [
            {"name": "sma_cross", "params": {"fast": 50, "slow": 200}},
        ],
    },
    "exit": {
        "name": "atr_brackets",
        "params": {"rr": 2.0, "sldist_atr_mult": 1.5, "atr_period": 14},
    },
    "sizing": {
        "name": "fixed_risk",
        "params": {"risk_pct": 0.01},
    },
}
