from __future__ import annotations

import pandas as pd

from quantbt.plugins.registry import register_entry
from quantbt.strategies import sma_cross_strategy


@register_entry("sma_cross")
def signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    p = sma_cross_strategy.Params(**params)
    df_feat = sma_cross_strategy.compute_features(df, p)
    df_sig = sma_cross_strategy.compute_signals(df_feat)

    out = pd.DataFrame(index=df_sig.index)
    out["long_entry"] = df_sig["bull_cross"].astype(bool)
    out["short_entry"] = df_sig["bear_cross"].astype(bool)
    return out


def _validate(params: dict) -> bool:
    try:
        sma_cross_strategy.Params(**params)
    except Exception:
        return False
    return True


signals.validate = _validate
