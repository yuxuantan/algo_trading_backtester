from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Callable, Any
import random
import pandas as pd
import time


def _to_dict(x: Any) -> dict:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return x
    return dict(x)


def limited_test(
    *,
    df_sig: pd.DataFrame,
    cfg,
    entry_iter_fn,
    run_backtest_fn,
    build_exit_fn,
    exit_param_sampler,
    n_iter: int = 100,
    favourable_fn=None,
    seed: int = 42,
    min_trades: int = 30,
    progress_fn=None,        # 👈 NEW
) -> pd.DataFrame:

    """
    Generic limited test for ENTRY testing:
      - entry fixed (df_sig)
      - exit varied (exit_param_sampler)
      - objective is % favourable over n_iter

    favourable_fn(summary) default: total_return_% > 0
    Returns: DataFrame of iteration results (params + summary + favourable)
    """
    rng = random.Random(seed)
    favourable_fn = favourable_fn or (lambda s: float(s.get("total_return_%", -1e9)) > 0)

    rows = []
    start_ts = time.time()

    for k in range(1, n_iter + 1):
        exit_params = exit_param_sampler(rng)

        equity, trades, summary = run_backtest_fn(
            df_sig,
            cfg=cfg,
            entry_iter_fn=entry_iter_fn,
            build_exit_fn=build_exit_fn,
            exit_params=exit_params,
        )

        # reject tiny samples if desired
        if summary.get("trades", 0) < min_trades:
            ok = False
        else:
            ok = bool(favourable_fn(summary))

        rows.append({
            "iter": k,
            **_to_dict(exit_params),
            **summary,
            "favourable": ok,
        })

        if progress_fn is not None:
            elapsed = time.time() - start_ts
            progress_fn(
                i=k,
                total=n_iter,
                elapsed=elapsed,
                last_summary=summary,
                favourable_so_far=sum(r["favourable"] for r in rows),
            )

    return pd.DataFrame(rows)


def limited_test_pass_rate(results: pd.DataFrame) -> float:
    if results.empty:
        return 0.0
    return float(results["favourable"].mean() * 100.0)
