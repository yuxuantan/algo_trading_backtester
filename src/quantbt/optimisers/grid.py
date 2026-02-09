from __future__ import annotations
from itertools import product
import pandas as pd
import time


def grid_search(
    *,
    param_space: dict,
    run_once_fn,
    objective_key: str,
    constraints_fn=None,
    progress_fn=None,   # 👈 NEW
) -> pd.DataFrame:

    keys = list(param_space.keys())
    values = [param_space[k] for k in keys]
    combos = list(product(*values))
    total = len(combos)

    rows = []
    start_ts = time.time()

    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))

        if constraints_fn is not None and not constraints_fn(params):
            continue

        summary = run_once_fn(params)
        if summary is None:
            continue

        rows.append({**params, **summary})

        if progress_fn is not None:
            progress_fn(
                i=i,
                total=total,
                params=params,
                summary=summary,
                elapsed=time.time() - start_ts,
                rows_so_far=len(rows),
            )

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(objective_key, ascending=False).reset_index(drop=True)

    return res
