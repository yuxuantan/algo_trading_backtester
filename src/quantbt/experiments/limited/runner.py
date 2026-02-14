from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pandas as pd

from quantbt.core.engine import BacktestConfig
from quantbt.core.engine_limited import run_backtest_limited
from quantbt.io.datasets import read_dataset_meta, dataset_tag_for_runs
from quantbt.plugins import get_exit, get_sizing, load_default_plugins
from quantbt.plugins.combiner import combine_signals

from .base import limited_test_pass_rate
from .criteria import criteria_pass, parse_favourable_criteria
from .data_prep import build_signal_frame, compute_atr, iter_entries_from_signals, load_price_frame
from .param_grid import build_entry_variants, build_exit_param_space, total_iterations
from .progress import print_progress
from .runlog import make_limited_run_dir, write_json


def run_spec(spec: dict, *, progress_every: int = 10):
    load_default_plugins()

    if "data" not in spec:
        raise ValueError("spec must include 'data'")

    data_path = Path(spec["data"])
    ts_col = spec.get("ts_col", "timestamp")

    df = load_price_frame(data_path, ts_col=ts_col)

    cfg_dict = spec.get("config", {})
    cfg = BacktestConfig(**cfg_dict)

    strat = spec["strategy"]
    entry_spec = strat["entry"]
    exit_spec = strat["exit"]
    sizing_spec = strat.get("sizing", {"name": "fixed_risk", "params": {}})

    entry_mode = entry_spec.get("mode", "all")
    vote_k = entry_spec.get("vote_k")
    rules = entry_spec.get("rules", [])

    entry_variants, skipped = build_entry_variants(rules)
    if skipped:
        print(f"skipped {len(skipped)} invalid entry param sets")

    exit_plugin = get_exit(exit_spec["name"])
    exit_param_space = build_exit_param_space(exit_spec)

    sizing_plugin = get_sizing(sizing_spec.get("name", "fixed_risk"))
    sizing_params = sizing_spec.get("params", {})

    criteria = parse_favourable_criteria(spec.get("test", {}).get("favourable_criteria"))
    min_trades = int(spec.get("test", {}).get("min_trades", 30))
    pass_threshold = float(spec.get("test", {}).get("pass_threshold_pct", 70.0))

    dataset_meta = read_dataset_meta(data_path)
    dataset_tag = dataset_tag_for_runs(data_path, dataset_meta)

    run_dir = make_limited_run_dir(
        base=spec.get("test", {}).get("run_base", "runs/limited"),
        strategy=spec.get("test", {}).get("strategy_tag", entry_spec.get("tag", "spec")),
        dataset_tag=dataset_tag,
        test_name=spec.get("test", {}).get("test_name", "limited_test"),
    )

    run_meta = {
        "spec": spec,
        "dataset_meta": dataset_meta,
        "criteria": criteria,
        "pass_threshold_%": pass_threshold,
        "min_trades": min_trades,
    }
    write_json(run_dir / "run_meta.json", run_meta)

    requires_atr = bool(getattr(exit_plugin, "requires_atr", False))
    atr_period = int(exit_spec.get("params", {}).get("atr_period", 14))
    atr_series = compute_atr(df, atr_period) if requires_atr else None

    signals_cache: dict[tuple, pd.DataFrame] = {}

    rows = []
    trade_rows: list[pd.DataFrame] = []
    total = total_iterations(entry_variants, exit_param_space)

    start_ts = time.time()
    iter_count = 0
    pass_count = 0

    def size_fn(**kwargs):
        return sizing_plugin(**kwargs, params=sizing_params)

    for entry_combo in itertools.product(*entry_variants):
        signals_list = []
        for rule in entry_combo:
            key = (rule["name"], tuple(sorted(rule["params"].items())))
            sig = signals_cache.get(key)
            if sig is None:
                sig = rule["plugin"](df, rule["params"])
                signals_cache[key] = sig
            signals_list.append(sig)

        combined = combine_signals(signals_list, mode=entry_mode, vote_k=vote_k)
        df_sig = build_signal_frame(
            df,
            combined,
            atr_series=atr_series if requires_atr else None,
        )

        entry_iter_fn = lambda d: iter_entries_from_signals(d, use_atr=requires_atr)

        for exit_params in exit_param_space:
            iter_count += 1
            _eq, _trades, summary = run_backtest_limited(
                df_sig,
                cfg=cfg,
                entry_iter_fn=entry_iter_fn,
                build_exit_fn=exit_plugin,
                exit_params=exit_params,
                size_fn=size_fn,
            )

            if _trades is not None and not _trades.empty:
                tdf = _trades.copy()
                tdf["iter"] = iter_count
                tdf["entry_params"] = json.dumps([r["params"] for r in entry_combo], default=str)
                tdf["exit_params"] = json.dumps(exit_params, default=str)
                trade_rows.append(tdf)

            ok = summary.get("trades", 0) >= min_trades and criteria_pass(summary, criteria)
            if ok:
                pass_count += 1

            rows.append({
                "iter": iter_count,
                "entry_params": [r["params"] for r in entry_combo],
                "exit_params": exit_params,
                **summary,
                "favourable": ok,
            })

            if progress_every and (iter_count % progress_every == 0 or iter_count == total):
                elapsed = time.time() - start_ts
                pass_pct = (pass_count / iter_count) * 100 if iter_count else 0.0
                print_progress(iter_count, total, elapsed, summary, pass_pct=pass_pct, criteria=criteria)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(run_dir / "limited_results.csv", index=False)
    if trade_rows:
        trades_df = pd.concat(trade_rows, ignore_index=True)
        trades_df.to_csv(run_dir / "limited_trades.csv", index=False)

    pass_rate = limited_test_pass_rate(res_df)
    pass_summary = {
        "favourable_pct": pass_rate,
        "pass_threshold_%": pass_threshold,
        "passed": pass_rate >= pass_threshold,
        "total_iters": int(len(res_df)),
        "min_trades": min_trades,
    }
    write_json(run_dir / "pass_summary.json", pass_summary)

    print(f"Favourable%: {pass_rate:.1f}% -> {'PASS' if pass_rate >= pass_threshold else 'FAIL'}")
    print(f"Saved: {run_dir}/limited_results.csv")
    if trade_rows:
        print(f"Saved: {run_dir}/limited_trades.csv")
