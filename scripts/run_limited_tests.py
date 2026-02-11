"""
CLI wrapper for spec-driven limited tests.

Example (entry-only: SMA grid + fixed ATR exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin sma_cross \
  --entry-params '{"fast":[10,20,30,40,50,60,70,80,90,100],"slow":[100,125,150,175,200,225,250,275,300,325]}' \
  --exit-plugin atr_brackets \
  --exit-params '{"rr":2.0,"sldist_atr_mult":1.5,"atr_period":14}' \
  --commission-rt 5

Example (entry-only: SMA grid + time stop exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin sma_cross \
  --entry-params '{"fast":[20,30,40,50,60,70,80],"slow":[125,150,175,200,225,250,275,300,325,350]}' \
  --exit-plugin time_exit \
  --exit-params '{"hold_bars":[1]}' \
  --commission-rt 5

Example (exit-only: similar-approach Donchian entry, use strategy exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin donchian_breakout \
  --entry-params '{"lookback":[20]}' \
  --favourable-criteria '{"total_return_%":{">":0}}' \
  --exit-plugin atr_brackets \
  --exit-params '{"rr":[1.0,1.5,2.0,2.5,3.0],"sldist_atr_mult":[0.5,1.0,1.5,2.0,2.5,3.0],"atr_period":14}' \
  --commission-rt 5

Example (monkey entry: random 132 entry signals, keep strategy exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin monkey_entry \
  --entry-params '{"target_entries":132,"side":"both","long_ratio":0.5}' \
  --seed-count 100 \
  --seed-start 1 \
  --favourable-criteria '{"mode":"all","rules":[{"metric":"total_return_%","op":"<","value":16.3},{"metric":"max_drawdown_abs_%","op":">","value":11.4}]}' \
  --pass-threshold 90 \
  --commission-rt 5

Example (monkey exit: random exit timing around core avg bars held):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --exit-plugin monkey_exit \
  --exit-params '{"avg_hold_bars":15.75}' \
  --exit-seed-count 100 \
  --exit-seed-start 1 \
  --favourable-criteria '{"mode":"all","rules":[{"metric":"total_return_%","op":"<","value":16.3},{"metric":"max_drawdown_abs_%","op":">","value":11.4}]}' \
  --pass-threshold 90 \
  --commission-rt 5
"""

from __future__ import annotations

import argparse
import copy
import importlib
import itertools
import json
import math
import time
from pathlib import Path

import pandas as pd

from quantbt.core.engine import BacktestConfig
from quantbt.core.engine_limited import run_backtest_limited
from quantbt.experiments.limited.base import limited_test_pass_rate
from quantbt.experiments.limited.criteria import parse_favourable_criteria, criteria_pass
from quantbt.experiments.limited.runlog import make_limited_run_dir, write_json
from quantbt.plugins import load_default_plugins, get_entry, get_exit, get_sizing
from quantbt.plugins.combiner import combine_signals
from quantbt.io.datasets import read_dataset_meta, dataset_tag_for_runs


def load_json_arg(value: str) -> dict:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def expand_params(params: dict) -> list[dict]:
    base = {}
    grid_keys = []
    grid_values = []
    for k, v in params.items():
        if isinstance(v, list) and not k.endswith("_values"):
            grid_keys.append(k)
            grid_values.append(v)
        else:
            base[k] = v

    if not grid_keys:
        return [dict(base)]

    combos = []
    for values in itertools.product(*grid_values):
        item = dict(base)
        item.update(dict(zip(grid_keys, values)))
        combos.append(item)
    return combos


def _entry_plugin_names(entry_spec: dict) -> tuple[str, ...]:
    rules = entry_spec.get("rules", [])
    names = [str(rule.get("name", "")).strip() for rule in rules if str(rule.get("name", "")).strip()]
    return tuple(sorted(names))


def _exit_plugin_name(strategy_spec: dict) -> str:
    return str(strategy_spec.get("exit", {}).get("name", "")).strip()


def classify_test_focus(strategy_spec: dict, base_strategy_spec: dict) -> str:
    current_entry = _entry_plugin_names(strategy_spec.get("entry", {}))
    base_entry = _entry_plugin_names(base_strategy_spec.get("entry", {}))

    current_exit = _exit_plugin_name(strategy_spec)
    base_exit = _exit_plugin_name(base_strategy_spec)

    entry_same = bool(current_entry) and current_entry == base_entry
    exit_same = bool(current_exit) and current_exit == base_exit

    if entry_same and exit_same:
        return "core_system_test"
    if entry_same and not exit_same:
        return "entry_test"
    if exit_same and not entry_same:
        return "exit_test"

    raise ValueError(
        "Both entry and exit plugins differ from the base strategy. "
        "This run is blocked because it is not testing the original strategy."
    )


def infer_test_name(strategy_spec: dict, *, test_focus: str) -> str:
    entry = strategy_spec.get("entry", {})
    exit_ = strategy_spec.get("exit", {})
    rules = entry.get("rules", [])

    entry_names = [str(r.get("name", "entry")) for r in rules] or ["entry"]
    entry_tag = "+".join(entry_names)
    exit_name = str(exit_.get("name", "exit"))

    if test_focus not in {"core_system_test", "entry_test", "exit_test"}:
        raise ValueError(f"unsupported test_focus: {test_focus}")

    exit_style_map = {
        "atr_brackets": "fixed_atr_exit",
        "time_exit": "time_exit",
        "random_time_exit": "random_exit",
        "monkey_exit": "monkey_exit",
    }
    exit_tag = exit_style_map.get(exit_name, f"{exit_name}_exit")
    return f"{test_focus}__{entry_tag}__{exit_tag}"


def iter_entries_from_signals(df_sig: pd.DataFrame, *, use_atr: bool):
    idx = df_sig.index.to_list()
    for i in range(len(idx) - 1):
        t = idx[i]
        t_next = idx[i + 1]
        if bool(df_sig.at[t, "long_entry"]):
            e = {
                "entry_i": i + 1,
                "entry_time": t_next,
                "side": "long",
                "entry_open": float(df_sig.at[t_next, "open"]),
                "prev_low": float(df_sig.at[t, "low"]),
                "prev_high": float(df_sig.at[t, "high"]),
            }
        elif bool(df_sig.at[t, "short_entry"]):
            e = {
                "entry_i": i + 1,
                "entry_time": t_next,
                "side": "short",
                "entry_open": float(df_sig.at[t_next, "open"]),
                "prev_low": float(df_sig.at[t, "low"]),
                "prev_high": float(df_sig.at[t, "high"]),
            }
        else:
            continue

        if use_atr:
            atr = float(df_sig.at[t_next, "atr"])
            if not math.isfinite(atr) or atr <= 0:
                continue
            e["atr"] = atr

        yield e


def _fmt_value(v):
    if v is None:
        return "nan"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.2f}"
    return str(v)


def _criteria_status(criteria: dict, summary: dict) -> str:
    parts = []
    for rule in criteria.get("rules", []):
        metric = rule["metric"]
        op = rule["op"]
        target = rule["value"]
        got = summary.get(metric)
        parts.append(f"{metric}{op}{_fmt_value(target)} (got {_fmt_value(got)})")
    mode = criteria.get("mode", "all")
    joined = "; ".join(parts) if parts else "none"
    return f"{mode}: {joined}"


def print_progress(i, total, elapsed, last_summary, *, pass_pct, criteria):
    pct = 100 * i / total if total else 100.0
    rate = elapsed / i if i > 0 else 0.0
    eta = rate * (total - i) if total else 0.0
    print(
        f"[{i:>4}/{total}] "
        f"{pct:6.2f}% | "
        f"elapsed {elapsed:6.1f}s | "
        f"ETA {eta:6.1f}s | "
        f"last_ret {last_summary.get('total_return_%', float('nan')):6.2f}% | "
        f"pass {pass_pct:6.2f}% | "
        f"criteria {_criteria_status(criteria, last_summary)}",
        flush=True,
    )


def run_spec(spec: dict, *, progress_every: int = 10):
    load_default_plugins()

    if "data" not in spec:
        raise ValueError("spec must include 'data'")

    data_path = Path(spec["data"])
    ts_col = spec.get("ts_col", "timestamp")

    df = pd.read_csv(data_path)
    if ts_col not in df.columns:
        raise ValueError(f"ts_col '{ts_col}' not in data")
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col)

    cfg_dict = spec.get("config", {})
    cfg = BacktestConfig(**cfg_dict)

    strat = spec["strategy"]
    entry_spec = strat["entry"]
    exit_spec = strat["exit"]
    sizing_spec = strat.get("sizing", {"name": "fixed_risk", "params": {}})

    entry_mode = entry_spec.get("mode", "all")
    vote_k = entry_spec.get("vote_k")
    rules = entry_spec.get("rules", [])
    if not rules:
        raise ValueError("strategy.entry.rules must be non-empty")

    entry_variants = []
    skipped = []
    for rule in rules:
        name = rule["name"]
        plugin = get_entry(name)
        params_list = expand_params(rule.get("params", {}))
        valid = []
        for p in params_list:
            validator = getattr(plugin, "validate", None)
            if callable(validator) and not validator(p):
                skipped.append((name, p))
                continue
            valid.append({"name": name, "plugin": plugin, "params": p})
        if not valid:
            raise ValueError(f"no valid params for entry rule '{name}'")
        entry_variants.append(valid)
    if skipped:
        print(f"skipped {len(skipped)} invalid entry param sets")

    exit_plugin = get_exit(exit_spec["name"])
    exit_param_space = expand_params(exit_spec.get("params", {}))
    if not exit_param_space:
        exit_param_space = [{}]

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
    total = 1
    for variants in entry_variants:
        total *= len(variants)
    total *= len(exit_param_space)

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
        df_sig = df.loc[combined.index].copy()
        df_sig["long_entry"] = combined["long_entry"]
        df_sig["short_entry"] = combined["short_entry"]
        if requires_atr:
            df_sig["atr"] = atr_series.reindex(df_sig.index)

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

def main():
    parser = argparse.ArgumentParser(description="Run limited tests with plugin overrides.")
    parser.add_argument("--strategy", required=True, help="Strategy module path with STRATEGY dict.")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV.")
    parser.add_argument("--ts-col", default="timestamp")
    parser.add_argument("--run-base", default=None)
    parser.add_argument("--test-name", default=None)
    parser.add_argument("--favourable-criteria", default=None)
    parser.add_argument("--pass-threshold", type=float, default=None)
    parser.add_argument("--min-trades", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--commission-rt", type=float, default=None, help="Commission per round trip (USD per standard lot).")
    parser.add_argument("--lot-size", type=float, default=None, help="Lot size in units for commission scaling.")

    parser.add_argument("--entry-plugin", default=None, help="Override entry plugin name.")
    parser.add_argument("--entry-params", default=None, help="JSON for entry params (string or file).")
    parser.add_argument("--seed-count", type=int, default=None, help="Generate a seed grid for entry params: seed=[seed_start..seed_start+seed_count-1].")
    parser.add_argument("--seed-start", type=int, default=1, help="Start value for --seed-count seed grid.")
    parser.add_argument("--entry-mode", default=None, choices=["all", "any", "vote"], help="Entry combiner mode.")
    parser.add_argument("--vote-k", type=int, default=None, help="Vote threshold for entry mode.")

    parser.add_argument("--exit-plugin", default=None, help="Override exit plugin name.")
    parser.add_argument("--exit-params", default=None, help="JSON for exit params (string or file).")
    parser.add_argument("--exit-seed-count", type=int, default=None, help="Generate a seed grid for exit params: seed=[exit_seed_start..exit_seed_start+exit_seed_count-1].")
    parser.add_argument("--exit-seed-start", type=int, default=1, help="Start value for --exit-seed-count seed grid.")

    parser.add_argument("--sizing-plugin", default=None, help="Override sizing plugin name.")
    parser.add_argument("--sizing-params", default=None, help="JSON for sizing params (string or file).")

    args = parser.parse_args()

    if args.seed_count is not None and args.seed_count <= 0:
        raise ValueError("--seed-count must be > 0")
    if args.seed_count is not None and args.entry_plugin is None:
        raise ValueError("--seed-count requires --entry-plugin")
    if args.exit_seed_count is not None and args.exit_seed_count <= 0:
        raise ValueError("--exit-seed-count must be > 0")
    if args.exit_seed_count is not None and args.exit_plugin is None:
        raise ValueError("--exit-seed-count requires --exit-plugin")

    mod = importlib.import_module(args.strategy)
    if not hasattr(mod, "STRATEGY"):
        raise ValueError(f"{args.strategy} missing STRATEGY dict")

    base_strategy = copy.deepcopy(mod.STRATEGY)
    spec = {"strategy": copy.deepcopy(base_strategy)}
    spec["data"] = args.data
    spec["ts_col"] = args.ts_col

    if args.run_base is not None:
        spec.setdefault("test", {})["run_base"] = args.run_base
    if args.test_name is not None:
        spec.setdefault("test", {})["test_name"] = args.test_name
    if args.favourable_criteria is not None:
        spec.setdefault("test", {})["favourable_criteria"] = args.favourable_criteria
    if args.pass_threshold is not None:
        spec.setdefault("test", {})["pass_threshold_pct"] = args.pass_threshold
    if args.min_trades is not None:
        spec.setdefault("test", {})["min_trades"] = args.min_trades
    # Default to strategy module name for run folder if not explicitly set.
    spec.setdefault("test", {}).setdefault("strategy_tag", args.strategy.rsplit(".", 1)[-1])
    if args.commission_rt is not None or args.lot_size is not None:
        cfg = spec.setdefault("config", {})
        if args.commission_rt is not None:
            cfg["commission_per_round_trip"] = args.commission_rt
        if args.lot_size is not None:
            cfg["lot_size"] = args.lot_size

    if args.entry_plugin is not None:
        entry_params = load_json_arg(args.entry_params) if args.entry_params else {}
        if args.seed_count is not None:
            existing_seed = entry_params.get("seed")
            if isinstance(existing_seed, list):
                raise ValueError("entry-params already contains seed list; remove it when using --seed-count")
            start = int(args.seed_start)
            stop = start + int(args.seed_count)
            entry_params["seed"] = list(range(start, stop))
        spec["strategy"]["entry"] = {
            "mode": args.entry_mode or "all",
            "rules": [{"name": args.entry_plugin, "params": entry_params}],
        }
        if args.entry_mode == "vote":
            if args.vote_k is None:
                raise ValueError("--vote-k is required for entry-mode vote")
            spec["strategy"]["entry"]["vote_k"] = args.vote_k
    else:
        if args.entry_mode is not None:
            spec["strategy"]["entry"]["mode"] = args.entry_mode
        if args.vote_k is not None:
            spec["strategy"]["entry"]["vote_k"] = args.vote_k

    if args.exit_plugin is not None:
        exit_params = load_json_arg(args.exit_params) if args.exit_params else {}
        if args.exit_seed_count is not None:
            existing_seed = exit_params.get("seed")
            if isinstance(existing_seed, list):
                raise ValueError("exit-params already contains seed list; remove it when using --exit-seed-count")
            start = int(args.exit_seed_start)
            stop = start + int(args.exit_seed_count)
            exit_params["seed"] = list(range(start, stop))
        spec["strategy"]["exit"] = {"name": args.exit_plugin, "params": exit_params}
    elif args.exit_params is not None:
        raise ValueError("--exit-params requires --exit-plugin")

    if args.sizing_plugin is not None:
        sizing_params = load_json_arg(args.sizing_params) if args.sizing_params else {}
        spec["strategy"]["sizing"] = {"name": args.sizing_plugin, "params": sizing_params}
    elif args.sizing_params is not None:
        raise ValueError("--sizing-params requires --sizing-plugin")

    test_focus = classify_test_focus(spec["strategy"], base_strategy)
    spec.setdefault("test", {})["test_focus"] = test_focus
    spec.setdefault("test", {})["core_system_test"] = (test_focus == "core_system_test")
    spec.setdefault("test", {}).setdefault(
        "test_name",
        infer_test_name(spec["strategy"], test_focus=test_focus),
    )

    run_spec(spec, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
