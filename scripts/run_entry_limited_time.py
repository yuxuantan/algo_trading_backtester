# Example:
# python scripts/run_entry_limited_time.py \
#   --data data/processed/eurusd_1h_20100101_20260209_dukascopy_python.csv \
#   --strategy sma_cross \
#   --entry-params '{"fast":50,"slow":200}' \
#   --hold-bars-range 1:50
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from quantbt.io.dataio import load_ohlc_csv
from quantbt.core.engine import BacktestConfig
from quantbt.core.engine_limited import run_backtest_limited
from quantbt.experiments.limited.base import limited_test, limited_test_pass_rate
from quantbt.experiments.limited.exits import (
    TimeExitParams, build_time_exit,
)
from quantbt.experiments.limited.runlog import make_limited_run_dir, write_json


def limited_progress_printer(i, total, elapsed, last_summary, favourable_so_far):
    pct = 100 * i / total
    rate = elapsed / i
    eta = rate * (total - i)
    fav_pct = 100 * favourable_so_far / i if i > 0 else 0.0

    print(
        f"[{i:>3}/{total}] "
        f"{pct:6.2f}% | "
        f"elapsed {elapsed:6.1f}s | "
        f"ETA {eta:6.1f}s | "
        f"favourable {fav_pct:5.1f}% | "
        f"last_ret {last_summary.get('total_return_%', float('nan')):6.2f}%",
        flush=True,
    )


def _to_dict(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return x
    return dict(x)


def parse_list(value: str, cast):
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("["):
        items = json.loads(raw)
    else:
        items = [s for s in raw.split(",") if s.strip() != ""]
    return [cast(v) for v in items]


def parse_range(value: str):
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 2:
        raise ValueError("hold-bars-range must be in 'start:end' format.")
    start = int(parts[0])
    end = int(parts[1])
    if end < start:
        raise ValueError("hold-bars-range end must be >= start.")
    return list(range(start, end + 1))


def load_strategy(strategy_name: str):
    module_name = strategy_name if "." in strategy_name else f"quantbt.strategies.{strategy_name}"
    mod = importlib.import_module(module_name)
    for attr in ("Params", "compute_features", "compute_signals", "iter_entries"):
        if not hasattr(mod, attr):
            raise AttributeError(f"{module_name} missing required attribute: {attr}")
    return mod, module_name


def load_entry_params(mod, entry_params_arg: str | None, cfg: BacktestConfig):
    params_cls = mod.Params
    if entry_params_arg is None:
        params = params_cls()
        if hasattr(params, "pip_size"):
            try:
                params = params_cls(**{**_to_dict(params), "pip_size": cfg.pip_size})
            except TypeError:
                pass
        return params

    path = Path(entry_params_arg)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(entry_params_arg)
    if "pip_size" in getattr(params_cls, "__annotations__", {}) and "pip_size" not in data:
        data["pip_size"] = cfg.pip_size
    return params_cls(**data)


def build_cfg(args) -> BacktestConfig:
    kwargs = {}
    if args.initial_equity is not None:
        kwargs["initial_equity"] = args.initial_equity
    if args.risk_pct is not None:
        kwargs["risk_pct"] = args.risk_pct
    if args.spread_pips is not None:
        kwargs["spread_pips"] = args.spread_pips
    if args.pip_size is not None:
        kwargs["pip_size"] = args.pip_size
    if args.conservative_same_bar is not None:
        kwargs["conservative_same_bar"] = args.conservative_same_bar
    if args.min_stop_dist is not None:
        kwargs["min_stop_dist"] = args.min_stop_dist
    if args.commission_rt is not None:
        kwargs["commission_per_round_trip"] = args.commission_rt
    if args.lot_size is not None:
        kwargs["lot_size"] = args.lot_size
    return BacktestConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Entry robustness test: time-exit params.")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV.")
    parser.add_argument("--ts-col", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--strategy", required=True, help="Strategy module (e.g., sma_cross).")
    parser.add_argument("--strategy-tag", default=None, help="Run folder tag; defaults to strategy module name.")
    parser.add_argument("--entry-params", default=None, help="JSON string or path to JSON file for entry params.")
    parser.add_argument("--run-base", default="runs/limited", help="Base directory for run logs.")
    parser.add_argument("--test-name", default="entry_test_time_exit", help="Run test name tag.")

    parser.add_argument("--hold-bars", default=None, help="Hold bars list (CSV or JSON list).")
    parser.add_argument("--hold-bars-range", default=None, help="Hold bars range (start:end, inclusive).")

    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--spread-pips", type=float, default=None)
    parser.add_argument("--pip-size", type=float, default=None)
    parser.add_argument("--min-stop-dist", type=float, default=None)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--commission-rt", type=float, default=5.0, help="Commission per round trip per 100k units.")
    parser.add_argument("--lot-size", type=float, default=100000.0, help="Units per standard lot for commission calc.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--conservative-same-bar", dest="conservative_same_bar", action="store_true")
    group.add_argument("--no-conservative-same-bar", dest="conservative_same_bar", action="store_false")
    parser.set_defaults(conservative_same_bar=None)

    args = parser.parse_args()

    cfg = build_cfg(args)
    strategy_mod, module_name = load_strategy(args.strategy)
    entry_params = load_entry_params(strategy_mod, args.entry_params, cfg)

    data_path = Path(args.data)
    df = load_ohlc_csv(str(data_path), ts_col=args.ts_col)

    df_feat = strategy_mod.compute_features(df, entry_params)
    df_sig = strategy_mod.compute_signals(df_feat)

    if args.hold_bars and args.hold_bars_range:
        raise ValueError("Use only one of --hold-bars or --hold-bars-range.")

    hold_bars_values = parse_list(args.hold_bars, int)
    if hold_bars_values is None:
        hold_bars_values = parse_range(args.hold_bars_range)
    if not hold_bars_values:
        raise ValueError("hold-bars must be non-empty (use --hold-bars or --hold-bars-range).")

    time_param_space = [TimeExitParams(hold_bars=hb) for hb in hold_bars_values]

    def make_param_sampler(param_list):
        it = iter(param_list)

        def _sample(_rng):
            return next(it)

        return _sample

    favourable_fn = lambda s: float(s.get("total_return_%", -999)) > 0

    strategy_tag = args.strategy_tag or module_name.split(".")[-1]
    run_dir = make_limited_run_dir(
        base=args.run_base,
        strategy=strategy_tag,
        dataset_tag=data_path.name,
        test_name=args.test_name,
    )

    pass_threshold = 70.0
    run_meta = {
        "strategy_module": module_name,
        "strategy_tag": strategy_tag,
        "dataset": str(data_path),
        "dataset_tag": data_path.name,
        "test_name": args.test_name,
        "entry_params": _to_dict(entry_params),
        "config": asdict(cfg),
        "time_param_space": [asdict(p) for p in time_param_space],
        "favourable_rule": "total_return_% > 0",
        "pass_threshold_%": pass_threshold,
        "min_trades": args.min_trades,
        "seed": args.seed,
    }
    write_json(run_dir / "run_meta.json", run_meta)

    res_time = limited_test(
        df_sig=df_sig,
        cfg=cfg,
        entry_iter_fn=strategy_mod.iter_entries,
        run_backtest_fn=run_backtest_limited,
        build_exit_fn=build_time_exit,
        exit_param_sampler=make_param_sampler(time_param_space),
        n_iter=len(time_param_space),
        favourable_fn=favourable_fn,
        seed=args.seed,
        min_trades=args.min_trades,
        progress_fn=limited_progress_printer,
    )

    pass_time = limited_test_pass_rate(res_time)
    pass_summary = {
        "favourable_pct": pass_time,
        "pass_threshold_%": pass_threshold,
        "passed": pass_time >= pass_threshold,
        "total_iters": int(len(res_time)),
        "min_trades": args.min_trades,
    }
    write_json(run_dir / "pass_summary.json", pass_summary)
    res_time.to_csv(run_dir / "limited_time_exit.csv", index=False)

    print(f"Time-exit favourable%:  {pass_time:.1f}%  -> {'PASS' if pass_time >= pass_threshold else 'FAIL'}")
    print(f"Saved: {run_dir}/limited_time_exit.csv")
    print(f"Run meta: {run_dir}/run_meta.json")
    print(f"Pass summary: {run_dir}/pass_summary.json")


if __name__ == "__main__":
    main()
