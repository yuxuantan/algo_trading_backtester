from __future__ import annotations

import argparse
import copy
import importlib

from .criteria import load_json_arg
from .naming import classify_test_focus, infer_test_name


def build_cli_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--signal-cache-max",
        type=int,
        default=128,
        help="Max cached entry-signal DataFrames (LRU). Use 0 to disable cache and minimize memory.",
    )
    parser.add_argument(
        "--no-save-trades",
        action="store_true",
        help="Do not collect/write tables/trades.csv (reduces memory usage on large runs).",
    )
    parser.add_argument("--commission-rt", type=float, default=5.0, help="Commission per round trip (USD per standard lot).")
    parser.add_argument("--spread-pips", type=float, default=0.2, help="Slippage/spread in pips.")
    parser.add_argument("--pip-size", type=float, default=None, help="Pip size used to convert spread pips into price units.")
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

    # Optional monkey exact-match prefilter (cheap flat-only schedule check before full backtest).
    parser.add_argument("--monkey-match-prefilter", action="store_true", help="Enable flat-only monkey exact-match prefilter when exit is time-based (e.g. monkey_exit/random_time_exit).")
    parser.add_argument("--monkey-match-target-trades", type=float, default=None, help="Baseline target executed trades for monkey prefilter.")
    parser.add_argument("--monkey-match-target-long-pct", type=float, default=None, help="Baseline target long trade %% (0-100) for monkey prefilter.")
    parser.add_argument("--monkey-match-target-avg-hold", type=float, default=None, help="Baseline target avg bars held for monkey prefilter.")
    parser.add_argument("--monkey-match-trade-tol-pct", type=float, default=10.0, help="Allowed +/- %% around target trades.")
    parser.add_argument("--monkey-match-long-tol-pp", type=float, default=5.0, help="Allowed +/- percentage points around target long %%.")
    parser.add_argument("--monkey-match-hold-tol-pct", type=float, default=5.0, help="Allowed +/- %% around target avg hold bars.")

    # Optional monkey runtime accelerators.
    parser.add_argument("--monkey-fast-summary", action="store_true", help="Use a faster summary-only evaluator for time-based monkey exits (no trade/equity export per iteration).")
    parser.add_argument(
        "--monkey-davey-style",
        action="store_true",
        help=(
            "Enable strict Davey monkey scoring: track return-worse%% and maxDD-worse%% separately; "
            "PASS requires both >= pass threshold."
        ),
    )
    parser.add_argument("--monkey-seq-stop", action="store_true", help="Enable sequential stopping for monkey tests using a Wilson CI on favourable %%. ")
    parser.add_argument("--monkey-seq-min-accepted", type=int, default=1000, help="Minimum accepted monkey runs before checking sequential stop.")
    parser.add_argument("--monkey-seq-check-every", type=int, default=200, help="Check sequential stop every N accepted runs.")
    parser.add_argument("--monkey-seq-fail-threshold", type=float, default=75.0, help="Early FAIL threshold for monkey favourable %% (upper CI must be below this).")
    parser.add_argument("--monkey-seq-z", type=float, default=1.96, help="Wilson interval z-score for sequential stopping (default ~95%% CI).")
    return parser


def _apply_seed_grid(
    params: dict,
    *,
    count: int | None,
    start: int,
    count_flag: str,
    params_flag: str,
) -> None:
    if count is None:
        return
    existing_seed = params.get("seed")
    if existing_seed is not None:
        raise ValueError(
            f"{params_flag} already contains 'seed'; remove it when using {count_flag}"
        )
    stop = int(start) + int(count)
    params["seed"] = list(range(int(start), stop))


def _validate_seed_args(args: argparse.Namespace) -> None:
    if args.seed_count is not None and args.seed_count <= 0:
        raise ValueError("--seed-count must be > 0")
    if args.seed_count is not None and args.entry_plugin is None:
        raise ValueError("--seed-count requires --entry-plugin")
    if args.exit_seed_count is not None and args.exit_seed_count <= 0:
        raise ValueError("--exit-seed-count must be > 0")
    if args.exit_seed_count is not None and args.exit_plugin is None:
        raise ValueError("--exit-seed-count requires --exit-plugin")
    if args.seed_count is not None and args.exit_seed_count is not None:
        print(
            "[INFO] Both --seed-count and --exit-seed-count are set. "
            "Iterations use Cartesian product (entry seeds x exit seeds).",
            flush=True,
        )


def build_spec_from_args(args: argparse.Namespace) -> dict:
    _validate_seed_args(args)

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
    if args.signal_cache_max is not None:
        if int(args.signal_cache_max) < 0:
            raise ValueError("--signal-cache-max must be >= 0")
        spec.setdefault("test", {})["signal_cache_max"] = int(args.signal_cache_max)
    if args.no_save_trades:
        spec.setdefault("test", {})["save_trades"] = False
    spec.setdefault("test", {}).setdefault("strategy_tag", args.strategy.rsplit(".", 1)[-1])
    spec.setdefault("test", {}).setdefault("strategy_module", args.strategy)

    if (
        args.commission_rt is not None
        or args.spread_pips is not None
        or args.pip_size is not None
        or args.lot_size is not None
    ):
        cfg = spec.setdefault("config", {})
        if args.commission_rt is not None:
            cfg["commission_per_round_trip"] = args.commission_rt
        if args.spread_pips is not None:
            cfg["spread_pips"] = args.spread_pips
        if args.pip_size is not None:
            cfg["pip_size"] = args.pip_size
        if args.lot_size is not None:
            cfg["lot_size"] = args.lot_size

    if args.entry_plugin is not None:
        entry_params = load_json_arg(args.entry_params) if args.entry_params else {}
        _apply_seed_grid(
            entry_params,
            count=args.seed_count,
            start=args.seed_start,
            count_flag="--seed-count",
            params_flag="entry-params",
        )
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
        _apply_seed_grid(
            exit_params,
            count=args.exit_seed_count,
            start=args.exit_seed_start,
            count_flag="--exit-seed-count",
            params_flag="exit-params",
        )
        spec["strategy"]["exit"] = {"name": args.exit_plugin, "params": exit_params}
    elif args.exit_params is not None:
        raise ValueError("--exit-params requires --exit-plugin")

    if args.sizing_plugin is not None:
        sizing_params = load_json_arg(args.sizing_params) if args.sizing_params else {}
        spec["strategy"]["sizing"] = {"name": args.sizing_plugin, "params": sizing_params}
    elif args.sizing_params is not None:
        raise ValueError("--sizing-params requires --sizing-plugin")

    if args.monkey_match_prefilter:
        if args.monkey_match_target_trades is None:
            raise ValueError("--monkey-match-prefilter requires --monkey-match-target-trades")
        if args.monkey_match_target_trades <= 0:
            raise ValueError("--monkey-match-target-trades must be > 0")
        if args.monkey_match_target_long_pct is not None and not (0.0 <= args.monkey_match_target_long_pct <= 100.0):
            raise ValueError("--monkey-match-target-long-pct must be between 0 and 100")
        if args.monkey_match_target_avg_hold is not None and args.monkey_match_target_avg_hold <= 0:
            raise ValueError("--monkey-match-target-avg-hold must be > 0")
        if args.monkey_match_trade_tol_pct < 0:
            raise ValueError("--monkey-match-trade-tol-pct must be >= 0")
        if args.monkey_match_long_tol_pp < 0:
            raise ValueError("--monkey-match-long-tol-pp must be >= 0")
        if args.monkey_match_hold_tol_pct < 0:
            raise ValueError("--monkey-match-hold-tol-pct must be >= 0")
        spec.setdefault("test", {})["monkey_match_prefilter"] = {
            "enabled": True,
            "target_trades": float(args.monkey_match_target_trades),
            "target_long_trade_pct": (
                None if args.monkey_match_target_long_pct is None else float(args.monkey_match_target_long_pct)
            ),
            "target_avg_hold_bars": (
                None if args.monkey_match_target_avg_hold is None else float(args.monkey_match_target_avg_hold)
            ),
            "trade_tol_pct": float(args.monkey_match_trade_tol_pct),
            "long_tol_pp": float(args.monkey_match_long_tol_pp),
            "hold_tol_pct": float(args.monkey_match_hold_tol_pct),
        }

    if args.monkey_fast_summary or args.monkey_seq_stop:
        if args.monkey_seq_min_accepted <= 0:
            raise ValueError("--monkey-seq-min-accepted must be > 0")
        if args.monkey_seq_check_every <= 0:
            raise ValueError("--monkey-seq-check-every must be > 0")
        if args.monkey_seq_z <= 0:
            raise ValueError("--monkey-seq-z must be > 0")
        runtime_cfg: dict = {
            "fast_summary": bool(args.monkey_fast_summary),
            "sequential_stop": {
                "enabled": bool(args.monkey_seq_stop),
                "min_accepted": int(args.monkey_seq_min_accepted),
                "check_every": int(args.monkey_seq_check_every),
                "fail_threshold_pct": float(args.monkey_seq_fail_threshold),
                "z": float(args.monkey_seq_z),
            },
        }
        spec.setdefault("test", {})["monkey_runtime"] = runtime_cfg

    if args.monkey_davey_style:
        if args.pass_threshold is None:
            spec.setdefault("test", {})["pass_threshold_pct"] = 90.0
        spec.setdefault("test", {})["monkey_davey"] = {"enabled": True}

    test_focus = classify_test_focus(spec["strategy"], base_strategy)
    spec.setdefault("test", {})["test_focus"] = test_focus
    spec.setdefault("test", {})["core_system_test"] = (test_focus == "core_system_test")
    spec.setdefault("test", {}).setdefault(
        "test_name",
        infer_test_name(spec["strategy"], test_focus=test_focus),
    )
    return spec
