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
    spec.setdefault("test", {}).setdefault("strategy_tag", args.strategy.rsplit(".", 1)[-1])
    spec.setdefault("test", {}).setdefault("strategy_module", args.strategy)

    if args.commission_rt is not None or args.lot_size is not None:
        cfg = spec.setdefault("config", {})
        if args.commission_rt is not None:
            cfg["commission_per_round_trip"] = args.commission_rt
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

    test_focus = classify_test_focus(spec["strategy"], base_strategy)
    spec.setdefault("test", {})["test_focus"] = test_focus
    spec.setdefault("test", {})["core_system_test"] = (test_focus == "core_system_test")
    spec.setdefault("test", {}).setdefault(
        "test_name",
        infer_test_name(spec["strategy"], test_focus=test_focus),
    )
    return spec
