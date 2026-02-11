from __future__ import annotations

import argparse
import importlib
import time
from typing import Any

from quantbt.io.dataio import load_ohlc_csv
from quantbt.io.datasets import read_dataset_meta, sha256_file, dataset_tag_for_runs
from quantbt.core.engine import BacktestConfig, run_backtest_sma_cross
from quantbt.optimisers.grid import grid_search
from quantbt.experiments.runners import (
    make_run_dir,
    save_artifacts,
    append_run_index,
)
from quantbt.optimisers.optuna_opt import optuna_search


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def parse_list(s: str, cast=float) -> list:
    if s is None or s.strip() == "":
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in parse_list(s, cast=float)]


def parse_float_list(s: str) -> list[float]:
    return [float(x) for x in parse_list(s, cast=float)]


def load_strategy(strategy_name: str):
    mod = importlib.import_module(f"quantbt.strategies.{strategy_name}")
    required = ["Params", "compute_features", "compute_signals", "build_brackets_from_signal"]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise ImportError(
            f"Strategy '{strategy_name}' missing required symbols: {missing}"
        )
    return mod


def build_param_space(strategy_name: str, args) -> dict[str, list]:
    if strategy_name == "sma_cross_test_strat":
        return {
            "fast": parse_int_list(args.fast),
            "slow": parse_int_list(args.slow),
            "rr": parse_float_list(args.rr),
            "sl_buffer_pips": parse_float_list(args.sl_buffer_pips),
        }
    raise ValueError(f"No param space defined for strategy '{strategy_name}'")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Universal strategy optimizer")

    # Core
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--optimizer", default="grid", choices=["grid", "optuna"])

    # Optuna knobs
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--timeout-s", type=int, default=0)         # 0 = no timeout
    ap.add_argument("--sampler", default="tpe", choices=["tpe", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])

    # Data
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ts-col", default="timestamp")
    ap.add_argument("--timeframe", default="")
    ap.add_argument("--run-base", default="runs")

    # Objective / constraints
    ap.add_argument("--objective", default="total_return_%")
    ap.add_argument("--min-trades", type=int, default=30)

    # Backtest config
    ap.add_argument("--initial-equity", type=float, default=100_000.0)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--spread-pips", type=float, default=0.2)
    ap.add_argument("--pip-size", type=float, default=0.0001)
    ap.add_argument("--conservative-same-bar", action="store_true")

    # SMA cross params
    ap.add_argument("--fast", default="20,30,40,50,60")
    ap.add_argument("--slow", default="100,150,200,250,300")
    ap.add_argument("--rr", default="1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--sl-buffer-pips", default="0.5,1,2,5,10")

    args = ap.parse_args()

    # -------------------------------------------------
    # Load strategy + data
    # -------------------------------------------------
    strat = load_strategy(args.strategy)
    df = load_ohlc_csv(args.dataset, ts_col=args.ts_col)

    # Dataset metadata (versioning)
    dataset_meta = read_dataset_meta(args.dataset)
    if dataset_meta is None:
        dataset_meta = {
            "dataset_id": f"unknown_{sha256_file(args.dataset)[:12]}",
            "file_sha256": sha256_file(args.dataset),
            "file_path": args.dataset,
        }

    # -------------------------------------------------
    # Backtest config
    # -------------------------------------------------
    cfg = BacktestConfig(
        initial_equity=args.initial_equity,
        risk_pct=args.risk_pct,
        spread_pips=args.spread_pips,
        pip_size=args.pip_size,
        conservative_same_bar=args.conservative_same_bar,
    )

    param_space = build_param_space(args.strategy, args)

    def suggest_fn(trial):
        # Use CLI bounds by reading min/max from lists
        fast_min, fast_max = min(param_space["fast"]), max(param_space["fast"])
        slow_min, slow_max = min(param_space["slow"]), max(param_space["slow"])
        rr_min, rr_max = min(param_space["rr"]), max(param_space["rr"])
        slb_min, slb_max = min(param_space["sl_buffer_pips"]), max(param_space["sl_buffer_pips"])

        params = {
            "fast": trial.suggest_int("fast", fast_min, fast_max),
            "slow": trial.suggest_int("slow", slow_min, slow_max),
            "rr": trial.suggest_float("rr", rr_min, rr_max),
            "sl_buffer_pips": trial.suggest_float("sl_buffer_pips", slb_min, slb_max),
        }
        return params

    def constraints(p: dict[str, Any]) -> bool:
        if "fast" in p and "slow" in p and p["slow"] <= p["fast"]:
            return False
        return True

    # -------------------------------------------------
    # Run directory
    # -------------------------------------------------
    run_dir = make_run_dir(
        base=args.run_base,
        mode="optimize",
        strategy=args.strategy,
        dataset_tag=dataset_tag_for_runs(args.dataset, dataset_meta),
        variant=args.optimizer,
    )

    save_artifacts(
        run_dir,
        config={
            "strategy": args.strategy,
            "optimizer": args.optimizer,
            "dataset": args.dataset,
            "dataset_meta": dataset_meta,
            "timeframe": args.timeframe,
            "objective": args.objective,
            "min_trades": args.min_trades,
            "param_space": param_space,
            "backtest_config": cfg.__dict__,
        },
    )

    # -------------------------------------------------
    # Progress printer (Option A)
    # -------------------------------------------------
    best_so_far = {"value": float("-inf")}

    def optuna_progress(info):
        obj = info["objective"]
        if obj is not None and obj > best_so_far["value"]:
            best_so_far["value"] = obj
        print(
            f"[trial {info['trial_number']}] "
            f"obj={obj:.2f} | best={best_so_far['value']:.2f} | params={info['params']}",
            flush=True
        )

    start_ts = time.time()

    def progress_printer(i, total, params, summary, elapsed, rows_so_far):
        obj = summary.get(args.objective)
        if obj is not None and obj > best_so_far["value"]:
            best_so_far["value"] = obj

        pct = 100 * i / total
        rate = elapsed / i
        eta = rate * (total - i)

        print(
            f"[{i:>5}/{total}] "
            f"{pct:6.2f}% | "
            f"elapsed {elapsed:6.1f}s | "
            f"ETA {eta:6.1f}s | "
            f"best {best_so_far['value']:.2f} | "
            f"params {params}",
            flush=True,
        )

    # -------------------------------------------------
    # Optimizer callback
    # -------------------------------------------------
    def run_once(p):
        try:
            sp = strat.Params(**{**p, "pip_size": cfg.pip_size} if "pip_size" in strat.Params.__annotations__ else p)
        except Exception as e:
            print(f"[SKIP] bad params {p} -> {e}", flush=True)
            return None

        df_feat = strat.compute_features(df, sp)
        df_sig = strat.compute_signals(df_feat)

        equity, trades, summary = run_backtest_sma_cross(
            df_sig,
            build_brackets_fn=strat.build_brackets_from_signal,
            strategy_params=sp,
            cfg=cfg,
        )

        if summary["trades"] < args.min_trades:
            return None

        return summary

    # -------------------------------------------------
    # Run optimization
    # -------------------------------------------------
    if args.optimizer == "grid":
        results = grid_search(
            param_space=param_space,
            run_once_fn=run_once,
            objective_key=args.objective,
            constraints_fn=constraints,
            progress_fn=progress_printer,
        )

    elif args.optimizer == "optuna":
        # store the study DB inside the run folder so the run is self-contained
        storage_url = f"sqlite:///{(run_dir / 'study.db').as_posix()}"
        study_name = f"{args.strategy}_{dataset_meta['dataset_id']}"

        timeout_s = None if args.timeout_s <= 0 else args.timeout_s

        results = optuna_search(
            study_name=study_name,
            storage_url=storage_url,
            sampler=args.sampler,
            direction=args.direction,
            n_trials=args.n_trials,
            timeout_s=timeout_s,
            seed=args.seed,
            suggest_fn=suggest_fn,
            run_once_fn=run_once,
            objective_key=args.objective,
            constraints_fn=constraints,
            prune_fn=None,                 # optional
            progress_fn=optuna_progress,
        )

    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


    if results.empty:
        save_artifacts(run_dir, results_df=results, summary={"status": "empty_results"})
        print(f"No valid results. See {run_dir}")
        return

    best = results.iloc[0].to_dict()

    # -------------------------------------------------
    # Rerun best for artifacts
    # -------------------------------------------------
    best_params = {k: best[k] for k in param_space.keys()}
    if "pip_size" in strat.Params.__annotations__:
        best_params["pip_size"] = cfg.pip_size

    sp_best = strat.Params(**best_params)
    df_feat = strat.compute_features(df, sp_best)
    df_sig = strat.compute_signals(df_feat)

    equity, trades, summary = run_backtest_sma_cross(
        df_sig,
        build_brackets_fn=strat.build_brackets_from_signal,
        strategy_params=sp_best,
        cfg=cfg,
    )

    save_artifacts(
        run_dir,
        results_df=results,
        trades_df=trades,
        equity_df=equity,
        summary={
            "status": "ok",
            "best_params": best_params,
            "best_metrics": summary,
        },
    )

    append_run_index(
        run_dir,
        strategy=args.strategy,
        optimizer=args.optimizer,
        dataset=args.dataset,
        timeframe=args.timeframe,
        objective=args.objective,
        best_row=best,
        index_path=f"{args.run_base}/_index.csv",
        dataset_meta=dataset_meta
    )

    print(f"\n✔ Run complete. Artifacts saved to:\n{run_dir}")


if __name__ == "__main__":
    main()
