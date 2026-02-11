from __future__ import annotations

import copy
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quantbt.core.engine import BacktestConfig, run_backtest_sma_cross
from quantbt.experiments.runners import make_run_dir
from quantbt.io.dataio import load_ohlc_csv
from quantbt.io.datasets import dataset_tag_for_runs, read_dataset_meta, sha256_file
from quantbt.optimisers.grid import grid_search
from quantbt.optimisers.optuna_opt import optuna_search

from .fitness import enrich_summary_with_fitness
from .splits import WalkForwardFold, build_walkforward_splits, validate_oos_ratio


def _fmt_float(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "nan"
    if pd.isna(x):
        return "nan"
    return f"{x:.4f}"


def _load_json_arg(value: str) -> dict:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def load_strategy_module(strategy: str):
    if "." in strategy:
        mod = importlib.import_module(strategy)
    else:
        mod = importlib.import_module(f"quantbt.strategies.{strategy}")

    required = ["Params", "compute_features", "compute_signals", "build_brackets_from_signal"]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise ImportError(f"Strategy '{strategy}' missing required symbols: {missing}")
    return mod


def resolve_param_space(strategy_mod, param_space_arg: str | None) -> dict[str, list]:
    if param_space_arg is not None:
        raw = _load_json_arg(param_space_arg)
    else:
        raw = getattr(strategy_mod, "PARAM_SPACE", {})

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("param space must be a dict of key -> list/scalar")

    out: dict[str, list] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            vals = v
        else:
            vals = [v]
        if len(vals) == 0:
            raise ValueError(f"param space for '{k}' cannot be empty")
        out[str(k)] = vals
    return out


def has_optimizable_params(param_space: dict[str, list]) -> bool:
    return any(len(v) > 1 for v in param_space.values())


def fixed_params_from_space(param_space: dict[str, list]) -> dict[str, Any]:
    return {k: v[0] for k, v in param_space.items()}


def _slice_with_warmup(df: pd.DataFrame, *, start: int, end: int, warmup_bars: int) -> tuple[pd.DataFrame, Any, Any]:
    if start < 0 or end > len(df) or start >= end:
        raise ValueError(f"invalid slice bounds start={start}, end={end}, len={len(df)}")
    context_start = max(0, int(start) - max(0, int(warmup_bars)) - 1)
    df_ctx = df.iloc[context_start:end]
    start_ts = df.index[start]
    end_ts_exclusive = df.index[end] if end < len(df) else None
    return df_ctx, start_ts, end_ts_exclusive


def _filter_signal_window(df_sig: pd.DataFrame, *, start_ts, end_ts_exclusive):
    out = df_sig[df_sig.index >= start_ts]
    if end_ts_exclusive is not None:
        out = out[out.index < end_ts_exclusive]
    return out


def _build_strategy_params(strategy_mod, params: dict, cfg: BacktestConfig):
    payload = dict(params)
    ann = getattr(strategy_mod.Params, "__annotations__", {})
    if "pip_size" in ann and "pip_size" not in payload:
        payload["pip_size"] = cfg.pip_size
    return strategy_mod.Params(**payload)


def _constraints_ok(strategy_mod, params: dict) -> bool:
    fn = getattr(strategy_mod, "constraints", None)
    if callable(fn):
        return bool(fn(params))
    return True


def evaluate_on_slice(
    *,
    df: pd.DataFrame,
    strategy_mod,
    params: dict,
    cfg: BacktestConfig,
    start: int,
    end: int,
    warmup_bars: int,
    margin_rate: float,
    required_margin_abs_override: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    if not _constraints_ok(strategy_mod, params):
        return None

    try:
        sp = _build_strategy_params(strategy_mod, params, cfg)
    except Exception:
        return None

    try:
        df_ctx, start_ts, end_ts_exclusive = _slice_with_warmup(
            df,
            start=start,
            end=end,
            warmup_bars=warmup_bars,
        )
        df_feat = strategy_mod.compute_features(df_ctx, sp)
        df_sig_all = strategy_mod.compute_signals(df_feat)
        df_sig = _filter_signal_window(
            df_sig_all,
            start_ts=start_ts,
            end_ts_exclusive=end_ts_exclusive,
        )
        if len(df_sig) < 2:
            return None

        equity_df, trades_df, summary = run_backtest_sma_cross(
            df_sig,
            build_brackets_fn=strategy_mod.build_brackets_from_signal,
            strategy_params=sp,
            cfg=cfg,
        )
    except Exception:
        return None

    summary = enrich_summary_with_fitness(
        summary,
        equity_df=equity_df,
        trades_df=trades_df,
        initial_equity=cfg.initial_equity,
        margin_rate=margin_rate,
        required_margin_abs_override=required_margin_abs_override,
    )
    return equity_df, trades_df, summary


def _make_optuna_suggest_fn(param_space: dict[str, list]):
    def suggest_fn(trial):
        params = {}
        for k, vals in param_space.items():
            if all(isinstance(v, int) for v in vals):
                params[k] = trial.suggest_int(k, int(min(vals)), int(max(vals)))
            else:
                params[k] = trial.suggest_float(k, float(min(vals)), float(max(vals)))
        return params

    return suggest_fn


def _sort_results(df: pd.DataFrame, *, objective: str, direction: str) -> pd.DataFrame:
    if df.empty:
        return df
    asc = direction == "minimize"
    return df.sort_values(objective, ascending=asc).reset_index(drop=True)


def optimize_slice(
    *,
    df: pd.DataFrame,
    strategy_mod,
    param_space: dict[str, list],
    cfg: BacktestConfig,
    start: int,
    end: int,
    warmup_bars: int,
    optimizer: str,
    objective: str,
    direction: str,
    min_trades: int,
    n_trials: int,
    timeout_s: int,
    sampler: str,
    seed: int,
    margin_rate: float,
    required_margin_abs_override: float | None,
    study_name: str,
    storage_url: str,
    progress_prefix: str | None = None,
    progress_every: int = 20,
) -> tuple[dict, dict, pd.DataFrame]:
    optimizable = has_optimizable_params(param_space)
    fixed_params = fixed_params_from_space(param_space)

    def run_once(p: dict):
        out = evaluate_on_slice(
            df=df,
            strategy_mod=strategy_mod,
            params=p,
            cfg=cfg,
            start=start,
            end=end,
            warmup_bars=warmup_bars,
            margin_rate=margin_rate,
            required_margin_abs_override=required_margin_abs_override,
        )
        if out is None:
            return None
        _eq, _trades, summary = out
        if int(summary.get("trades", 0)) < int(min_trades):
            return None
        if objective not in summary:
            return None
        return summary

    if not optimizable:
        if progress_prefix:
            print(
                f"{progress_prefix} no optimizable params; evaluating fixed parameter set",
                flush=True,
            )
        summary = run_once(fixed_params)
        if summary is None:
            raise RuntimeError("Fixed-parameter evaluation failed on slice")
        row = {**fixed_params, **summary}
        results = pd.DataFrame([row])
        return fixed_params, summary, results

    if optimizer == "grid":
        best_so_far: float | None = None

        def _grid_progress(i, total, params, summary, elapsed, rows_so_far):
            nonlocal best_so_far
            obj = summary.get(objective)
            if obj is not None and not pd.isna(obj):
                obj_f = float(obj)
                if best_so_far is None:
                    best_so_far = obj_f
                elif direction == "maximize":
                    best_so_far = max(best_so_far, obj_f)
                else:
                    best_so_far = min(best_so_far, obj_f)

            if progress_prefix is None:
                return
            if progress_every > 1 and i % progress_every != 0 and i != total:
                return

            print(
                f"{progress_prefix} [grid {i}/{total}] "
                f"elapsed={elapsed:.1f}s "
                f"best_{objective}={_fmt_float(best_so_far)} "
                f"last_{objective}={_fmt_float(obj)} "
                f"rows={rows_so_far}",
                flush=True,
            )

        results = grid_search(
            param_space=param_space,
            run_once_fn=run_once,
            objective_key=objective,
            constraints_fn=lambda p: _constraints_ok(strategy_mod, p),
            progress_fn=_grid_progress,
        )
        results = _sort_results(results, objective=objective, direction=direction)
    elif optimizer == "optuna":
        def _optuna_progress(info: dict):
            if progress_prefix is None:
                return
            trial_no = int(info.get("trial_number", -1)) + 1
            if progress_every > 1 and trial_no % progress_every != 0:
                return
            print(
                f"{progress_prefix} [optuna trial {trial_no}/{n_trials}] "
                f"best_{objective}={_fmt_float(info.get('best_value'))} "
                f"last_{objective}={_fmt_float(info.get('objective'))}",
                flush=True,
            )

        timeout = None if timeout_s <= 0 else timeout_s
        results = optuna_search(
            study_name=study_name,
            storage_url=storage_url,
            sampler=sampler,
            direction=direction,
            n_trials=n_trials,
            timeout_s=timeout,
            seed=seed,
            suggest_fn=_make_optuna_suggest_fn(param_space),
            run_once_fn=run_once,
            objective_key=objective,
            constraints_fn=lambda p: _constraints_ok(strategy_mod, p),
            prune_fn=None,
            progress_fn=_optuna_progress,
        )
        results = _sort_results(results, objective=objective, direction=direction)
    else:
        raise ValueError(f"unsupported optimizer: {optimizer}")

    if results.empty:
        raise RuntimeError("optimizer produced no valid rows for slice")

    best = results.iloc[0].to_dict()
    best_params = {k: best[k] for k in param_space.keys() if k in best}

    out = evaluate_on_slice(
        df=df,
        strategy_mod=strategy_mod,
        params=best_params,
        cfg=cfg,
        start=start,
        end=end,
        warmup_bars=warmup_bars,
        margin_rate=margin_rate,
        required_margin_abs_override=required_margin_abs_override,
    )
    if out is None:
        raise RuntimeError("best parameter replay failed on slice")
    _eq, _trades, best_summary = out
    return best_params, best_summary, results


def infer_warmup_bars(param_space: dict[str, list]) -> int:
    keys = ("lookback", "period", "window", "length", "bars", "fast", "slow")
    candidates = []
    for k, vals in param_space.items():
        lk = str(k).lower()
        if not any(t in lk for t in keys):
            continue
        numeric = []
        for v in vals:
            if isinstance(v, (int, float)):
                numeric.append(int(v))
        if numeric:
            candidates.append(max(numeric))
    return int(max(candidates)) if candidates else 0


def _dataset_meta_or_fallback(dataset_path: str) -> dict:
    meta = read_dataset_meta(dataset_path)
    if meta is not None:
        return meta
    return {
        "dataset_id": f"unknown_{sha256_file(dataset_path)[:12]}",
        "file_sha256": sha256_file(dataset_path),
        "file_path": dataset_path,
    }


def _fold_to_row(df: pd.DataFrame, fold: WalkForwardFold, best_params: dict, is_summary: dict, oos_summary: dict) -> dict:
    return {
        "fold": fold.fold,
        "is_start_bar": fold.is_start,
        "is_end_bar_exclusive": fold.is_end,
        "oos_start_bar": fold.oos_start,
        "oos_end_bar_exclusive": fold.oos_end,
        "is_start_time": str(df.index[fold.is_start]),
        "is_end_time_exclusive": str(df.index[fold.is_end]) if fold.is_end < len(df) else None,
        "oos_start_time": str(df.index[fold.oos_start]),
        "oos_end_time_exclusive": str(df.index[fold.oos_end]) if fold.oos_end < len(df) else None,
        "best_params": json.dumps(best_params, sort_keys=True, default=str),
        **{f"is_{k}": v for k, v in is_summary.items()},
        **{f"oos_{k}": v for k, v in oos_summary.items()},
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def run_walkforward(
    *,
    strategy: str,
    dataset: str,
    ts_col: str,
    run_base: str,
    optimizer: str,
    objective: str,
    direction: str,
    min_trades: int,
    is_bars: int,
    oos_bars: int,
    step_bars: int | None,
    anchored: bool,
    start_bar: int,
    end_bar: int | None,
    warmup_bars: int | None,
    margin_rate: float,
    required_margin_abs: float | None,
    baseline_full_data: bool,
    compound_oos: bool,
    n_trials: int,
    timeout_s: int,
    sampler: str,
    seed: int,
    param_space_arg: str | None,
    progress_every: int,
    cfg: BacktestConfig,
) -> Path:
    strategy_mod = load_strategy_module(strategy)
    param_space = resolve_param_space(strategy_mod, param_space_arg)
    warmup = infer_warmup_bars(param_space) if warmup_bars is None else int(warmup_bars)

    validate_oos_ratio(is_bars, oos_bars)

    df = load_ohlc_csv(dataset, ts_col=ts_col)
    dataset_meta = _dataset_meta_or_fallback(dataset)

    folds = build_walkforward_splits(
        total_bars=len(df),
        is_bars=is_bars,
        oos_bars=oos_bars,
        step_bars=step_bars,
        anchored=anchored,
        start_bar=start_bar,
        end_bar=end_bar,
    )
    print(
        f"[WFA] mode={'anchored' if anchored else 'unanchored'} "
        f"folds={len(folds)} is_bars={is_bars} oos_bars={oos_bars} "
        f"step_bars={step_bars if step_bars is not None else oos_bars}",
        flush=True,
    )

    strategy_tag = strategy.rsplit(".", 1)[-1]
    run_dir = make_run_dir(
        base=run_base,
        mode="walkforward",
        strategy=strategy_tag,
        dataset_tag=dataset_tag_for_runs(dataset, dataset_meta),
        variant=f"{optimizer}_{'anchored' if anchored else 'unanchored'}",
    )
    (run_dir / "folds").mkdir(parents=True, exist_ok=True)

    config = {
        "strategy": strategy,
        "dataset": dataset,
        "dataset_meta": dataset_meta,
        "optimizer": optimizer,
        "objective": objective,
        "direction": direction,
        "min_trades": min_trades,
        "is_bars": is_bars,
        "oos_bars": oos_bars,
        "step_bars": step_bars if step_bars is not None else oos_bars,
        "anchored": anchored,
        "start_bar": start_bar,
        "end_bar": end_bar,
        "warmup_bars": warmup,
        "margin_rate": margin_rate,
        "required_margin_abs": required_margin_abs,
        "baseline_full_data": baseline_full_data,
        "compound_oos": compound_oos,
        "param_space": param_space,
        "backtest_config": asdict(cfg),
    }
    _write_json(run_dir / "config.json", config)

    baseline = None
    if baseline_full_data:
        print("[WFA] running full-data baseline optimization...", flush=True)
        baseline_params, baseline_summary, baseline_results = optimize_slice(
            df=df,
            strategy_mod=strategy_mod,
            param_space=param_space,
            cfg=cfg,
            start=0,
            end=len(df),
            warmup_bars=warmup,
            optimizer=optimizer,
            objective=objective,
            direction=direction,
            min_trades=min_trades,
            n_trials=n_trials,
            timeout_s=timeout_s,
            sampler=sampler,
            seed=seed,
            margin_rate=margin_rate,
            required_margin_abs_override=required_margin_abs,
            study_name=f"{strategy_tag}_baseline",
            storage_url=f"sqlite:///{(run_dir / 'baseline_study.db').as_posix()}",
            progress_prefix="[baseline]",
            progress_every=progress_every,
        )
        baseline_results.to_csv(run_dir / "baseline_results.csv", index=False)
        baseline = {
            "best_params": baseline_params,
            "summary": baseline_summary,
            "n_rows": int(len(baseline_results)),
        }
        _write_json(run_dir / "baseline_summary.json", baseline)
        print(
            f"[WFA] baseline done best_{objective}={_fmt_float(baseline_summary.get(objective))} "
            f"trades={int(baseline_summary.get('trades', 0))}",
            flush=True,
        )

    fold_rows = []
    schedule_rows = []
    stitched_equity_parts = []
    stitched_trades_parts = []
    current_oos_equity = cfg.initial_equity
    oos_under_min_count = 0

    for fold in folds:
        print(
            f"[WFA] fold {fold.fold}/{len(folds)} "
            f"IS[{fold.is_start}:{fold.is_end}) OOS[{fold.oos_start}:{fold.oos_end})",
            flush=True,
        )
        fold_tag = f"fold_{fold.fold:03d}"
        fold_dir = run_dir / "folds" / fold_tag
        fold_dir.mkdir(parents=True, exist_ok=True)

        study_path = fold_dir / "study.db"
        best_params, is_summary, is_results = optimize_slice(
            df=df,
            strategy_mod=strategy_mod,
            param_space=param_space,
            cfg=cfg,
            start=fold.is_start,
            end=fold.is_end,
            warmup_bars=warmup,
            optimizer=optimizer,
            objective=objective,
            direction=direction,
            min_trades=min_trades,
            n_trials=n_trials,
            timeout_s=timeout_s,
            sampler=sampler,
            seed=seed + fold.fold,
            margin_rate=margin_rate,
            required_margin_abs_override=required_margin_abs,
            study_name=f"{strategy_tag}_{fold_tag}",
            storage_url=f"sqlite:///{study_path.as_posix()}",
            progress_prefix=f"[fold {fold.fold}/{len(folds)} IS]",
            progress_every=progress_every,
        )
        is_results.to_csv(fold_dir / "is_results.csv", index=False)

        oos_cfg = copy.deepcopy(cfg)
        if compound_oos:
            oos_cfg = BacktestConfig(**{**asdict(cfg), "initial_equity": float(current_oos_equity)})

        oos_eval = evaluate_on_slice(
            df=df,
            strategy_mod=strategy_mod,
            params=best_params,
            cfg=oos_cfg,
            start=fold.oos_start,
            end=fold.oos_end,
            warmup_bars=warmup,
            margin_rate=margin_rate,
            required_margin_abs_override=required_margin_abs,
        )
        if oos_eval is None:
            raise RuntimeError(f"OOS evaluation failed for fold {fold.fold}")

        oos_equity, oos_trades, oos_summary = oos_eval
        oos_min_trades_met = int(oos_summary.get("trades", 0)) >= int(min_trades)
        if not oos_min_trades_met:
            oos_under_min_count += 1

        current_oos_equity = float(oos_summary["final_equity"])

        oos_equity.to_csv(fold_dir / "oos_equity_curve.csv")
        oos_trades.to_csv(fold_dir / "oos_trades.csv", index=False)
        _write_json(
            fold_dir / "fold_summary.json",
            {
                "best_params": best_params,
                "is_summary": is_summary,
                "oos_summary": oos_summary,
            },
        )

        schedule_rows.append(
            {
                "fold": fold.fold,
                "oos_start_time": str(df.index[fold.oos_start]),
                "oos_end_time_exclusive": str(df.index[fold.oos_end]) if fold.oos_end < len(df) else None,
                "params": best_params,
            }
        )

        fold_row = _fold_to_row(df, fold, best_params, is_summary, oos_summary)
        fold_row["oos_min_trades_met"] = oos_min_trades_met
        fold_rows.append(fold_row)
        stitched_equity_parts.append(oos_equity)
        stitched_trades_parts.append(oos_trades)
        print(
            f"[WFA] fold {fold.fold}/{len(folds)} done "
            f"oos_{objective}={_fmt_float(oos_summary.get(objective))} "
            f"oos_trades={int(oos_summary.get('trades', 0))} "
            f"oos_min_trades_met={oos_min_trades_met}",
            flush=True,
        )

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(run_dir / "folds.csv", index=False)
    _write_json(run_dir / "walkforward_param_schedule.json", schedule_rows)

    oos_equity_df = pd.concat(stitched_equity_parts).sort_index()
    oos_trades_df = pd.concat(stitched_trades_parts, ignore_index=True)
    oos_equity_df.to_csv(run_dir / "oos_equity_curve.csv")
    oos_trades_df.to_csv(run_dir / "oos_trades.csv", index=False)

    agg_initial = cfg.initial_equity
    agg_summary = {
        "trades": int(len(oos_trades_df)),
        "final_equity": float(oos_equity_df["equity"].iloc[-1]),
        "total_return_%": float(((oos_equity_df["equity"].iloc[-1] / agg_initial) - 1.0) * 100.0),
    }
    agg_summary = enrich_summary_with_fitness(
        agg_summary,
        equity_df=oos_equity_df,
        trades_df=oos_trades_df,
        initial_equity=agg_initial,
        margin_rate=margin_rate,
        required_margin_abs_override=required_margin_abs,
    )

    wf_objective = agg_summary.get(objective)
    baseline_objective = None if baseline is None else baseline["summary"].get(objective)
    comparison = {
        "walkforward_objective": wf_objective,
        "baseline_objective": baseline_objective,
        "objective_delta_walkforward_minus_baseline": (
            None
            if baseline_objective is None or wf_objective is None
            else float(wf_objective - baseline_objective)
        ),
    }

    summary = {
        "status": "ok",
        "fold_count": int(len(folds)),
        "oos_under_min_trades_count": int(oos_under_min_count),
        "objective": objective,
        "direction": direction,
        "aggregated_oos_summary": agg_summary,
        "comparison": comparison,
        "baseline_included": baseline is not None,
    }
    _write_json(run_dir / "summary.json", summary)
    print(
        f"[WFA] complete aggregated_{objective}={_fmt_float(agg_summary.get(objective))} "
        f"oos_trades={int(agg_summary.get('trades', 0))}",
        flush=True,
    )
    return run_dir
