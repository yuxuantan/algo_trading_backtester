from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quantbt.artifacts import (
    limited_iterations_path,
    limited_trades_path,
    spec_path,
    summary_path,
    walkforward_folds_path,
)
from quantbt.experiments.limited.criteria import criteria_pass, parse_favourable_criteria


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _limited_net_profit_series(results: pd.DataFrame, *, initial_equity: float) -> pd.Series:
    if "net_profit_abs" in results.columns:
        return pd.to_numeric(results["net_profit_abs"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if "total_return_%" in results.columns:
        ret = pd.to_numeric(results["total_return_%"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return (ret / 100.0) * float(initial_equity)

    if "final_equity" in results.columns:
        final_eq = pd.to_numeric(results["final_equity"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return final_eq - float(initial_equity)

    return pd.Series(np.nan, index=results.index)


def enrich_limited_results(
    run_dir: Path,
    results: pd.DataFrame,
    run_meta: Dict[str, Any],
    pass_summary: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Backfill missing limited-run metrics and recompute favourable flags.

    This keeps older run artifacts usable after summary-shape changes by deriving
    per-iteration metrics from `trades.csv` when needed.
    """

    out = results.copy()
    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    initial_equity = float(spec.get("initial_equity", 100000.0)) if isinstance(spec, dict) else 100000.0

    if "net_profit_abs" not in out.columns:
        out["net_profit_abs"] = _limited_net_profit_series(out, initial_equity=initial_equity)

    trades_path = limited_trades_path(run_dir)
    if trades_path.exists() and "iter" in out.columns:
        trades = pd.read_csv(trades_path)
        if "iter" in trades.columns and "pnl" in trades.columns:
            trade_iters = pd.to_numeric(trades["iter"], errors="coerce")
            pnl = pd.to_numeric(trades["pnl"], errors="coerce")
            valid = trade_iters.notna() & pnl.notna()
            if valid.any():
                grouped = pnl[valid].groupby(trade_iters[valid].astype(int))
                avg_profit_by_iter = grouped.mean()
                net_profit_by_iter = grouped.sum()
                iter_values = pd.to_numeric(out["iter"], errors="coerce")

                if "avg_profit_per_trade" not in out.columns:
                    out["avg_profit_per_trade"] = np.nan
                avg_profit = pd.to_numeric(out["avg_profit_per_trade"], errors="coerce").replace([np.inf, -np.inf], np.nan)
                avg_missing = avg_profit.isna()
                if avg_missing.any():
                    out.loc[avg_missing, "avg_profit_per_trade"] = iter_values[avg_missing].map(avg_profit_by_iter)

                net_profit = pd.to_numeric(out["net_profit_abs"], errors="coerce").replace([np.inf, -np.inf], np.nan)
                net_missing = net_profit.isna()
                if net_missing.any():
                    out.loc[net_missing, "net_profit_abs"] = iter_values[net_missing].map(net_profit_by_iter)

    criteria = parse_favourable_criteria(run_meta.get("criteria"))
    min_trades = _as_float(pass_summary.get("min_trades"))
    valid_mask = pd.Series(True, index=out.index)
    if "trades" in out.columns and np.isfinite(min_trades):
        trades_series = pd.to_numeric(out["trades"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_mask = trades_series >= min_trades
        valid_mask = valid_mask.fillna(False)
    criteria_metrics = [str(rule.get("metric", "")).strip() for rule in criteria.get("rules", [])]
    can_recompute_favourable = bool(criteria_metrics) and all(metric in out.columns for metric in criteria_metrics)
    if "iter" in out.columns and not out.empty and ("favourable" not in out.columns or can_recompute_favourable):
        favourable: list[bool] = []
        for _, row in out.iterrows():
            trades_n = _as_float(row.get("trades"))
            trades_ok = bool(np.isfinite(min_trades) and np.isfinite(trades_n) and trades_n >= min_trades) if np.isfinite(min_trades) else True
            ok = bool(trades_ok and criteria_pass(row.to_dict(), criteria))
            favourable.append(ok)
        out["favourable"] = favourable

    updated_pass_summary = dict(pass_summary) if isinstance(pass_summary, dict) else {}
    valid_iters = int(valid_mask.sum()) if len(valid_mask) else 0
    required_valid_iters = int(updated_pass_summary.get("required_valid_iters", 100))
    updated_pass_summary["valid_iters"] = valid_iters
    updated_pass_summary["required_valid_iters"] = required_valid_iters
    if "favourable" in out.columns:
        fav = out["favourable"]
        if fav.dtype != bool:
            fav = fav.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
        favourable_valid = int((fav & valid_mask).sum()) if len(fav) else 0
        favourable_pct = float((favourable_valid / valid_iters) * 100.0) if valid_iters > 0 else 0.0
        updated_pass_summary["favourable_pct"] = favourable_pct
        threshold_val = _as_float(updated_pass_summary.get("pass_threshold_%"))
        if np.isfinite(threshold_val):
            updated_pass_summary["passed"] = bool(
                favourable_pct >= threshold_val and valid_iters >= required_valid_iters
            )

    return out, updated_pass_summary


def load_limited_summary(run_dir: Path) -> Dict[str, Any]:
    """Load limited run summary data.

    Returns a dict with keys: decision, pass_threshold, min_trades, criteria, metrics, is_monkey, etc.
    """

    results_path = limited_iterations_path(run_dir)
    pass_path = summary_path(run_dir)
    meta_path = spec_path(run_dir)

    if not (results_path.exists() and pass_path.exists() and meta_path.exists()):
        raise FileNotFoundError("Missing required limited run output files.")

    pass_summary = _read_json(pass_path)
    run_meta = _read_json(meta_path)
    results = pd.read_csv(results_path)

    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    initial_equity = float(spec.get("initial_equity", 100000.0)) if isinstance(spec, dict) else 100000.0

    results, pass_summary = enrich_limited_results(run_dir, results, run_meta, pass_summary)
    results = results.copy()
    results["_net_profit_abs"] = _limited_net_profit_series(results, initial_equity=initial_equity)

    total_iters = int(len(results))

    criteria = run_meta.get("criteria") or {}
    pass_threshold = pass_summary.get("pass_threshold_%")
    min_trades = pass_summary.get("min_trades")

    is_monkey = bool(str(run_meta.get("strategy", "")).lower().endswith("monkey")) or bool(
        str(run_meta.get("strategy", "")).lower().startswith("monkey")
    )
    pass_summary = dict(pass_summary) if isinstance(pass_summary, dict) else {}
    decision = "PASS" if bool(pass_summary.get("passed", False)) else "FAIL"

    metrics: Dict[str, Any] = {}
    if is_monkey:
        davey_cfg = pass_summary.get("davey_style", {})
        metrics.update(
            {
                "total_trials": total_iters,
                "davey_return_worse_pct": _as_float(davey_cfg.get("return_worse_pct")),
                "davey_maxdd_worse_pct": _as_float(davey_cfg.get("maxdd_worse_pct")),
            }
        )
    else:
        profitable_runs = int((results.get("_net_profit_abs", 0) > 0).sum()) if not results.empty else 0
        median_profit = float(results.get("_net_profit_abs", pd.Series(dtype=float)).median())
        median_trades = float(pd.to_numeric(results.get("trades", pd.Series(dtype=float)), errors="coerce").median())
        metrics.update(
            {
                "total_iters": total_iters,
                "valid_iters": int(_as_float(pass_summary.get("valid_iters"))) if np.isfinite(_as_float(pass_summary.get("valid_iters"))) else 0,
                "required_valid_iters": int(_as_float(pass_summary.get("required_valid_iters"))) if np.isfinite(_as_float(pass_summary.get("required_valid_iters"))) else 100,
                "profitable_runs": profitable_runs,
                "median_profit": median_profit,
                "median_trades": median_trades,
            }
        )

    return {
        "decision": decision,
        "criteria": criteria,
        "pass_threshold": pass_threshold,
        "min_trades": min_trades,
        "is_monkey": is_monkey,
        "metrics": metrics,
        "pass_summary": pass_summary,
        "run_meta": run_meta,
    }


def load_walkforward_summary(run_dir: Path) -> Dict[str, Any]:
    wf_summary_path = summary_path(run_dir)
    folds_path = walkforward_folds_path(run_dir)

    if not (wf_summary_path.exists() and folds_path.exists()):
        raise FileNotFoundError("Missing walk-forward summary or folds file.")

    summary = _read_json(wf_summary_path)
    folds = pd.read_csv(folds_path)

    segment_count = int(len(folds))
    profit_col = "oos_net_profit_abs" if "oos_net_profit_abs" in folds.columns else "oos_total_return_%"
    profits = pd.to_numeric(folds.get(profit_col), errors="coerce").replace([np.inf, -np.inf], np.nan)
    profitable_pct = float((profits > 0).sum() * 100.0 / segment_count) if segment_count else float("nan")

    profits_finite = profits.dropna()
    total_oos_profit = float(profits_finite.sum()) if not profits_finite.empty else float("nan")
    top2_sum = float(profits_finite.nlargest(min(2, len(profits_finite))).sum()) if not profits_finite.empty else float("nan")
    top2_share_pct = (
        float((top2_sum / total_oos_profit) * 100.0)
        if np.isfinite(total_oos_profit) and total_oos_profit != 0
        else (float("inf") if segment_count else float("nan"))
    )

    return {
        "summary": summary,
        "segment_count": segment_count,
        "profitable_pct": profitable_pct,
        "top2_share_pct": top2_share_pct,
    }


def load_montecarlo_summary(run_dir: Path) -> Dict[str, Any]:
    mc_summary_path = summary_path(run_dir)
    if not mc_summary_path.exists():
        raise FileNotFoundError("Missing Monte Carlo summary json.")

    summary = _read_json(mc_summary_path)
    metrics = summary.get("metrics", {})
    thresholds = summary.get("thresholds", {})
    checks = summary.get("threshold_checks", {})

    return {
        "summary": summary,
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
    }
