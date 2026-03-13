from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_limited_summary(run_dir: Path) -> Dict[str, Any]:
    """Load limited run summary data.

    Returns a dict with keys: decision, pass_threshold, min_trades, criteria, metrics, is_monkey, etc.
    """

    results_path = run_dir / "limited_results.csv"
    pass_path = run_dir / "pass_summary.json"
    meta_path = run_dir / "run_meta.json"

    if not (results_path.exists() and pass_path.exists() and meta_path.exists()):
        raise FileNotFoundError("Missing required limited run output files.")

    pass_summary = _read_json(pass_path)
    run_meta = _read_json(meta_path)
    results = pd.read_csv(results_path)

    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    initial_equity = float(spec.get("initial_equity", 100000.0)) if isinstance(spec, dict) else 100000.0

    results = results.copy()
    results["_net_profit_abs"] = (results.get("net_profit_abs")
                                 if "net_profit_abs" in results.columns
                                 else np.nan)

    total_iters = int(len(results))

    decision = "PASS" if bool(pass_summary.get("passed", False)) else "FAIL"

    criteria = run_meta.get("criteria") or {}
    pass_threshold = pass_summary.get("pass_threshold_%")
    min_trades = pass_summary.get("min_trades")

    is_monkey = bool(str(run_meta.get("strategy", "")).lower().endswith("monkey")) or bool(
        str(run_meta.get("strategy", "")).lower().startswith("monkey")
    )

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
    summary_path = run_dir / "summary.json"
    folds_path = run_dir / "folds.csv"

    if not (summary_path.exists() and folds_path.exists()):
        raise FileNotFoundError("Missing walk-forward summary or folds file.")

    summary = _read_json(summary_path)
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
    summary_path = run_dir / "mc_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Missing Monte Carlo summary json.")

    summary = _read_json(summary_path)
    metrics = summary.get("metrics", {})
    thresholds = summary.get("thresholds", {})
    checks = summary.get("threshold_checks", {})

    return {
        "summary": summary,
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
    }
