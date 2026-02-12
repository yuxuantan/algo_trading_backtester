from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _make_mc_run_dir(run_dir: Path) -> Path:
    base = run_dir / "monte_carlo"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%d%m%y_%H%M%S")
    stem = f"run_{ts}"
    out = base / stem
    suffix = 1
    while out.exists():
        suffix += 1
        out = base / f"{stem}_{suffix:02d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _load_initial_equity(run_dir: Path) -> float:
    cfg = _read_json(run_dir / "config.json")
    bt = cfg.get("backtest_config", {})
    if isinstance(bt, dict) and bt.get("initial_equity") is not None:
        return float(bt["initial_equity"])

    eq_path = run_dir / "oos_equity_curve.csv"
    if eq_path.exists():
        eq = pd.read_csv(eq_path)
        if not eq.empty and "equity" in eq.columns:
            return float(eq["equity"].iloc[0])
    raise ValueError("could not infer initial equity from run config or oos_equity_curve.csv")


def _build_trade_pool(
    trades_df: pd.DataFrame,
    *,
    pnl_mode: str,
    fixed_risk_dollars: float | None,
) -> np.ndarray:
    if pnl_mode == "actual":
        pool = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna().to_numpy(dtype=float)
    elif pnl_mode == "fixed_risk":
        if fixed_risk_dollars is None or fixed_risk_dollars <= 0:
            raise ValueError("--fixed-risk-dollars must be > 0 when pnl-mode=fixed_risk")
        if "r_multiple" not in trades_df.columns:
            raise ValueError("oos_trades.csv missing r_multiple required for pnl-mode=fixed_risk")
        r = pd.to_numeric(trades_df["r_multiple"], errors="coerce").dropna().to_numpy(dtype=float)
        pool = r * float(fixed_risk_dollars)
    else:
        raise ValueError(f"unsupported pnl_mode: {pnl_mode}")

    if len(pool) == 0:
        raise ValueError("no valid trades found in oos_trades.csv")
    return pool


def _sample_trade_sequence(
    rng: np.random.Generator,
    pool: np.ndarray,
    *,
    n_trades: int,
    replace: bool,
) -> np.ndarray:
    if replace:
        idx = rng.integers(0, len(pool), size=n_trades)
        return pool[idx]

    if n_trades > len(pool):
        raise ValueError(
            f"without replacement requires n_trades <= pool size. Got n_trades={n_trades}, pool={len(pool)}"
        )
    idx = rng.choice(len(pool), size=n_trades, replace=False)
    return pool[idx]


def _simulate_path(
    trade_pnls: np.ndarray,
    *,
    initial_equity: float,
    ruin_equity: float,
    stop_at_ruin: bool,
) -> tuple[dict, np.ndarray]:
    equity = float(initial_equity)
    peak = float(initial_equity)
    trough = float(initial_equity)
    ruin_hit = False
    ruin_trade_n = None
    max_dd_abs = 0.0
    n_trades = len(trade_pnls)
    path = np.empty(n_trades + 1, dtype=float)
    path[0] = equity

    for i, pnl in enumerate(trade_pnls, start=1):
        equity += float(pnl)
        path[i] = equity
        if equity > peak:
            peak = equity
        if equity < trough:
            trough = equity

        dd_abs = peak - equity
        if dd_abs > max_dd_abs:
            max_dd_abs = dd_abs

        if (not ruin_hit) and equity < ruin_equity:
            ruin_hit = True
            ruin_trade_n = i
            if stop_at_ruin:
                # Flatline the rest of the path once trading stops.
                if i < n_trades:
                    path[i + 1 :] = equity
                break
    else:
        # Loop exhausted without break.
        i = n_trades

    if i < n_trades and not stop_at_ruin:
        # Defensive branch; should not happen with current control flow.
        path[i + 1 :] = equity

    ret_pct = ((equity / float(initial_equity)) - 1.0) * 100.0
    max_dd_pct = (max_dd_abs / float(initial_equity)) * 100.0 if initial_equity > 0 else np.nan
    return_dd_ratio = ret_pct / max_dd_pct if max_dd_pct > 0 else np.nan

    summary = {
        "final_equity": float(equity),
        "return_%": float(ret_pct),
        "max_drawdown_abs": float(max_dd_abs),
        "max_drawdown_%": float(max_dd_pct),
        "return_drawdown_ratio": float(return_dd_ratio) if np.isfinite(return_dd_ratio) else np.nan,
        "worst_equity": float(trough),
        "ruin_hit": bool(ruin_hit),
        "ruin_trade_n": int(ruin_trade_n) if ruin_trade_n is not None else np.nan,
    }
    return summary, path


def _finite_median(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    return float(s.median())


def _finite_quantile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


def run_monte_carlo(
    *,
    run_dir: str | Path,
    n_sims: int,
    n_trades: int | None,
    replace: bool,
    seed: int,
    ruin_equity: float,
    stop_at_ruin: bool,
    pnl_mode: str,
    fixed_risk_dollars: float | None,
    progress_every: int,
    save_sample_paths_count: int,
    save_quantile_paths: bool,
    target_risk_of_ruin_pct_max: float,
    target_median_max_dd_pct_max: float,
    target_median_return_pct_min: float,
    target_return_dd_ratio_min: float,
) -> Path:
    run_dir = Path(run_dir)
    trades_path = run_dir / "oos_trades.csv"
    if not trades_path.exists():
        raise ValueError(f"missing oos_trades.csv under run dir: {run_dir}")

    if n_sims <= 0:
        raise ValueError("n_sims must be > 0")
    if progress_every <= 0:
        raise ValueError("progress_every must be > 0")
    if save_sample_paths_count < 0:
        raise ValueError("save_sample_paths_count must be >= 0")

    trades_df = pd.read_csv(trades_path)
    initial_equity = _load_initial_equity(run_dir)

    pool = _build_trade_pool(
        trades_df,
        pnl_mode=pnl_mode,
        fixed_risk_dollars=fixed_risk_dollars,
    )
    sim_trades = len(pool) if n_trades is None else int(n_trades)
    if sim_trades <= 0:
        raise ValueError("n_trades must be > 0")

    rng = np.random.default_rng(int(seed))
    rows = []
    sample_paths: list[tuple[int, np.ndarray]] = []
    quantile_matrix = np.empty((n_sims, sim_trades + 1), dtype=float) if save_quantile_paths else None

    print(
        f"[MC] sims={n_sims} trades_per_sim={sim_trades} pool_size={len(pool)} "
        f"replace={replace} ruin_equity={ruin_equity:.2f}",
        flush=True,
    )
    for i in range(1, n_sims + 1):
        seq = _sample_trade_sequence(rng, pool, n_trades=sim_trades, replace=replace)
        res, path = _simulate_path(
            seq,
            initial_equity=initial_equity,
            ruin_equity=ruin_equity,
            stop_at_ruin=stop_at_ruin,
        )
        rows.append({"sim": i, **res})
        if quantile_matrix is not None:
            quantile_matrix[i - 1, :] = path
        if i <= save_sample_paths_count:
            sample_paths.append((i, path.copy()))

        if i % progress_every == 0 or i == n_sims:
            print(
                f"[MC] {i}/{n_sims} "
                f"last_ret={res['return_%']:.2f}% "
                f"last_dd={res['max_drawdown_%']:.2f}% "
                f"last_ruin={res['ruin_hit']}",
                flush=True,
            )

    sims_df = pd.DataFrame(rows)
    mc_run_dir = _make_mc_run_dir(run_dir)
    sims_df.to_csv(mc_run_dir / "mc_simulations.csv", index=False)
    if sample_paths:
        sp_rows = []
        for sim_id, p in sample_paths:
            for trade_n, equity in enumerate(p):
                sp_rows.append({"sim": sim_id, "trade_n": int(trade_n), "equity": float(equity)})
        pd.DataFrame(sp_rows).to_csv(mc_run_dir / "mc_paths_sample.csv", index=False)
    if quantile_matrix is not None:
        qs = np.quantile(quantile_matrix, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        qdf = pd.DataFrame(
            {
                "trade_n": np.arange(sim_trades + 1, dtype=int),
                "q05": qs[0, :],
                "q25": qs[1, :],
                "q50": qs[2, :],
                "q75": qs[3, :],
                "q95": qs[4, :],
            }
        )
        qdf.to_csv(mc_run_dir / "mc_paths_quantiles.csv", index=False)

    risk_of_ruin_pct = float(sims_df["ruin_hit"].mean() * 100.0)
    median_max_dd_pct = _finite_median(sims_df["max_drawdown_%"])
    median_return_pct = _finite_median(sims_df["return_%"])
    median_ratio_direct = _finite_median(sims_df["return_drawdown_ratio"])
    ratio_of_medians = (
        float(median_return_pct / median_max_dd_pct)
        if np.isfinite(median_return_pct) and np.isfinite(median_max_dd_pct) and median_max_dd_pct > 0
        else float("nan")
    )

    thresholds = {
        "risk_of_ruin_pct_max": float(target_risk_of_ruin_pct_max),
        "median_max_drawdown_pct_max": float(target_median_max_dd_pct_max),
        "median_return_pct_min": float(target_median_return_pct_min),
        "return_drawdown_ratio_min": float(target_return_dd_ratio_min),
    }
    checks = {
        "risk_of_ruin_ok": bool(risk_of_ruin_pct < thresholds["risk_of_ruin_pct_max"]),
        "median_max_drawdown_ok": bool(median_max_dd_pct < thresholds["median_max_drawdown_pct_max"]),
        "median_return_ok": bool(median_return_pct > thresholds["median_return_pct_min"]),
        "return_drawdown_ratio_ok": bool(ratio_of_medians > thresholds["return_drawdown_ratio_min"]),
    }
    checks["all_passed"] = all(checks.values())

    summary = {
        "run_dir": str(run_dir),
        "mc_run_dir": str(mc_run_dir),
        "config": {
            "n_sims": int(n_sims),
            "n_trades_per_sim": int(sim_trades),
            "replace": bool(replace),
            "seed": int(seed),
            "pnl_mode": pnl_mode,
            "fixed_risk_dollars": fixed_risk_dollars,
            "initial_equity": float(initial_equity),
            "ruin_equity": float(ruin_equity),
            "stop_at_ruin": bool(stop_at_ruin),
            "save_sample_paths_count": int(save_sample_paths_count),
            "save_quantile_paths": bool(save_quantile_paths),
        },
        "metrics": {
            "risk_of_ruin_pct": risk_of_ruin_pct,
            "median_max_drawdown_%": median_max_dd_pct,
            "median_return_%": median_return_pct,
            "return_drawdown_ratio_median_of_ratios": median_ratio_direct,
            "return_drawdown_ratio_ratio_of_medians": ratio_of_medians,
            "max_pain_worst_equity_p5": _finite_quantile(sims_df["worst_equity"], 0.05),
            "max_drawdown_%_p95": _finite_quantile(sims_df["max_drawdown_%"], 0.95),
            "return_%_p05": _finite_quantile(sims_df["return_%"], 0.05),
            "return_%_p50": _finite_quantile(sims_df["return_%"], 0.50),
            "return_%_p95": _finite_quantile(sims_df["return_%"], 0.95),
        },
        "thresholds": thresholds,
        "threshold_checks": checks,
    }
    _write_json(mc_run_dir / "mc_summary.json", summary)

    print(
        "[MC] done "
        f"risk_of_ruin={risk_of_ruin_pct:.2f}% "
        f"median_dd={median_max_dd_pct:.2f}% "
        f"median_return={median_return_pct:.2f}% "
        f"ratio_of_medians={ratio_of_medians:.3f}",
        flush=True,
    )
    return mc_run_dir
