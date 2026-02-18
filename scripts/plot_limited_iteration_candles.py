"""
Interactive candlestick overlay for a single limited-test iteration.

Overlays:
- OHLC candlesticks from the run dataset
- Iteration trades (entry/exit markers, SL/TP brackets)
- InterEquity liquidity line annotations reconstructed from strategy events
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from quantbt.core.engine import BacktestConfig
from quantbt.experiments.limited.data_prep import load_price_frame

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_COLORS = {
    0: "#111827",  # black-ish
    1: "#a855f7",  # purple
    2: "#dc2626",  # red
}
STATE_LABEL = {
    0: "black",
    1: "purple",
    2: "red",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_params_cell(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _to_ts_utc(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _resolve_strategy_module(run_meta: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit

    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    test_cfg = spec.get("test", {}) if isinstance(spec, dict) else {}
    strategy_module = test_cfg.get("strategy_module")
    if strategy_module:
        return str(strategy_module)

    entry_rules = (
        spec.get("strategy", {})
        .get("entry", {})
        .get("rules", [])
    )
    if isinstance(entry_rules, list):
        for rule in entry_rules:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("name", "")).strip() == "interequity_liqsweep_entry":
                return "quantbt.strategies.InterEquity2026-02 LiqSweepA"

    strategy_tag = str(test_cfg.get("strategy_tag", "")).strip()
    if strategy_tag:
        return f"quantbt.strategies.{strategy_tag}"

    raise ValueError(
        "Unable to resolve strategy module from run metadata. "
        "Pass --strategy-module explicitly."
    )


def _build_strategy_params(mod: Any, run_meta: dict[str, Any], iter_row: pd.Series) -> Any:
    params_cls = getattr(mod, "Params", None)
    if params_cls is None:
        raise ValueError("Strategy module missing Params dataclass/class.")

    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    strategy_cfg = spec.get("strategy", {}) if isinstance(spec, dict) else {}
    default_entry = (
        strategy_cfg.get("entry", {})
        .get("rules", [{}])[0]
        .get("params", {})
    )
    if not isinstance(default_entry, dict):
        default_entry = {}

    entry_params = default_entry.copy()
    parsed_cell = _parse_params_cell(iter_row.get("entry_params"))
    if isinstance(parsed_cell, list) and parsed_cell:
        first = parsed_cell[0]
        if isinstance(first, dict):
            entry_params.update(first)
    elif isinstance(parsed_cell, dict):
        entry_params.update(parsed_cell)

    return params_cls(**entry_params)


def _build_backtest_cfg(run_meta: dict[str, Any]) -> BacktestConfig:
    cfg_raw = run_meta.get("spec", {}).get("config", {})
    if not isinstance(cfg_raw, dict):
        return BacktestConfig()

    valid_fields = set(BacktestConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg_raw.items() if k in valid_fields}
    return BacktestConfig(**filtered)


def _segments_from_events(events: list[dict[str, Any]], fallback_end: pd.Timestamp) -> list[dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    segments: list[dict[str, Any]] = []

    events_sorted = sorted(
        events,
        key=lambda e: (
            _to_ts_utc(e.get("time")) or pd.Timestamp.min.tz_localize("UTC"),
            {"line_created": 0, "line_promoted_red": 1, "line_deactivated": 2}.get(str(e.get("type")), 99),
        ),
    )

    for ev in events_sorted:
        ev_type = str(ev.get("type", ""))
        pool = str(ev.get("pool", ""))
        line_id = int(ev.get("line_id", -1))
        t = _to_ts_utc(ev.get("time"))
        if line_id < 0 or t is None:
            continue

        key = (pool, line_id)
        if ev_type == "line_created":
            lvl = float(ev.get("level"))
            st = int(ev.get("state", 0))
            records[key] = {
                "pool": pool,
                "line_id": line_id,
                "level": lvl,
                "start": t,
                "state": st,
                "changes": [],
                "end": None,
            }
            continue

        rec = records.get(key)
        if rec is None:
            continue

        if ev_type == "line_promoted_red":
            rec["changes"].append((t, 2))
        elif ev_type == "line_deactivated":
            end = rec.get("end")
            rec["end"] = t if end is None else min(pd.Timestamp(end), t)

    for rec in records.values():
        start = pd.Timestamp(rec["start"])
        end = pd.Timestamp(rec["end"]) if rec.get("end") is not None else pd.Timestamp(fallback_end)
        if end < start:
            continue

        current_state = int(rec["state"])
        seg_start = start
        for change_time, new_state in sorted(rec["changes"], key=lambda x: x[0]):
            ct = pd.Timestamp(change_time)
            if ct <= seg_start:
                current_state = int(new_state)
                continue
            seg_end = min(ct, end)
            if seg_end > seg_start:
                segments.append(
                    {
                        "pool": rec["pool"],
                        "line_id": rec["line_id"],
                        "state": current_state,
                        "level": float(rec["level"]),
                        "start": seg_start,
                        "end": seg_end,
                    }
                )
            seg_start = ct
            current_state = int(new_state)
            if seg_start >= end:
                break

        if seg_start < end:
            segments.append(
                {
                    "pool": rec["pool"],
                    "line_id": rec["line_id"],
                    "state": current_state,
                    "level": float(rec["level"]),
                    "start": seg_start,
                    "end": end,
                }
            )

    return segments


def _infer_bar_seconds(index: pd.Index) -> int:
    if len(index) < 2:
        return 300
    ts = pd.to_datetime(index, utc=True, errors="coerce")
    diffs = ts.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 300
    med = float(diffs.median())
    if not math.isfinite(med) or med <= 0:
        return 300
    return int(round(med))


def _slice_chart_window(
    df: pd.DataFrame,
    trades_iter: pd.DataFrame,
    *,
    window_mode: str,
    context_bars: int,
) -> pd.DataFrame:
    if window_mode == "full" or trades_iter.empty:
        return df

    bar_seconds = _infer_bar_seconds(df.index)
    pad = pd.Timedelta(seconds=bar_seconds * max(int(context_bars), 0))

    et = pd.to_datetime(trades_iter["entry_time"], utc=True, errors="coerce")
    xt = pd.to_datetime(trades_iter["exit_time"], utc=True, errors="coerce")
    t0 = pd.concat([et, xt], axis=0).min()
    t1 = pd.concat([et, xt], axis=0).max()
    if pd.isna(t0) or pd.isna(t1):
        return df

    start = pd.Timestamp(t0) - pad
    end = pd.Timestamp(t1) + pad
    out = df.loc[(df.index >= start) & (df.index <= end)].copy()
    return out if not out.empty else df


def _build_figure(
    *,
    chart_df: pd.DataFrame,
    trades_iter: pd.DataFrame,
    line_segments: list[dict[str, Any]],
    title: str,
) -> Any:
    if go is None:
        raise RuntimeError("plotly is required for this script.")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
        )
    )

    # Legend keys for states
    for st, label in STATE_LABEL.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": STATE_COLORS.get(st, "#6b7280"), "width": 2},
                name=f"Liquidity {label}",
                showlegend=True,
            )
        )

    window_start = pd.Timestamp(chart_df.index.min())
    window_end = pd.Timestamp(chart_df.index.max())

    for seg in line_segments:
        s0 = pd.Timestamp(seg["start"])
        s1 = pd.Timestamp(seg["end"])
        if s1 < window_start or s0 > window_end:
            continue
        x0 = max(s0, window_start)
        x1 = min(s1, window_end)
        if x1 <= x0:
            continue

        pool = str(seg.get("pool", ""))
        width = 2 if pool.startswith("htf_") else 1
        fig.add_trace(
            go.Scattergl(
                x=[x0, x1],
                y=[seg["level"], seg["level"]],
                mode="lines",
                line={
                    "color": STATE_COLORS.get(int(seg["state"]), "#6b7280"),
                    "width": width,
                },
                hovertemplate=(
                    f"Pool={pool}<br>"
                    f"State={STATE_LABEL.get(int(seg['state']), seg['state'])}<br>"
                    "Level=%{y:.5f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    if not trades_iter.empty:
        tr = trades_iter.copy()
        tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True, errors="coerce")
        tr["exit_time"] = pd.to_datetime(tr["exit_time"], utc=True, errors="coerce")
        tr = tr.dropna(subset=["entry_time", "exit_time"])

        for side, color, entry_symbol, exit_symbol in (
            ("long", "#16a34a", "triangle-up", "x"),
            ("short", "#dc2626", "triangle-down", "x"),
        ):
            side_df = tr.loc[tr["side"].astype(str) == side]
            if side_df.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=side_df["entry_time"],
                    y=pd.to_numeric(side_df["entry"], errors="coerce"),
                    mode="markers",
                    marker={"color": color, "size": 9, "symbol": entry_symbol},
                    name=f"{side} entry",
                    hovertemplate="Entry<br>%{x}<br>Px=%{y:.5f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=side_df["exit_time"],
                    y=pd.to_numeric(side_df["exit"], errors="coerce"),
                    mode="markers",
                    marker={"color": color, "size": 9, "symbol": exit_symbol},
                    name=f"{side} exit",
                    hovertemplate="Exit<br>%{x}<br>Px=%{y:.5f}<extra></extra>",
                )
            )

        # Trade bracket overlays
        for _, row in tr.iterrows():
            et = row.get("entry_time")
            xt = row.get("exit_time")
            if pd.isna(et) or pd.isna(xt):
                continue
            x0 = max(pd.Timestamp(et), window_start)
            x1 = min(pd.Timestamp(xt), window_end)
            if x1 <= x0:
                continue

            sl = pd.to_numeric(pd.Series([row.get("sl")]), errors="coerce").iloc[0]
            tp = pd.to_numeric(pd.Series([row.get("tp")]), errors="coerce").iloc[0]

            if math.isfinite(float(sl)):
                fig.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=float(sl),
                    y1=float(sl),
                    line={"color": "#2563eb", "width": 1, "dash": "dot"},
                )
            if math.isfinite(float(tp)):
                fig.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=float(tp),
                    y1=float(tp),
                    line={"color": "#16a34a", "width": 1, "dash": "dot"},
                )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser("Plot limited-test iteration candle overlays.")
    parser.add_argument("--run-dir", required=True, help="Path to limited run directory.")
    parser.add_argument("--iter", required=True, type=int, help="Iteration id from limited_results.csv.")
    parser.add_argument(
        "--strategy-module",
        default=None,
        help="Optional explicit strategy module path (e.g. quantbt.strategies.InterEquity2026-02 LiqSweepA).",
    )
    parser.add_argument(
        "--window-mode",
        choices=["trade_span", "full"],
        default="trade_span",
        help="trade_span: chart around iteration trades (+context bars). full: full dataset.",
    )
    parser.add_argument(
        "--context-bars",
        type=int,
        default=400,
        help="Context bars before first and after last trade when window-mode=trade_span.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Default: <run-dir>/iteration_<iter>_candles.html",
    )
    parser.add_argument("--show", action="store_true", help="Open interactive chart after write.")
    args = parser.parse_args()

    if go is None:
        raise SystemExit("plotly is not installed. Install it with: pip install plotly")

    run_dir = Path(args.run_dir)
    results_path = run_dir / "limited_results.csv"
    trades_path = run_dir / "limited_trades.csv"
    meta_path = run_dir / "run_meta.json"
    if not results_path.exists() or not meta_path.exists():
        raise ValueError("Missing required run files: limited_results.csv and/or run_meta.json.")

    results = pd.read_csv(results_path)
    if "iter" not in results.columns:
        raise ValueError("limited_results.csv is missing 'iter' column.")

    sel = results.loc[pd.to_numeric(results["iter"], errors="coerce") == int(args.iter)]
    if sel.empty:
        raise ValueError(f"Iteration {args.iter} not found in limited_results.csv.")
    iter_row = sel.iloc[0]

    run_meta = _read_json(meta_path)
    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    data_path_raw = spec.get("data")
    ts_col = str(spec.get("ts_col", "timestamp"))
    if not data_path_raw:
        raise ValueError("run_meta.json is missing spec.data")

    data_path = Path(data_path_raw)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()

    df = load_price_frame(data_path, ts_col=ts_col)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()

    trades_iter = pd.DataFrame()
    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        if "iter" in trades.columns:
            trades_iter = trades.loc[pd.to_numeric(trades["iter"], errors="coerce") == int(args.iter)].copy()

    strategy_module = _resolve_strategy_module(run_meta, args.strategy_module)
    mod = importlib.import_module(strategy_module)
    strategy_params = _build_strategy_params(mod, run_meta, iter_row)
    cfg = _build_backtest_cfg(run_meta)

    df_feat = mod.compute_features(df, strategy_params)
    debug: dict[str, Any] = {}
    mod.run_backtest(df_feat, strategy_params=strategy_params, cfg=cfg, debug=debug)
    line_events = debug.get("line_events", [])

    chart_df = _slice_chart_window(
        df_feat,
        trades_iter,
        window_mode=args.window_mode,
        context_bars=args.context_bars,
    )
    if chart_df.empty:
        raise ValueError("No chart data found for selected window.")

    line_segments = _segments_from_events(line_events, fallback_end=pd.Timestamp(df_feat.index.max()))

    title = (
        f"Limited Iteration {int(args.iter)} "
        f"| Strategy={strategy_module} "
        f"| Bars={len(chart_df)}"
    )
    fig = _build_figure(
        chart_df=chart_df,
        trades_iter=trades_iter,
        line_segments=line_segments,
        title=title,
    )

    out_path = Path(args.output) if args.output else (run_dir / f"iteration_{int(args.iter)}_candles.html")
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved iteration chart: {out_path}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
