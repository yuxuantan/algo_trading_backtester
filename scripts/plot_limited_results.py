"""
Interactive Plotly chart for limited test outputs.

Example:
python3 scripts/plot_limited_results.py \
  --run-dir runs/limited/sma_cross_test_strat/eurusd_1h_20100101_20130101/core_system_test__sma_cross__fixed_atr_exit/run_ddmmyy_hhmmss
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[name], errors="coerce").dropna()


def build_figure(*, results: pd.DataFrame, pass_summary: dict, title: str):
    if go is None or make_subplots is None:
        raise RuntimeError("plotly is required for this script.")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Return Distribution (%)",
            "Max Drawdown Distribution (%)",
            "Trade Count Distribution",
            "Return vs Max Drawdown",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    ret = _series(results, "total_return_%")
    if not ret.empty:
        fig.add_trace(
            go.Histogram(
                x=ret,
                nbinsx=80,
                marker={"color": "rgba(16,185,129,0.60)"},
                name="total_return_%",
                hovertemplate="Return=%{x:.2f}%<br>Count=%{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_vline(x=float(ret.median()), line_dash="dash", line_color="#047857", row=1, col=1)

    dd = _series(results, "max_drawdown_abs_%")
    if dd.empty:
        dd = _series(results, "max_drawdown_%")
    if not dd.empty:
        fig.add_trace(
            go.Histogram(
                x=dd,
                nbinsx=80,
                marker={"color": "rgba(239,68,68,0.60)"},
                name="max_drawdown_%",
                hovertemplate="Max DD=%{x:.2f}%<br>Count=%{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.add_vline(x=float(dd.median()), line_dash="dash", line_color="#b91c1c", row=1, col=2)

    trades = _series(results, "trades")
    if not trades.empty:
        fig.add_trace(
            go.Histogram(
                x=trades,
                nbinsx=80,
                marker={"color": "rgba(59,130,246,0.60)"},
                name="trades",
                hovertemplate="Trades=%{x:.0f}<br>Count=%{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if not ret.empty and not dd.empty and {"total_return_%", "favourable"}.issubset(results.columns):
        sc = results.copy()
        sc["total_return_%"] = pd.to_numeric(sc["total_return_%"], errors="coerce")
        if "max_drawdown_abs_%" in sc.columns:
            sc["dd_col"] = pd.to_numeric(sc["max_drawdown_abs_%"], errors="coerce")
        elif "max_drawdown_%" in sc.columns:
            sc["dd_col"] = pd.to_numeric(sc["max_drawdown_%"], errors="coerce")
        else:
            sc["dd_col"] = pd.NA
        sc = sc.dropna(subset=["total_return_%", "dd_col"])
        colors = sc["favourable"].fillna(False).map({True: "#2563eb", False: "#dc2626"})
        fig.add_trace(
            go.Scattergl(
                x=sc["dd_col"],
                y=sc["total_return_%"],
                mode="markers",
                marker={"size": 6, "color": colors, "opacity": 0.70},
                customdata=sc[["iter", "favourable"]].to_numpy() if {"iter", "favourable"}.issubset(sc.columns) else None,
                name="Iterations",
                hovertemplate=(
                    "Iter=%{customdata[0]}<br>"
                    "Favourable=%{customdata[1]}<br>"
                    "Max DD=%{x:.2f}%<br>"
                    "Return=%{y:.2f}%<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

    favourable_pct = float(pass_summary.get("favourable_pct", float("nan")))
    pass_threshold = float(pass_summary.get("pass_threshold_%", float("nan")))
    passed = bool(pass_summary.get("passed", False))
    win_med = _series(results, "win_rate_%").median() if "win_rate_%" in results.columns else float("nan")
    cagr_med = _series(results, "cagr_%").median() if "cagr_%" in results.columns else float("nan")
    mar_med = _series(results, "mar").median() if "mar" in results.columns else float("nan")
    sortino_med = _series(results, "sortino").median() if "sortino" in results.columns else float("nan")

    subtitle = (
        f"Favourable={favourable_pct:.2f}% | "
        f"Threshold={pass_threshold:.2f}% | "
        f"Status={'PASS' if passed else 'FAIL'} | "
        f"Median WinRate={win_med:.2f}% | "
        f"Median CAGR={cagr_med:.2f}% | "
        f"Median MAR={mar_med:.3f} | "
        f"Median Sortino={sortino_med:.3f}"
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.01},
        barmode="overlay",
        bargap=0.05,
        legend={"orientation": "h", "y": 1.12, "x": 0.0},
        margin={"l": 70, "r": 40, "t": 100, "b": 60},
    )

    fig.update_xaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Max Drawdown %", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Trades", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Max Drawdown %", row=2, col=2)
    fig.update_yaxes(title_text="Return %", row=2, col=2)
    return fig


def main():
    parser = argparse.ArgumentParser("Plot interactive chart for limited test run.")
    parser.add_argument("--run-dir", required=True, help="Path to limited test run_x folder.")
    parser.add_argument("--output", default=None, help="Output HTML path. Default: <run-dir>/limited_interactive.html")
    parser.add_argument("--title", default=None, help="Chart title.")
    parser.add_argument("--show", action="store_true", help="Open interactive window after writing HTML.")
    args = parser.parse_args()

    if go is None:
        raise SystemExit("plotly is not installed. Install it with: pip install plotly")

    run_dir = Path(args.run_dir)
    results_path = run_dir / "limited_results.csv"
    pass_path = run_dir / "pass_summary.json"
    if not results_path.exists():
        raise ValueError(f"missing limited_results.csv in {run_dir}")

    results = pd.read_csv(results_path)
    pass_summary = _read_json(pass_path)

    title = args.title or f"Limited Test - {run_dir.name}"
    fig = build_figure(results=results, pass_summary=pass_summary, title=title)

    output = Path(args.output) if args.output else run_dir / "limited_interactive.html"
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    print(f"Saved interactive chart: {output}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
