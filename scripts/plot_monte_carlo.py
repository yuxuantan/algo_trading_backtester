"""
Interactive Plotly chart for Monte Carlo run outputs.

Example:
python3 scripts/plot_monte_carlo.py \
  --mc-run-dir runs/walkforward/sma_cross_test_strat/eurusd_1h_20100101_20260209/grid_unanchored/run_ddmmyy_hhmmss/monte_carlo/run_ddmmyy_hhmmss
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def _safe_float(v) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x


def build_figure(
    *,
    sims_df: pd.DataFrame,
    summary: dict,
    sample_paths_df: pd.DataFrame | None,
    quantile_paths_df: pd.DataFrame | None,
    max_sample_lines: int,
    title: str,
):
    if go is None or make_subplots is None:
        raise RuntimeError("plotly is required for this script.")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Equity Path Envelope",
            "Return Distribution (%)",
            "Max Drawdown Distribution (%)",
            "Return vs Drawdown",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    # Panel 1: equity paths
    if quantile_paths_df is not None and not quantile_paths_df.empty:
        q = quantile_paths_df
        fig.add_trace(
            go.Scatter(
                x=q["trade_n"],
                y=q["q95"],
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                name="q95",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=q["trade_n"],
                y=q["q05"],
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.18)",
                line={"width": 0},
                name="5-95% band",
                hovertemplate="Trade=%{x}<br>q05=%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=q["trade_n"],
                y=q["q50"],
                mode="lines",
                line={"width": 2.4, "color": "#1d4ed8"},
                name="Median Path",
                hovertemplate="Trade=%{x}<br>Median Equity=%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if sample_paths_df is not None and not sample_paths_df.empty:
        sims = sorted(sample_paths_df["sim"].dropna().astype(int).unique().tolist())[:max_sample_lines]
        for sim_id in sims:
            d = sample_paths_df[sample_paths_df["sim"] == sim_id]
            fig.add_trace(
                go.Scattergl(
                    x=d["trade_n"],
                    y=d["equity"],
                    mode="lines",
                    line={"width": 1, "color": "rgba(30, 41, 59, 0.18)"},
                    name="Sample Paths",
                    showlegend=(sim_id == sims[0]),
                    hovertemplate=f"Sim={sim_id}<br>Trade=%{{x}}<br>Equity=%{{y:,.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Panel 2: return distribution
    ret = pd.to_numeric(sims_df["return_%"], errors="coerce").dropna()
    fig.add_trace(
        go.Histogram(
            x=ret,
            nbinsx=80,
            marker={"color": "rgba(16,185,129,0.6)"},
            name="Return %",
            hovertemplate="Return=%{x:.2f}%<br>Count=%{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    ret_med = float(ret.median()) if not ret.empty else np.nan
    if np.isfinite(ret_med):
        fig.add_vline(x=ret_med, line_width=2, line_dash="dash", line_color="#047857", row=1, col=2)

    # Panel 3: drawdown distribution
    dd = pd.to_numeric(sims_df["max_drawdown_%"], errors="coerce").dropna()
    fig.add_trace(
        go.Histogram(
            x=dd,
            nbinsx=80,
            marker={"color": "rgba(239,68,68,0.6)"},
            name="Max DD %",
            hovertemplate="Max DD=%{x:.2f}%<br>Count=%{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    dd_med = float(dd.median()) if not dd.empty else np.nan
    if np.isfinite(dd_med):
        fig.add_vline(x=dd_med, line_width=2, line_dash="dash", line_color="#b91c1c", row=2, col=1)

    # Panel 4: return vs drawdown
    sc = sims_df.copy()
    sc["return_%"] = pd.to_numeric(sc["return_%"], errors="coerce")
    sc["max_drawdown_%"] = pd.to_numeric(sc["max_drawdown_%"], errors="coerce")
    sc = sc.dropna(subset=["return_%", "max_drawdown_%"])
    sc["ruin_hit"] = sc["ruin_hit"].fillna(False).astype(bool)
    colors = np.where(sc["ruin_hit"], "#dc2626", "#2563eb")
    fig.add_trace(
        go.Scattergl(
            x=sc["max_drawdown_%"],
            y=sc["return_%"],
            mode="markers",
            marker={"size": 6, "color": colors, "opacity": 0.65},
            customdata=sc[["sim", "ruin_hit", "final_equity"]].to_numpy(),
            name="Sim Outcomes",
            hovertemplate=(
                "Sim=%{customdata[0]}<br>"
                "Max DD=%{x:.2f}%<br>"
                "Return=%{y:.2f}%<br>"
                "Ruin=%{customdata[1]}<br>"
                "Final Equity=%{customdata[2]:,.2f}<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )

    thresholds = summary.get("thresholds", {})
    dd_cap = _safe_float(thresholds.get("median_max_drawdown_pct_max"))
    ret_floor = _safe_float(thresholds.get("median_return_pct_min"))
    if np.isfinite(dd_cap):
        fig.add_vline(x=dd_cap, line_width=1.5, line_dash="dot", line_color="#7f1d1d", row=2, col=2)
    if np.isfinite(ret_floor):
        fig.add_hline(y=ret_floor, line_width=1.5, line_dash="dot", line_color="#14532d", row=2, col=2)

    metrics = summary.get("metrics", {})
    subtitle = (
        f"Risk of Ruin={_safe_float(metrics.get('risk_of_ruin_pct')):.2f}% | "
        f"Median DD={_safe_float(metrics.get('median_max_drawdown_%')):.2f}% | "
        f"Median Return={_safe_float(metrics.get('median_return_%')):.2f}% | "
        f"Median CAGR={_safe_float(metrics.get('median_cagr_%')):.2f}% | "
        f"Median Sortino={_safe_float(metrics.get('median_sortino')):.3f} | "
        f"Median MAR={_safe_float(metrics.get('median_mar')):.3f} | "
        f"R/DD (ratio of medians)={_safe_float(metrics.get('return_drawdown_ratio_ratio_of_medians')):.3f}"
    )

    fig.update_layout(
        template="plotly_white",
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.01},
        barmode="overlay",
        bargap=0.05,
        legend={"orientation": "h", "y": 1.12, "x": 0.0},
        margin={"l": 70, "r": 40, "t": 100, "b": 60},
    )

    fig.update_xaxes(title_text="Trade Number", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_xaxes(title_text="Return %", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Max Drawdown %", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Max Drawdown %", row=2, col=2)
    fig.update_yaxes(title_text="Return %", row=2, col=2)

    return fig


def main():
    parser = argparse.ArgumentParser("Plot interactive chart for Monte Carlo run.")
    parser.add_argument("--mc-run-dir", required=True, help="Path to monte_carlo/run_x folder.")
    parser.add_argument("--output", default=None, help="Output HTML path. Default: <mc-run-dir>/mc_interactive.html")
    parser.add_argument("--title", default=None, help="Chart title.")
    parser.add_argument("--max-sample-lines", type=int, default=40)
    parser.add_argument("--show", action="store_true", help="Open interactive window after writing HTML.")
    args = parser.parse_args()

    if go is None:
        raise SystemExit("plotly is not installed. Install it with: pip install plotly")

    mc_dir = Path(args.mc_run_dir)
    sims_path = mc_dir / "mc_simulations.csv"
    summary_path = mc_dir / "mc_summary.json"
    sample_path = mc_dir / "mc_paths_sample.csv"
    quantiles_path = mc_dir / "mc_paths_quantiles.csv"

    if not sims_path.exists():
        raise ValueError(f"missing mc_simulations.csv in {mc_dir}")

    sims_df = pd.read_csv(sims_path)
    summary = _read_json(summary_path)
    sample_df = pd.read_csv(sample_path) if sample_path.exists() else None
    quant_df = pd.read_csv(quantiles_path) if quantiles_path.exists() else None

    title = args.title or f"Monte Carlo - {mc_dir.name}"
    fig = build_figure(
        sims_df=sims_df,
        summary=summary,
        sample_paths_df=sample_df,
        quantile_paths_df=quant_df,
        max_sample_lines=max(0, int(args.max_sample_lines)),
        title=title,
    )

    output = Path(args.output) if args.output else mc_dir / "mc_interactive.html"
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    print(f"Saved interactive chart: {output}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
