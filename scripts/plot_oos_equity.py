"""
Interactive OOS equity curve plotter for walk-forward runs.

Example:
python3 scripts/plot_oos_equity.py \
  --run-dir runs/walkforward/sma_cross_test_strat/eurusd_1h_20100101_20260209/grid_unanchored/run_ddmmyy_hhmmss
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _read_equity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns or "equity" not in df.columns:
        raise ValueError(f"{path} must contain columns: time, equity")
    df["time"] = _to_utc(df["time"])
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["time", "equity"]).sort_values("time").reset_index(drop=True)
    return df


def _read_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"entry_time", "exit_time", "side", "entry", "exit", "pnl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["entry_time"] = _to_utc(df["entry_time"])
    df["exit_time"] = _to_utc(df["exit_time"])
    for col in ("entry", "exit", "pnl", "commission", "r_multiple", "equity_after"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["entry_time", "exit_time"]).reset_index(drop=True)


def _equity_at_times(eq_df: pd.DataFrame, ts: pd.Series) -> pd.Series:
    probe = pd.DataFrame({"_idx": range(len(ts)), "time": _to_utc(ts)})
    probe = probe.dropna(subset=["time"]).sort_values("time")
    merged = pd.merge_asof(
        probe,
        eq_df[["time", "equity"]].sort_values("time"),
        on="time",
        direction="nearest",
    )
    out = pd.Series(index=range(len(ts)), dtype="float64")
    out.loc[merged["_idx"].to_numpy()] = merged["equity"].to_numpy()
    return out


def _read_schedule(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        start = pd.to_datetime(item.get("oos_start_time"), utc=True, errors="coerce")
        end = pd.to_datetime(item.get("oos_end_time_exclusive"), utc=True, errors="coerce")
        cleaned.append({
            "fold": item.get("fold"),
            "start": start,
            "end": end,
            "params": item.get("params", {}),
        })
    return cleaned


def build_figure(*, equity_df: pd.DataFrame, trades_df: pd.DataFrame, schedule: list[dict], title: str):
    if go is None:
        raise RuntimeError("plotly is required for this script.")

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=equity_df["time"],
            y=equity_df["equity"],
            mode="lines",
            name="OOS Equity",
            line={"width": 2, "color": "#1f77b4"},
            hovertemplate="Time=%{x}<br>Equity=%{y:,.2f}<extra></extra>",
        )
    )

    if not trades_df.empty:
        exits = trades_df.copy()
        exits["equity_at_exit"] = _equity_at_times(equity_df, exits["exit_time"])
        exits["side_color"] = exits["side"].map({"long": "#16a34a", "short": "#dc2626"}).fillna("#374151")
        exit_custom = exits[
            [
                "side",
                "entry_time",
                "exit_time",
                "entry",
                "exit",
                "pnl",
                "commission",
                "r_multiple",
                "exit_reason",
            ]
        ].fillna("")
        fig.add_trace(
            go.Scatter(
                x=exits["exit_time"],
                y=exits["equity_at_exit"],
                mode="markers",
                name="Trade Exits",
                marker={
                    "size": 7,
                    "symbol": "triangle-down",
                    "color": exits["side_color"],
                    "line": {"width": 0},
                },
                customdata=exit_custom.to_numpy(),
                hovertemplate=(
                    "Side=%{customdata[0]}<br>"
                    "Entry Time=%{customdata[1]}<br>"
                    "Exit Time=%{customdata[2]}<br>"
                    "Entry=%{customdata[3]:,.5f}<br>"
                    "Exit=%{customdata[4]:,.5f}<br>"
                    "PnL=%{customdata[5]:,.2f}<br>"
                    "Commission=%{customdata[6]:,.2f}<br>"
                    "R Multiple=%{customdata[7]:,.3f}<br>"
                    "Reason=%{customdata[8]}<extra></extra>"
                ),
            )
        )

        entries = trades_df.copy()
        entries["equity_at_entry"] = _equity_at_times(equity_df, entries["entry_time"])
        entries["side_color"] = entries["side"].map({"long": "#22c55e", "short": "#ef4444"}).fillna("#6b7280")
        entry_custom = entries[
            ["side", "entry_time", "entry", "units", "sl", "tp"]
        ].fillna("")
        fig.add_trace(
            go.Scatter(
                x=entries["entry_time"],
                y=entries["equity_at_entry"],
                mode="markers",
                name="Trade Entries",
                marker={
                    "size": 6,
                    "symbol": "triangle-up",
                    "color": entries["side_color"],
                    "opacity": 0.55,
                    "line": {"width": 0},
                },
                customdata=entry_custom.to_numpy(),
                hovertemplate=(
                    "Side=%{customdata[0]}<br>"
                    "Entry Time=%{customdata[1]}<br>"
                    "Entry=%{customdata[2]:,.5f}<br>"
                    "Units=%{customdata[3]:,.0f}<br>"
                    "SL=%{customdata[4]:,.5f}<br>"
                    "TP=%{customdata[5]:,.5f}<extra></extra>"
                ),
            )
        )

    eq_min = float(equity_df["equity"].min())
    eq_max = float(equity_df["equity"].max())
    y_hover = eq_max if eq_max > eq_min else eq_max + 1.0
    fold_hover_rows = []
    for i, item in enumerate(schedule):
        start = item["start"]
        end = item["end"]
        if pd.isna(start):
            continue
        if pd.isna(end):
            end = equity_df["time"].iloc[-1]
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(148, 163, 184, 0.10)" if i % 2 == 0 else "rgba(59, 130, 246, 0.08)",
            line_width=0,
            layer="below",
        )
        midpoint = start + (end - start) / 2 if end >= start else start
        fold_hover_rows.append(
            {
                "x": midpoint,
                "fold": item.get("fold"),
                "start": start,
                "end": end,
                "params": json.dumps(item.get("params", {}), sort_keys=True),
            }
        )

    if fold_hover_rows:
        fold_df = pd.DataFrame(fold_hover_rows)
        fig.add_trace(
            go.Scatter(
                x=fold_df["x"],
                y=[y_hover] * len(fold_df),
                mode="markers",
                name="Fold Windows",
                marker={"size": 14, "opacity": 0.0, "color": "#000000"},
                customdata=fold_df[["fold", "start", "end", "params"]].to_numpy(),
                hovertemplate=(
                    "Fold=%{customdata[0]}<br>"
                    "OOS Start=%{customdata[1]}<br>"
                    "OOS End=%{customdata[2]}<br>"
                    "Params=%{customdata[3]}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Equity",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
    )
    return fig


def main():
    parser = argparse.ArgumentParser("Plot interactive OOS equity curve for a walk-forward run.")
    parser.add_argument("--run-dir", required=True, help="Walk-forward run directory.")
    parser.add_argument("--equity-file", default="oos_equity_curve.csv")
    parser.add_argument("--trades-file", default="oos_trades.csv")
    parser.add_argument("--schedule-file", default="walkforward_param_schedule.json")
    parser.add_argument("--output", default=None, help="Output HTML path. Default: <run-dir>/oos_equity_interactive.html")
    parser.add_argument("--title", default=None, help="Chart title.")
    parser.add_argument("--show", action="store_true", help="Open interactive window after writing HTML.")
    args = parser.parse_args()
    if go is None:
        raise SystemExit("plotly is not installed. Install it with: pip install plotly")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"run directory does not exist: {run_dir}")

    equity_path = run_dir / args.equity_file
    trades_path = run_dir / args.trades_file
    schedule_path = run_dir / args.schedule_file

    equity_df = _read_equity(equity_path)
    trades_df = _read_trades(trades_path)
    schedule = _read_schedule(schedule_path)

    title = args.title or f"OOS Equity Curve - {run_dir.name}"
    fig = build_figure(equity_df=equity_df, trades_df=trades_df, schedule=schedule, title=title)

    output_path = Path(args.output) if args.output else run_dir / "oos_equity_interactive.html"
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved interactive chart: {output_path}")

    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
