"""
TradingView-like chart renderer for one limited-test iteration using
Lightweight Charts (official TradingView OSS chart library).

Overlays:
- Candlesticks from run dataset
- Entry/exit markers
- SL/TP trade brackets
- Liquidity line segments (black/purple/red) reconstructed from strategy events
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


def _to_unix_s(value: Any) -> int | None:
    ts = _to_ts_utc(value)
    if ts is None:
        return None
    return int(ts.timestamp())


def _resolve_strategy_module(run_meta: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit

    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    test_cfg = spec.get("test", {}) if isinstance(spec, dict) else {}
    strategy_module = test_cfg.get("strategy_module")
    if strategy_module:
        return str(strategy_module)

    entry_rules = spec.get("strategy", {}).get("entry", {}).get("rules", [])
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
    default_entry = strategy_cfg.get("entry", {}).get("rules", [{}])[0].get("params", {})
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
                        "origin_start": start,
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
                    "origin_start": start,
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


def _initial_window_bounds(
    chart_df: pd.DataFrame,
    trades_iter: pd.DataFrame,
    *,
    initial_bars: int,
) -> tuple[int, int]:
    n = len(chart_df)
    if n == 0:
        return 0, 0
    if initial_bars <= 0 or n <= initial_bars:
        return 0, n

    focus_time = chart_df.index[-1]
    if not trades_iter.empty:
        xt = pd.to_datetime(trades_iter.get("exit_time"), utc=True, errors="coerce")
        et = pd.to_datetime(trades_iter.get("entry_time"), utc=True, errors="coerce")
        t = pd.concat([xt, et], axis=0).dropna()
        if not t.empty:
            focus_time = pd.Timestamp(t.max())

    idx = chart_df.index
    pos = int(idx.searchsorted(focus_time, side="left"))
    half = initial_bars // 2
    start_i = max(0, pos - half)
    end_i = min(n, start_i + initial_bars)
    start_i = max(0, end_i - initial_bars)
    return start_i, end_i


def _prepare_liq_segments(
    line_segments: list[dict[str, Any]],
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    include_black_lines: bool,
    max_liq_lines: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for seg in line_segments:
        state = int(seg.get("state", 0))
        if state == 0 and not include_black_lines:
            continue

        s0 = pd.Timestamp(seg.get("start"))
        s1 = pd.Timestamp(seg.get("end"))
        s_draw = pd.Timestamp(seg.get("origin_start", s0))
        if s1 < window_start or s0 > window_end:
            continue

        x1 = min(s1, window_end)
        if x1 <= s0:
            continue

        item = dict(seg)
        item["start"] = s_draw
        item["active_start"] = s0
        item["end"] = x1
        filtered.append(item)

    # Prefer stronger liquidity states and more recent segments.
    priority = {2: 0, 1: 1, 0: 2}
    filtered.sort(
        key=lambda x: (
            priority.get(int(x.get("state", 0)), 9),
            -int(pd.Timestamp(x.get("end")).timestamp()),
        )
    )
    if max_liq_lines > 0 and len(filtered) > max_liq_lines:
        filtered = filtered[:max_liq_lines]
    return filtered


def _table_value(value: Any) -> str:
    if value is None:
        return ""

    # Keep numeric fields numeric; do not coerce into epoch timestamps.
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8f}".rstrip("0").rstrip(".")

    if isinstance(value, pd.Timestamp):
        ts = _to_ts_utc(value)
        return "" if ts is None else ts.isoformat()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        return text

    return str(value)


def _table_value_for_col(value: Any, col_name: str) -> str:
    text = _table_value(value)
    if not text:
        return text
    c = str(col_name).strip().lower()
    is_time_col = c == "time" or c.endswith("_time") or c.endswith("time")
    if not is_time_col:
        return text
    ts = _to_ts_utc(text)
    return text if ts is None else ts.isoformat()


def _infer_price_precision(df: pd.DataFrame) -> int:
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if not cols:
        return 5
    vals = pd.to_numeric(df[cols].stack(), errors="coerce").dropna()
    if vals.empty:
        return 5
    sample = vals.head(min(5000, len(vals)))
    max_dec = 0
    for v in sample:
        s = f"{float(v):.10f}".rstrip("0").rstrip(".")
        if "." in s:
            dec = len(s.rsplit(".", 1)[1])
            if dec > max_dec:
                max_dec = dec
                if max_dec >= 8:
                    break
    return max(2, min(8, max_dec if max_dec > 0 else 5))


def _normalize_trades_for_plot(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    if "entry_time" in out.columns:
        out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    if "exit_time" in out.columns:
        out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    if "entry" in out.columns:
        out["entry"] = pd.to_numeric(out["entry"], errors="coerce")
    if "exit" in out.columns:
        out["exit"] = pd.to_numeric(out["exit"], errors="coerce")
    if "sl" in out.columns:
        out["sl"] = pd.to_numeric(out["sl"], errors="coerce")
    if "tp" in out.columns:
        out["tp"] = pd.to_numeric(out["tp"], errors="coerce")
    if "pnl" in out.columns:
        out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    return out


def _build_payload(
    *,
    chart_df: pd.DataFrame,
    trades_iter: pd.DataFrame,
    bt_trades: pd.DataFrame,
    line_segments: list[dict[str, Any]],
    title: str,
    include_black_lines: bool,
    max_liq_lines: int,
    max_visible_liq_lines: int,
    initial_bars: int,
    lazy_load_chunk: int,
) -> dict[str, Any]:
    bar_seconds = _infer_bar_seconds(chart_df.index)
    candles: list[dict[str, Any]] = []
    for t, row in chart_df.iterrows():
        ts = _to_unix_s(t)
        if ts is None:
            continue
        candles.append(
            {
                "time": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    initial_start_idx, initial_end_idx = _initial_window_bounds(
        chart_df,
        trades_iter,
        initial_bars=initial_bars,
    )

    markers: list[dict[str, Any]] = []
    bracket_lines: list[dict[str, Any]] = []
    trade_boxes: list[dict[str, Any]] = []
    trade_columns: list[str] = []
    trade_rows: list[dict[str, Any]] = []

    table_source = trades_iter.copy()
    if table_source.empty and not bt_trades.empty:
        table_source = bt_trades.copy()

    if not table_source.empty:
        trade_columns = [str(c) for c in table_source.columns]
        tr_for_table = table_source.copy()
        if "entry_time" in tr_for_table.columns:
            tr_for_table["entry_time"] = pd.to_datetime(tr_for_table["entry_time"], utc=True, errors="coerce")
        if "exit_time" in tr_for_table.columns:
            tr_for_table["exit_time"] = pd.to_datetime(tr_for_table["exit_time"], utc=True, errors="coerce")
        if "entry_time" in tr_for_table.columns:
            tr_for_table = tr_for_table.sort_values("entry_time")

        for _, row in tr_for_table.iterrows():
            rec: dict[str, Any] = {}
            for col in trade_columns:
                rec[col] = _table_value_for_col(row.get(col), col)
            rec["entry_time_unix"] = _to_unix_s(row.get("entry_time")) if "entry_time" in trade_columns else None
            rec["exit_time_unix"] = _to_unix_s(row.get("exit_time")) if "exit_time" in trade_columns else None
            trade_rows.append(rec)

        tr_plot = _normalize_trades_for_plot(table_source)
        tr_plot = tr_plot.dropna(subset=["entry_time", "exit_time"])

        for _, row in tr_plot.iterrows():
            side = str(row.get("side", "")).lower()
            et = _to_unix_s(row.get("entry_time"))
            xt = _to_unix_s(row.get("exit_time"))
            if et is None or xt is None:
                continue
            # Same-bar exits (entry_time == exit_time) still need a drawable span.
            draw_end = max(int(xt), int(et) + int(max(bar_seconds, 1)))

            entry_px = pd.to_numeric(pd.Series([row.get("entry")]), errors="coerce").iloc[0]
            exit_px = pd.to_numeric(pd.Series([row.get("exit")]), errors="coerce").iloc[0]
            sl = pd.to_numeric(pd.Series([row.get("sl")]), errors="coerce").iloc[0]
            tp = pd.to_numeric(pd.Series([row.get("tp")]), errors="coerce").iloc[0]
            pnl = pd.to_numeric(pd.Series([row.get("pnl")]), errors="coerce").iloc[0]
            reason = str(row.get("exit_reason", ""))

            if side == "long":
                entry_color = "#16a34a"
                entry_shape = "arrowUp"
                entry_pos = "belowBar"
                exit_color = "#16a34a" if (math.isfinite(float(pnl)) and float(pnl) >= 0) else "#dc2626"
                exit_pos = "aboveBar"
                exit_shape = "circle"
            else:
                entry_color = "#dc2626"
                entry_shape = "arrowDown"
                entry_pos = "aboveBar"
                exit_color = "#16a34a" if (math.isfinite(float(pnl)) and float(pnl) >= 0) else "#dc2626"
                exit_pos = "belowBar"
                exit_shape = "circle"

            if math.isfinite(float(entry_px)):
                markers.append(
                    {
                        "time": et,
                        "position": entry_pos,
                        "color": entry_color,
                        "shape": entry_shape,
                        "text": f"{side[0].upper()} in {float(entry_px):.5f}",
                    }
                )
            if math.isfinite(float(exit_px)):
                markers.append(
                    {
                        "time": xt,
                        "position": exit_pos,
                        "color": exit_color,
                        "shape": exit_shape,
                        "text": f"{side[0].upper()} out {float(exit_px):.5f} {reason}",
                    }
                )

            if math.isfinite(float(sl)):
                bracket_lines.append(
                    {
                        "start": et,
                        "end": draw_end,
                        "level": float(sl),
                        "color": "#2563eb",
                        "width": 1,
                        "style": "dashed",
                    }
                )
            if math.isfinite(float(tp)):
                bracket_lines.append(
                    {
                        "start": et,
                        "end": draw_end,
                        "level": float(tp),
                        "color": "#16a34a",
                        "width": 1,
                        "style": "dashed",
                    }
                )

            if math.isfinite(float(entry_px)) and math.isfinite(float(tp)):
                trade_boxes.append(
                    {
                        "start": et,
                        "end": draw_end,
                        "base": float(entry_px),
                        "value": float(tp),
                        "fillColor": "rgba(22,163,74,0.14)",
                        "lineColor": "rgba(22,163,74,0.85)",
                        "kind": "tp",
                    }
                )

            if math.isfinite(float(entry_px)) and math.isfinite(float(sl)):
                trade_boxes.append(
                    {
                        "start": et,
                        "end": draw_end,
                        "base": float(entry_px),
                        "value": float(sl),
                        "fillColor": "rgba(37,99,235,0.14)",
                        "lineColor": "rgba(37,99,235,0.85)",
                        "kind": "sl",
                    }
                )

    liq_lines: list[dict[str, Any]] = []
    window_start = pd.Timestamp(chart_df.index.min())
    window_end = pd.Timestamp(chart_df.index.max())
    prepared = _prepare_liq_segments(
        line_segments,
        window_start=window_start,
        window_end=window_end,
        include_black_lines=include_black_lines,
        max_liq_lines=max_liq_lines,
    )
    for idx, seg in enumerate(prepared):
        s0 = _to_unix_s(seg.get("start"))
        s0_active = _to_unix_s(seg.get("active_start"))
        s1 = _to_unix_s(seg.get("end"))
        if s0 is None or s1 is None or s0_active is None:
            continue
        if s1 <= s0_active:
            continue
        state = int(seg.get("state", 0))
        pool = str(seg.get("pool", ""))
        liq_lines.append(
            {
                "start": s0,
                "active_start": s0_active,
                "end": s1,
                "level": float(seg.get("level", 0.0)),
                "id": int(idx),
                "state": state,
                "pool": pool,
                "color": STATE_COLORS.get(state, "#6b7280"),
                "width": 3 if pool.startswith("htf_") else 1,
                "style": "solid",
            }
        )

    price_precision = _infer_price_precision(chart_df)
    price_min_move = 10 ** (-price_precision)

    return {
        "title": title,
        "bar_seconds": bar_seconds,
        "focus_window_bars": 240,
        "price_precision": int(price_precision),
        "price_min_move": float(price_min_move),
        "candles": candles,
        "initial_start_idx": int(initial_start_idx),
        "initial_end_idx": int(initial_end_idx),
        "lazy_load_chunk": int(max(100, lazy_load_chunk)),
        "max_visible_liq_lines": int(max(100, max_visible_liq_lines)),
        "markers": markers,
        "liq_lines": liq_lines,
        "bracket_lines": bracket_lines,
        "trade_boxes": trade_boxes,
        "trade_columns": trade_columns,
        "trade_rows": trade_rows,
        "state_legend": [{"state": k, "label": v, "color": STATE_COLORS[k]} for k, v in STATE_LABEL.items()],
    }


def _build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{payload.get("title", "Iteration Chart")}</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: #f8fafc;
      color: #0f172a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      height: 100%;
    }}
    .wrap {{
      display: grid;
      grid-template-rows: auto auto auto 52vh minmax(220px, 1fr);
      gap: 8px;
      height: 100vh;
      padding: 10px 12px;
      box-sizing: border-box;
    }}
    .title {{
      font-size: 14px;
      font-weight: 600;
      color: #0f172a;
    }}
    .legend {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 12px;
      color: #334155;
    }}
    .hover-info {{
      margin-left: auto;
      color: #0f172a;
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 62vw;
    }}
    .sw {{
      width: 14px;
      height: 3px;
      border-radius: 2px;
      display: inline-block;
      margin-right: 6px;
    }}
    .replay-controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 12px;
      color: #334155;
    }}
    .replay-time {{
      color: #0f172a;
      font-variant-numeric: tabular-nums;
    }}
    .replay-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      user-select: none;
      color: #334155;
    }}
    .replay-btn {{
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #0f172a;
      border-radius: 6px;
      padding: 3px 8px;
      font-size: 12px;
      cursor: pointer;
    }}
    .replay-btn:hover {{
      background: #f1f5f9;
    }}
    #chart {{
      width: 100%;
      height: 100%;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      overflow: hidden;
    }}
    .trades-box {{
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #ffffff;
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
    }}
    .trades-title {{
      padding: 8px 10px;
      font-size: 13px;
      font-weight: 600;
      color: #0f172a;
      border-bottom: 1px solid #e2e8f0;
    }}
    .trades-wrap {{
      overflow: auto;
      min-height: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      color: #0f172a;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
      padding: 6px 8px;
      white-space: nowrap;
    }}
    tbody td {{
      border-bottom: 1px solid #f1f5f9;
      padding: 6px 8px;
      color: #1e293b;
      white-space: nowrap;
    }}
    tbody tr:hover {{
      background: #f8fafc;
    }}
    .jump-btn {{
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #0369a1;
      cursor: pointer;
      border-radius: 6px;
      padding: 2px 6px;
      font-size: 11px;
      font-family: inherit;
    }}
    .jump-btn:hover {{
      background: #e0f2fe;
      border-color: #7dd3fc;
    }}
    .empty-msg {{
      padding: 12px;
      color: #64748b;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title" id="chart-title"></div>
    <div class="legend" id="legend"></div>
    <div class="replay-controls">
      <span>Replay</span>
      <button class="replay-btn" id="replay-step-forward" type="button">+1 Bar</button>
      <button class="replay-btn" id="replay-refresh-lines" type="button">Refresh Lines</button>
      <label class="replay-toggle" for="replay-performance-mode">
        <input id="replay-performance-mode" type="checkbox" checked />
        <span>Performance mode</span>
      </label>
      <span class="replay-time" id="replay-time"></span>
    </div>
    <div id="chart"></div>
    <div class="trades-box">
      <div class="trades-title">Trades (click entry/exit time to center chart)</div>
      <div class="trades-wrap" id="trades-wrap">
        <div class="empty-msg">No trades available for this iteration.</div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <script>
    const data = {data_json};
    document.getElementById("chart-title").textContent = data.title || "Iteration Chart";

    const legendEl = document.getElementById("legend");
    for (const row of (data.state_legend || [])) {{
      const item = document.createElement("div");
      item.innerHTML = `<span class="sw" style="background:${{row.color}}"></span>Liquidity ${{row.label}}`;
      legendEl.appendChild(item);
    }}
    const extra = document.createElement("div");
    extra.innerHTML = `<span class="sw" style="background:#2563eb"></span>SL`;
    legendEl.appendChild(extra);
    const extra2 = document.createElement("div");
    extra2.innerHTML = `<span class="sw" style="background:#16a34a"></span>TP`;
    legendEl.appendChild(extra2);
    const hoverInfo = document.createElement("div");
    hoverInfo.className = "hover-info";
    hoverInfo.id = "hover-info";
    legendEl.appendChild(hoverInfo);

    const chartContainer = document.getElementById("chart");
    const chart = LightweightCharts.createChart(chartContainer, {{
      autoSize: true,
      layout: {{
        background: {{ type: "solid", color: "#ffffff" }},
        textColor: "#111827",
      }},
      grid: {{
        vertLines: {{ color: "rgba(148,163,184,0.25)" }},
        horzLines: {{ color: "rgba(148,163,184,0.25)" }},
      }},
      crosshair: {{
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: {{
          visible: true,
          labelVisible: true,
        }},
        horzLine: {{
          visible: true,
          labelVisible: true,
        }},
      }},
      rightPriceScale: {{
        borderColor: "rgba(100,116,139,0.5)",
        autoScale: true,
        mode: LightweightCharts.PriceScaleMode.Normal,
        alignLabels: true,
        entireTextOnly: false,
        scaleMargins: {{
          top: 0.02,
          bottom: 0.02,
        }},
      }},
      timeScale: {{
        borderColor: "rgba(100,116,139,0.5)",
        timeVisible: true,
        secondsVisible: false,
      }},
    }});

    const candle = chart.addCandlestickSeries({{
      upColor: "#16a34a",
      downColor: "#dc2626",
      borderDownColor: "#b91c1c",
      borderUpColor: "#15803d",
      wickDownColor: "#b91c1c",
      wickUpColor: "#15803d",
      priceLineVisible: false,
      priceFormat: {{
        type: "price",
        precision: Number(data.price_precision || 5),
        minMove: Number(data.price_min_move || 0.00001),
      }},
    }});
    const allCandles = data.candles || [];
    let startIdx = Math.max(0, Number(data.initial_start_idx || 0));
    let endIdx = Math.min(allCandles.length, Number(data.initial_end_idx || allCandles.length));
    if (endIdx <= startIdx) {{
      startIdx = 0;
      endIdx = allCandles.length;
    }}
    const loadChunk = Math.max(100, Number(data.lazy_load_chunk || 1000));
    const maxVisibleLiq = Math.max(100, Number(data.max_visible_liq_lines || 250));
    const replayZoomBars = Math.max(
      24,
      Math.min(140, Math.floor(Number(data.focus_window_bars || 240) * 0.35))
    );
    const atrLen = Math.max(5, Math.min(50, Number(data.atr_len || 14)));

    const replayStepForwardEl = document.getElementById("replay-step-forward");
    const replayRefreshLinesEl = document.getElementById("replay-refresh-lines");
    const replayPerfModeEl = document.getElementById("replay-performance-mode");
    const replayTimeEl = document.getElementById("replay-time");
    replayStepForwardEl.title = "Step +1 bar (Shift+Right)";
    let performanceMode = true;
    let replayIdx = allCandles.length > 0
      ? Math.max(0, Math.min(allCandles.length - 1, Math.max(0, endIdx - 1)))
      : -1;

    function getReplayIdx() {{
      if (allCandles.length === 0) return -1;
      replayIdx = Math.max(0, Math.min(allCandles.length - 1, Number(replayIdx)));
      return replayIdx;
    }}

    function visibleSliceBounds() {{
      let s = startIdx;
      let e = endIdx;
      const ridx = getReplayIdx();
      if (ridx >= 0) {{
        e = Math.min(e, ridx + 1);
        if (e <= s) {{
          const span = Math.max(300, Number(data.focus_window_bars || 240));
          s = Math.max(0, e - span);
        }}
      }}
      return {{ s, e }};
    }}

    function applyCandleWindow() {{
      const b = visibleSliceBounds();
      candle.setData(allCandles.slice(b.s, b.e));
      if ((data.markers || []).length > 0) {{
        const ts = getReplayTs();
        candle.setMarkers((data.markers || []).filter((m) => Number(m.time) <= Number(ts)));
      }}
    }}

    applyCandleWindow();

    function lineStyle(style) {{
      if (style === "dashed") return LightweightCharts.LineStyle.Dashed;
      if (style === "dotted") return LightweightCharts.LineStyle.Dotted;
      return LightweightCharts.LineStyle.Solid;
    }}

    function addLineSegments(segments, sink) {{
      for (const seg of segments || []) {{
        const start = Number(seg.start);
        let end = Number(seg.end);
        if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
        if (end <= start) {{
          end = start + Math.max(1, Number(data.bar_seconds || 300));
        }}
        const s = chart.addLineSeries({{
          color: seg.color || "#6b7280",
          lineWidth: seg.width || 1,
          lineStyle: lineStyle(seg.style),
          autoscaleInfoProvider: () => null,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        }});
        s.setData([
          {{ time: start, value: seg.level }},
          {{ time: end, value: seg.level }},
        ]);
        if (Array.isArray(sink)) sink.push(s);
      }}
    }}

    function addTradeBoxes(boxes, sink) {{
      for (const b of boxes || []) {{
        const start = Number(b.start);
        let end = Number(b.end);
        const base = Number(b.base);
        const value = Number(b.value);
        if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
        if (end <= start) {{
          end = start + Math.max(1, Number(data.bar_seconds || 300));
        }}
        if (!Number.isFinite(base) || !Number.isFinite(value)) continue;

        const fillColor = b.fillColor || "rgba(100,116,139,0.12)";
        const lineColor = b.lineColor || "rgba(100,116,139,0.8)";
        const s = chart.addBaselineSeries({{
          baseValue: {{ type: "price", price: base }},
          topLineColor: lineColor,
          topFillColor1: fillColor,
          topFillColor2: fillColor,
          bottomLineColor: lineColor,
          bottomFillColor1: fillColor,
          bottomFillColor2: fillColor,
          autoscaleInfoProvider: () => null,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        }});
        s.setData([
          {{ time: start, value: value }},
          {{ time: end, value: value }},
        ]);
        if (Array.isArray(sink)) sink.push(s);
      }}
    }}

    const staticSeries = [];
    addTradeBoxes(data.trade_boxes || [], staticSeries);
    addLineSegments(data.bracket_lines || [], staticSeries);
    chart.timeScale().fitContent();

    function fmtPrice(v) {{
      if (!Number.isFinite(v)) return "";
      const p = Number(data.price_precision || 5);
      return Number(v).toFixed(Math.max(0, p));
    }}

    function fmtTime(tsUnix) {{
      if (!Number.isFinite(tsUnix)) return "";
      const d = new Date(Number(tsUnix) * 1000);
      const pad = (n) => String(n).padStart(2, "0");
      return `${{d.getUTCFullYear()}}-${{pad(d.getUTCMonth() + 1)}}-${{pad(d.getUTCDate())}} ` +
             `${{pad(d.getUTCHours())}}:${{pad(d.getUTCMinutes())}}:${{pad(d.getUTCSeconds())}} UTC`;
    }}

    function getReplayTs() {{
      if (allCandles.length === 0) return null;
      const idx = getReplayIdx();
      const bar = allCandles[idx];
      return bar ? Number(bar.time) : null;
    }}

    function refreshReplayLabel() {{
      const ts = getReplayTs();
      replayTimeEl.textContent = ts === null ? "" : `Replay time: ${{fmtTime(ts)}}`;
    }}

    function setReplayVisibleRange(tsUnix) {{
      if (!Number.isFinite(tsUnix)) return;
      const barSec = Number(data.bar_seconds || 300);
      const halfBars = Math.max(12, Math.floor(replayZoomBars / 2));
      const halfSec = halfBars * barSec;
      chart.timeScale().setVisibleRange({{
        from: tsUnix - halfSec,
        to: tsUnix + halfSec,
      }});
    }}

    function computeAtr(endIdxExclusive, len) {{
      const end = Math.max(1, Math.min(allCandles.length, endIdxExclusive));
      const start = Math.max(1, end - Math.max(1, len));
      let sumTr = 0;
      let count = 0;
      for (let i = start; i < end; i++) {{
        const curr = allCandles[i];
        const prev = allCandles[i - 1];
        if (!curr || !prev) continue;
        const hi = Number(curr.high);
        const lo = Number(curr.low);
        const pc = Number(prev.close);
        if (!Number.isFinite(hi) || !Number.isFinite(lo) || !Number.isFinite(pc)) continue;
        const tr = Math.max(hi - lo, Math.abs(hi - pc), Math.abs(lo - pc));
        if (!Number.isFinite(tr)) continue;
        sumTr += tr;
        count += 1;
      }}
      return count > 0 ? (sumTr / count) : null;
    }}

    function applyVerticalStretch(tsUnix) {{
      if (!Number.isFinite(tsUnix) || allCandles.length === 0) return;
      const idx = findCandleIndex(tsUnix);
      if (idx < 0) return;
      const focusBar = allCandles[idx];
      if (!focusBar) return;

      const half = Math.max(12, Math.floor(replayZoomBars / 2));
      const ws = Math.max(0, idx - half);
      const we = Math.min(allCandles.length, idx + half + 1);
      if (we <= ws) return;

      let lo = Number.POSITIVE_INFINITY;
      let hi = Number.NEGATIVE_INFINITY;
      for (let i = ws; i < we; i++) {{
        const b = allCandles[i];
        if (!b) continue;
        const l = Number(b.low);
        const h = Number(b.high);
        if (Number.isFinite(l)) lo = Math.min(lo, l);
        if (Number.isFinite(h)) hi = Math.max(hi, h);
      }}
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return;

      const atr = computeAtr(idx + 1, atrLen);
      const span = hi - lo;
      const minMove = Math.max(1e-12, Number(data.price_min_move || 0.00001));
      const barHi = Number(focusBar.high);
      const barLo = Number(focusBar.low);
      const barClose = Number(focusBar.close);
      let center = (Number.isFinite(barHi) && Number.isFinite(barLo)) ? ((barHi + barLo) / 2.0) : barClose;
      if (!Number.isFinite(center)) center = (lo + hi) / 2.0;

      const halfByWindow = Math.max(minMove * 40, (span * 0.56));
      const halfByAtr = Number.isFinite(atr) ? (atr * 2.0) : 0;
      const halfSpan = Math.max(halfByWindow, halfByAtr);
      const from = center - halfSpan;
      const to = center + halfSpan;

      const ps = chart.priceScale("right");
      if (ps && typeof ps.setVisibleRange === "function") {{
        if (ps.applyOptions) {{
          ps.applyOptions({{ autoScale: false }});
        }}
        ps.setVisibleRange({{ from, to }});
        return;
      }}
      if (ps && ps.applyOptions) {{
        // Fallback for API variants without setVisibleRange():
        // lock current vertical scale to avoid per-bar drift during replay.
        ps.applyOptions({{ autoScale: false }});
      }}
    }}

    function ensureLatestReplayBarVisibleKeepZoom() {{
      const b = visibleSliceBounds();
      const visibleLen = Math.max(0, b.e - b.s);
      if (visibleLen <= 0) return;
      const target = visibleLen - 1;
      const lr = chart.timeScale().getVisibleLogicalRange();
      if (!lr || !Number.isFinite(lr.from) || !Number.isFinite(lr.to)) {{
        centerLatestReplayBarKeepCurrentZoom(replayZoomBars);
        return;
      }}
      const from = Number(lr.from);
      const to = Number(lr.to);
      if (target >= from && target <= to) return;
      const span = Math.max(10, to - from);
      const half = span / 2.0;
      chart.timeScale().setVisibleLogicalRange({{
        from: target - half,
        to: target + half,
      }});
    }}

    function centerLatestReplayBarKeepCurrentZoom(preferredSpan = null) {{
      const b = visibleSliceBounds();
      const visibleLen = Math.max(0, b.e - b.s);
      if (visibleLen <= 0) return;
      const latestLogical = visibleLen - 1;
      let span = Number(preferredSpan);
      if (!Number.isFinite(span) || span <= 0) {{
        const lr = chart.timeScale().getVisibleLogicalRange();
        span = replayZoomBars;
        if (lr && Number.isFinite(lr.from) && Number.isFinite(lr.to)) {{
          span = Math.max(10, Number(lr.to) - Number(lr.from));
        }}
      }}
      const half = span / 2.0;
      chart.timeScale().setVisibleLogicalRange({{
        from: latestLogical - half,
        to: latestLogical + half,
      }});
    }}

    let suppressLazyLoadUntil = 0;
    function suppressLazyLoad(ms) {{
      const until = Date.now() + Math.max(0, Number(ms || 0));
      if (until > suppressLazyLoadUntil) {{
        suppressLazyLoadUntil = until;
      }}
    }}
    function isLazyLoadSuppressed() {{
      return Date.now() < suppressLazyLoadUntil;
    }}

    function hardCenterOnReplayBar(tsUnix, opts = {{}}) {{
      if (!Number.isFinite(tsUnix)) return;
      const zoomHorizontal = Boolean(opts.zoomHorizontal);
      const stretchVertical = Boolean(opts.stretchVertical);
      suppressLazyLoad(2000);

      const applyOnce = () => {{
        if (zoomHorizontal) {{
          setReplayVisibleRange(tsUnix);
          centerLatestReplayBarKeepCurrentZoom(replayZoomBars);
        }} else {{
          ensureLatestReplayBarVisibleKeepZoom();
          centerLatestReplayBarKeepCurrentZoom(null);
        }}
        if (stretchVertical) {{
          applyVerticalStretch(tsUnix);
        }}
      }};

      applyOnce();
      if (typeof requestAnimationFrame === "function") {{
        requestAnimationFrame(() => applyOnce());
      }}
      setTimeout(() => applyOnce(), 30);
      setTimeout(() => applyOnce(), 120);
    }}

    function captureViewportSnapshot() {{
      const lr = chart.timeScale().getVisibleLogicalRange();
      const out = {{ logical: null }};
      if (lr && Number.isFinite(lr.from) && Number.isFinite(lr.to)) {{
        out.logical = {{ from: Number(lr.from), to: Number(lr.to) }};
      }}
      return out;
    }}

    function restoreViewportSnapshot(snap) {{
      if (!snap || !snap.logical) return;
      chart.timeScale().setVisibleLogicalRange({{
        from: Number(snap.logical.from),
        to: Number(snap.logical.to),
      }});
    }}

    function restoreViewportSnapshotWithLock(snap) {{
      if (!snap || !snap.logical) return;
      restoreViewportSnapshot(snap);
      if (typeof requestAnimationFrame === "function") {{
        requestAnimationFrame(() => restoreViewportSnapshot(snap));
      }}
      setTimeout(() => restoreViewportSnapshot(snap), 40);
      setTimeout(() => restoreViewportSnapshot(snap), 120);
    }}

    const priority = (state) => state === 2 ? 0 : (state === 1 ? 1 : 2);
    const allLiqSorted = (data.liq_lines || [])
      .map((seg, idx) => ({{ ...seg, _idx: idx, _prio: priority(Number(seg.state || 0)) }}))
      .sort((a, b) => {{
        if (a._prio !== b._prio) return a._prio - b._prio;
        return Number(b.end || 0) - Number(a.end || 0);
      }});

    const liqSeries = new Map();

    function getVisibleRangeUnix() {{
      const vr = chart.timeScale().getVisibleRange();
      if (vr && Number.isFinite(vr.from) && Number.isFinite(vr.to)) {{
        return {{ from: Number(vr.from), to: Number(vr.to) }};
      }}
      if (allCandles.length === 0) return {{ from: 0, to: 0 }};
      return {{
        from: Number(allCandles[Math.max(0, startIdx)].time),
        to: Number(allCandles[Math.max(0, endIdx - 1)].time),
      }};
    }}

    function renderLiquidityLinesNow() {{
      const replayTs = getReplayTs();
      const vr = getVisibleRangeUnix();
      const margin = Math.max(60, Number(data.bar_seconds || 300) * 80);
      const from = vr.from - margin;
      const to = vr.to + margin;

      const chosen = [];
      for (const seg of allLiqSorted) {{
        let s = Number(seg.start);
        const sActive = Number(seg.active_start ?? seg.start);
        let e = Number(seg.end);
        if (!Number.isFinite(s) || !Number.isFinite(sActive) || !Number.isFinite(e)) continue;
        if (replayTs !== null) {{
          if (sActive > replayTs) continue;
          e = Math.min(e, replayTs);
        }}
        if (e <= s) continue;
        if (e < from || s > to) continue;
        chosen.push({{ seg, s, e }});
        if (chosen.length >= maxVisibleLiq) break;
      }}

      const keep = new Set();
      for (const item of chosen) {{
        const seg = item.seg;
        const id = String(seg.id ?? seg._idx);
        keep.add(id);

        let rec = liqSeries.get(id);
        if (!rec) {{
          const s = chart.addLineSeries({{
            color: seg.color || "#6b7280",
            lineWidth: seg.width || 1,
            lineStyle: lineStyle(seg.style),
            autoscaleInfoProvider: () => null,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          }});
          rec = {{
            series: s,
            s: Number.NaN,
            e: Number.NaN,
            lvl: Number.NaN,
          }};
          liqSeries.set(id, rec);
        }}

        const lvl = Number(seg.level);
        if (rec.s !== item.s || rec.e !== item.e || rec.lvl !== lvl) {{
          rec.series.setData([
            {{ time: item.s, value: lvl }},
            {{ time: item.e, value: lvl }},
          ]);
          rec.s = item.s;
          rec.e = item.e;
          rec.lvl = lvl;
        }}
      }}

      for (const [id, rec] of liqSeries.entries()) {{
        if (keep.has(id)) continue;
        try {{ chart.removeSeries(rec.series); }} catch (_) {{}}
        liqSeries.delete(id);
      }}
    }}

    let liqRenderTimer = null;
    function scheduleRenderLiquidityLines() {{
      if (liqRenderTimer !== null) {{
        clearTimeout(liqRenderTimer);
      }}
      liqRenderTimer = setTimeout(() => {{
        liqRenderTimer = null;
        renderLiquidityLinesNow();
      }}, 90);
    }}

    replayRefreshLinesEl.addEventListener("click", () => {{
      scheduleRenderLiquidityLines();
    }});
    replayPerfModeEl.addEventListener("change", () => {{
      performanceMode = Boolean(replayPerfModeEl.checked);
      if (!performanceMode) {{
        scheduleRenderLiquidityLines();
      }}
    }});

    chart.subscribeCrosshairMove((param) => {{
      if (!param || param.time === undefined) {{
        hoverInfo.textContent = "";
        return;
      }}
      const bar = param.seriesData.get(candle);
      if (!bar) {{
        hoverInfo.textContent = `t=${{fmtTime(Number(param.time))}}`;
        return;
      }}
      hoverInfo.textContent =
        `t=${{fmtTime(Number(param.time))}}  O=${{fmtPrice(Number(bar.open))}}` +
        `  H=${{fmtPrice(Number(bar.high))}}  L=${{fmtPrice(Number(bar.low))}}` +
        `  C=${{fmtPrice(Number(bar.close))}}`;
    }});

    function findCandleIndex(tsUnix) {{
      if (!Number.isFinite(tsUnix) || allCandles.length === 0) return -1;
      let lo = 0;
      let hi = allCandles.length - 1;
      while (lo <= hi) {{
        const mid = (lo + hi) >> 1;
        const t = Number(allCandles[mid].time);
        if (t < tsUnix) {{
          lo = mid + 1;
        }} else if (t > tsUnix) {{
          hi = mid - 1;
        }} else {{
          return mid;
        }}
      }}
      if (lo >= allCandles.length) return allCandles.length - 1;
      if (lo <= 0) return 0;
      const a = Number(allCandles[lo - 1].time);
      const b = Number(allCandles[lo].time);
      return Math.abs(a - tsUnix) <= Math.abs(b - tsUnix) ? (lo - 1) : lo;
    }}

    function ensureTimeLoaded(tsUnix) {{
      const idx = findCandleIndex(tsUnix);
      if (idx < 0) return;
      if (idx >= startIdx && idx < endIdx) return;
      const span = Math.max(endIdx - startIdx, Math.max(500, Number(data.focus_window_bars || 240)));
      const half = Math.floor(span / 2);
      startIdx = Math.max(0, idx - half);
      endIdx = Math.min(allCandles.length, startIdx + span);
      startIdx = Math.max(0, endIdx - span);
      applyCandleWindow();
    }}

    function focusAt(tsUnix) {{
      if (!Number.isFinite(tsUnix)) return;
      ensureTimeLoaded(tsUnix);
      const idx = findCandleIndex(tsUnix);
      if (idx >= 0) {{
        replayIdx = idx;
        refreshReplayLabel();
        applyCandleWindow();
        // Date-click jump: hard center the selected replay/latest bar horizontally and vertically.
        hardCenterOnReplayBar(tsUnix, {{ zoomHorizontal: true, stretchVertical: true }});
        // After datetime jump, keep camera fixed while stepping bars.
        pinCameraOnStep = true;
        const ps = chart.priceScale("right");
        if (ps && ps.applyOptions) {{
          ps.applyOptions({{ autoScale: false }});
        }}
      }}
      scheduleRenderLiquidityLines();
    }}

    let loadingWindow = false;
    let pinCameraOnStep = false;
    function maybeLoadMore(range) {{
      if (loadingWindow || !range || isLazyLoadSuppressed() || pinCameraOnStep) return;
      const info = candle.barsInLogicalRange(range);
      if (!info) return;
      let changed = false;
      loadingWindow = true;

      if (info.barsBefore !== null && info.barsBefore < 120 && startIdx > 0) {{
        const add = Math.min(loadChunk, startIdx);
        startIdx -= add;
        changed = true;
      }}
      // Replay mode is always ON, so we avoid auto-loading future bars.

      if (changed) {{
        applyCandleWindow();
      }}
      loadingWindow = false;
    }}

    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
      maybeLoadMore(range);
      if (!pinCameraOnStep) {{
        scheduleRenderLiquidityLines();
      }}
    }});

    function stepReplayForwardOneBar() {{
      if (allCandles.length === 0) return;
      const curr = getReplayIdx();
      if (curr < 0) return;
      const next = Math.min(allCandles.length - 1, curr + 1);
      if (next >= endIdx - 2) {{
        endIdx = Math.min(allCandles.length, next + loadChunk);
      }}
      replayIdx = next;
      refreshReplayLabel();
      const lockedViewport = pinCameraOnStep ? captureViewportSnapshot() : null;
      applyCandleWindow();
      // When camera is pinned from datetime jump, stepping should not move chart.
      if (pinCameraOnStep) {{
        restoreViewportSnapshotWithLock(lockedViewport);
      }} else {{
        ensureLatestReplayBarVisibleKeepZoom();
      }}
      if (!performanceMode) {{
        scheduleRenderLiquidityLines();
      }}
    }}
    replayStepForwardEl.addEventListener("click", stepReplayForwardOneBar);
    document.addEventListener("keydown", (ev) => {{
      if (!(ev.shiftKey && ev.key === "ArrowRight")) return;
      const target = ev.target;
      const tag = (target && target.tagName ? String(target.tagName).toLowerCase() : "");
      if (tag === "textarea" || tag === "select") return;
      if (target && target.isContentEditable) return;
      if (tag === "input") {{
        const inType = String(target.type || "").toLowerCase();
        if (inType !== "range") return;
      }}
      ev.preventDefault();
      stepReplayForwardOneBar();
    }});
    refreshReplayLabel();
    applyCandleWindow();
    const initialTs = getReplayTs();
    // Initial load should use replay zoom.
    hardCenterOnReplayBar(initialTs, {{ zoomHorizontal: true, stretchVertical: true }});
    scheduleRenderLiquidityLines();

    function buildTradesTable() {{
      const wrap = document.getElementById("trades-wrap");
      const cols = data.trade_columns || [];
      const rows = data.trade_rows || [];
      if (rows.length === 0 || cols.length === 0) {{
        wrap.innerHTML = '<div class="empty-msg">No trades available for this iteration.</div>';
        return;
      }}

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const hr = document.createElement("tr");
      for (const c of cols) {{
        const th = document.createElement("th");
        th.textContent = c;
        hr.appendChild(th);
      }}
      thead.appendChild(hr);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const row of rows) {{
        const tr = document.createElement("tr");
        for (const c of cols) {{
          const td = document.createElement("td");
          const v = row[c] ?? "";
          if (c === "entry_time" && Number.isFinite(row.entry_time_unix)) {{
            const btn = document.createElement("button");
            btn.className = "jump-btn";
            btn.textContent = String(v);
            btn.addEventListener("click", () => focusAt(Number(row.entry_time_unix)));
            td.appendChild(btn);
          }} else if (c === "exit_time" && Number.isFinite(row.exit_time_unix)) {{
            const btn = document.createElement("button");
            btn.className = "jump-btn";
            btn.textContent = String(v);
            btn.addEventListener("click", () => focusAt(Number(row.exit_time_unix)));
            td.appendChild(btn);
          }} else {{
            td.textContent = String(v);
          }}
          tr.appendChild(td);
        }}
        tbody.appendChild(tr);
      }}
      table.appendChild(tbody);

      wrap.innerHTML = "";
      wrap.appendChild(table);
    }}

    buildTradesTable();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser("Render limited iteration chart using TradingView lightweight-charts.")
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
        "--max-bars",
        type=int,
        default=6000,
        help="Initial candles to render before lazy-loading more on scroll.",
    )
    parser.add_argument(
        "--lazy-load-chunk",
        type=int,
        default=2000,
        help="How many candles to add per lazy-load step when scrolling near edges.",
    )
    parser.add_argument(
        "--max-liq-lines",
        type=int,
        default=0,
        help="Maximum liquidity segments to include in payload. 0 means no cap.",
    )
    parser.add_argument(
        "--max-visible-liq-lines",
        type=int,
        default=180,
        help="Maximum liquidity segments drawn at once in viewport/replay mode.",
    )
    parser.add_argument(
        "--include-black-lines",
        action="store_true",
        help="Include black liquidity lines. Off by default for performance.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Default: <run-dir>/iteration_<iter>_lightweight.html",
    )
    args = parser.parse_args()

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
    bt_out = mod.run_backtest(df_feat, strategy_params=strategy_params, cfg=cfg, debug=debug)
    line_events = debug.get("line_events", [])
    bt_trades = pd.DataFrame()
    if isinstance(bt_out, tuple) and len(bt_out) >= 2 and isinstance(bt_out[1], pd.DataFrame):
        bt_trades = bt_out[1].copy()

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
        f"Limited Iteration {int(args.iter)} | Strategy={strategy_module} | Bars={len(chart_df)}"
    )
    payload = _build_payload(
        chart_df=chart_df,
        trades_iter=trades_iter,
        bt_trades=bt_trades,
        line_segments=line_segments,
        title=title,
        include_black_lines=bool(args.include_black_lines),
        max_liq_lines=int(args.max_liq_lines),
        max_visible_liq_lines=int(args.max_visible_liq_lines),
        initial_bars=int(args.max_bars),
        lazy_load_chunk=int(args.lazy_load_chunk),
    )

    html = _build_html(payload)
    out_path = Path(args.output) if args.output else (run_dir / f"iteration_{int(args.iter)}_lightweight.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved lightweight chart: {out_path}")


if __name__ == "__main__":
    main()
