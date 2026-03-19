from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "plot_limited_iteration_lightweight.py"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_limited_iteration_lightweight", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load plot_limited_iteration_lightweight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_limited_iteration_lightweight = _load_module()


class PlotLimitedIterationLightweightTests(unittest.TestCase):
    def test_load_iteration_trades_reads_selected_iteration_from_saved_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"iter": 1, "entry_time": "2025-01-01T00:00:00Z", "sl": 1.0, "tp": 2.0},
                    {"iter": 2, "entry_time": "2025-01-02T00:00:00Z", "sl": 3.0, "tp": 4.0},
                ]
            ).to_csv(tables_dir / "trades.csv", index=False)

            out = plot_limited_iteration_lightweight._load_iteration_trades(run_dir, iter_id=2)  # pylint: disable=protected-access

            self.assertEqual(len(out), 1)
            self.assertEqual(str(out.iloc[0]["entry_time"]), "2025-01-02T00:00:00Z")
            self.assertAlmostEqual(float(out.iloc[0]["sl"]), 3.0, places=8)
            self.assertAlmostEqual(float(out.iloc[0]["tp"]), 4.0, places=8)

    def test_load_iteration_trades_returns_empty_without_iter_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"entry_time": "2025-01-01T00:00:00Z"}]).to_csv(tables_dir / "trades.csv", index=False)

            out = plot_limited_iteration_lightweight._load_iteration_trades(run_dir, iter_id=1)  # pylint: disable=protected-access

            self.assertTrue(out.empty)

    def test_build_summary_metrics_prefers_iteration_metrics_and_formats_cards(self) -> None:
        iter_row = pd.Series(
            {
                "_net_profit_abs": 1234.5,
                "total_return_%": 12.34,
                "trades": 9,
                "win_rate_%": 66.67,
                "avg_trade_net_profit_abs": 137.17,
                "profit_factor": 1.8,
                "max_drawdown_abs_%": 4.2,
                "favourable": True,
            }
        )
        trades = pd.DataFrame(
            [
                {"pnl": 200.0},
                {"pnl": -50.0},
            ]
        )

        out = plot_limited_iteration_lightweight._build_summary_metrics(iter_row, trades)  # pylint: disable=protected-access
        values = {row["label"]: row["value"] for row in out}

        self.assertEqual(values["Net P&L"], "1,234.50")
        self.assertEqual(values["Return %"], "12.34%")
        self.assertEqual(values["Trades"], "9")
        self.assertEqual(values["Win Rate"], "66.67%")
        self.assertEqual(values["Avg Trade"], "137.17")
        self.assertEqual(values["Profit Factor"], "1.80")
        self.assertEqual(values["Max DD %"], "4.20%")
        self.assertEqual(values["Favourable"], "Yes")

    def test_build_summary_metrics_falls_back_to_trade_metrics(self) -> None:
        iter_row = pd.Series(dtype=object)
        trades = pd.DataFrame(
            [
                {"pnl": 100.0},
                {"pnl": -25.0},
                {"pnl": 50.0},
            ]
        )

        out = plot_limited_iteration_lightweight._build_summary_metrics(iter_row, trades)  # pylint: disable=protected-access
        values = {row["label"]: row["value"] for row in out}

        self.assertEqual(values["Net P&L"], "125.00")
        self.assertEqual(values["Trades"], "3")
        self.assertEqual(values["Win Rate"], "66.67%")
        self.assertEqual(values["Avg Trade"], "41.67")
        self.assertEqual(values["Profit Factor"], "6.00")

    def test_build_html_includes_replay_to_click_controls(self) -> None:
        html = plot_limited_iteration_lightweight._build_html(  # pylint: disable=protected-access
            {
                "title": "Test",
                "candles": [],
                "state_legend": [],
                "summary_metrics": [],
                "trade_columns": [],
                "trade_rows": [],
                "markers": [],
                "liq_lines": [],
                "bracket_lines": [],
                "trade_boxes": [],
                "bar_seconds": 300,
                "focus_window_bars": 240,
                "price_precision": 5,
                "price_min_move": 0.00001,
                "initial_start_idx": 0,
                "initial_end_idx": 0,
                "lazy_load_chunk": 2000,
                "max_visible_liq_lines": 180,
            }
        )

        self.assertIn('id="replay-jump-to-click"', html)
        self.assertIn("chart.subscribeClick((param)", html)

    def test_build_payload_adds_trade_artifact_ids_and_cap(self) -> None:
        chart_df = pd.DataFrame(
            [
                {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05},
                {"open": 1.05, "high": 1.2, "low": 1.0, "close": 1.1},
            ],
            index=pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:05:00Z"], utc=True),
        )
        bt_trades = pd.DataFrame(
            [
                {
                    "entry_time": "2025-01-01T00:00:00Z",
                    "exit_time": "2025-01-01T00:05:00Z",
                    "side": "long",
                    "entry": 1.0,
                    "exit": 1.1,
                    "sl": 0.95,
                    "tp": 1.1,
                    "pnl": 100.0,
                    "exit_reason": "tp",
                }
            ]
        )

        payload = plot_limited_iteration_lightweight._build_payload(  # pylint: disable=protected-access
            chart_df=chart_df,
            bt_trades=bt_trades,
            iter_row=None,
            line_segments=[],
            title="Test",
            include_black_lines=False,
            max_liq_lines=0,
            max_visible_liq_lines=180,
            initial_bars=100,
            lazy_load_chunk=2000,
        )

        self.assertEqual(payload["max_visible_trade_artifacts"], 240)
        self.assertEqual(payload["bracket_lines"][0]["id"], "sl-0")
        self.assertEqual(payload["bracket_lines"][1]["id"], "tp-0")
        self.assertEqual(payload["trade_boxes"][0]["id"], "tp-box-0")
        self.assertEqual(payload["trade_boxes"][1]["id"], "sl-box-0")


if __name__ == "__main__":
    unittest.main()
