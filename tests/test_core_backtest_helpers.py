from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantbt.core.engine import BacktestConfig
from quantbt.core.performance import build_backtest_summary
from quantbt.core.trades import close_trade_with_costs, resolve_intrabar_bracket_exit


class CoreBacktestHelperTests(unittest.TestCase):
    def test_close_trade_with_costs_applies_spread_commission_and_r_multiple(self) -> None:
        cfg = BacktestConfig(
            spread_pips=0.2,
            pip_size=0.0001,
            commission_per_round_trip=5.0,
            lot_size=100_000.0,
        )
        pos = {
            "entry_time": pd.Timestamp("2025-01-01T00:00:00Z"),
            "side": "long",
            "entry": 1.1000,
            "sl": 1.0990,
            "tp": 1.1020,
            "units": 100_000.0,
            "risk_dollars": 100.0,
        }

        equity_after, trade = close_trade_with_costs(
            pos=pos,
            exit_price=1.1010,
            exit_time=pd.Timestamp("2025-01-01T00:15:00Z"),
            exit_reason="TP",
            equity_now=100_000.0,
            cfg=cfg,
        )

        self.assertAlmostEqual(equity_after, 100_093.0, places=8)
        self.assertAlmostEqual(float(trade["pnl"]), 93.0, places=8)
        self.assertAlmostEqual(float(trade["commission"]), 5.0, places=8)
        self.assertAlmostEqual(float(trade["r_multiple"]), 0.93, places=8)
        self.assertEqual(str(trade["side"]), "long")

    def test_build_backtest_summary_can_merge_common_metrics(self) -> None:
        idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        equity_df = pd.DataFrame({"equity": [100_000.0, 100_500.0, 101_000.0]}, index=idx)
        trades_df = pd.DataFrame(
            {
                "pnl": [100.0, -50.0],
                "r_multiple": [1.0, -0.5],
            }
        )

        summary = build_backtest_summary(
            equity_like=equity_df,
            trades_df=trades_df,
            initial_equity=100_000.0,
            include_common_metrics=True,
        )

        self.assertEqual(int(summary["trades"]), 2)
        self.assertAlmostEqual(float(summary["final_equity"]), 101_000.0, places=8)
        self.assertAlmostEqual(float(summary["total_return_%"]), 1.0, places=8)
        self.assertAlmostEqual(float(summary["profit_factor"]), 2.0, places=8)
        self.assertAlmostEqual(float(summary["avg_R"]), 0.25, places=8)
        self.assertAlmostEqual(float(summary["avg_profit_per_trade"]), 25.0, places=8)
        self.assertEqual(int(summary["wins"]), 1)
        self.assertEqual(int(summary["losses"]), 1)
        self.assertIn("cagr_%", summary)
        self.assertIn("sortino", summary)

    def test_build_backtest_summary_exposes_avg_profit_per_trade_by_default(self) -> None:
        idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        equity_df = pd.DataFrame({"equity": [100_000.0, 100_250.0, 100_500.0]}, index=idx)
        trades_df = pd.DataFrame({"pnl": [100.0, -25.0, 50.0]})

        summary = build_backtest_summary(
            equity_like=equity_df,
            trades_df=trades_df,
            initial_equity=100_000.0,
        )

        self.assertAlmostEqual(float(summary["avg_profit_per_trade"]), 41.6666666667, places=6)
        self.assertAlmostEqual(float(summary["win_rate_%"]), 66.6666666667, places=6)

    def test_resolve_intrabar_bracket_exit_respects_conservative_same_bar(self) -> None:
        exit_price, exit_reason = resolve_intrabar_bracket_exit(
            side="long",
            bar_high=1.1020,
            bar_low=1.0980,
            sl=1.0990,
            tp=1.1010,
            conservative_same_bar=True,
        )

        self.assertAlmostEqual(float(exit_price), 1.0990, places=8)
        self.assertEqual(str(exit_reason), "SL_and_TP_same_bar_assume_SL")

    def test_resolve_intrabar_bracket_exit_allows_custom_same_bar_labels(self) -> None:
        exit_price, exit_reason = resolve_intrabar_bracket_exit(
            side="short",
            bar_high=1.1020,
            bar_low=1.0980,
            sl=1.1010,
            tp=1.0990,
            conservative_same_bar=False,
            same_bar_sl_reason="SL_and_TP_same_bar",
            same_bar_tp_reason="SL_and_TP_same_bar",
        )

        self.assertAlmostEqual(float(exit_price), 1.0990, places=8)
        self.assertEqual(str(exit_reason), "SL_and_TP_same_bar")


if __name__ == "__main__":
    unittest.main()
