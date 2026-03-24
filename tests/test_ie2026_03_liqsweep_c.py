from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

liq_sweep_c = importlib.import_module("quantbt.strategies.interequity_2026_03_liqsweep_c")

from quantbt.plugins.entries import interequity_liqsweepc_entry
from quantbt.plugins.exits.time_exit import build_exit as build_time_exit


def _sample_ohlc(rows: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    base = 1.10 + np.sin(np.linspace(0, 2, rows)) * 0.002
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + 0.0005
    low = np.minimum(open_, close) - 0.0005
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


class Ie202603LiqSweepCTests(unittest.TestCase):
    def test_strategy_metadata_points_to_c_plugins(self) -> None:
        self.assertEqual(liq_sweep_c.STRATEGY["name"], "IE2026-03 LiqSweep C")
        self.assertEqual(liq_sweep_c.STRATEGY["entry"]["rules"][0]["name"], "interequity_liqsweepc_entry")
        self.assertEqual(liq_sweep_c.STRATEGY["exit"]["name"], "interequity_liqsweepc_exit")
        self.assertEqual(liq_sweep_c.STRATEGY["entry"]["rules"][0]["params"]["min_rr"], 1.5)
        self.assertEqual(liq_sweep_c.Params().min_rr, 1.5)

    def test_run_backtest_enters_immediately_at_swept_red_level(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.1002, 1.0999],
                "high": [1.1008, 1.1000],
                "low": [1.0998, 1.0992],
                "close": [1.1001, 1.0994],
            },
            index=idx,
        )
        params = liq_sweep_c.Params()
        captured: dict[str, float | int] = {}

        def fake_build_entry_spec_from_exit_override(**kwargs):
            captured["entry_open"] = float(kwargs["entry_open"])
            captured["entry_index"] = int(kwargs["entry_index"])
            captured["structural_atr"] = float(kwargs["structural_atr"])
            return {
                "side": "short",
                "qty": 1.0,
                "sl": float("nan"),
                "tp": float("nan"),
                "risk_dollars": 100.0,
                "time_exit_i": int(kwargs["entry_index"]) + 1,
            }

        sweep_seq = [(1.1000, None), (None, None)]
        with mock.patch.multiple(
            liq_sweep_c,
            track_breach_high=lambda *args, **kwargs: None,
            track_breach_low=lambda *args, **kwargs: None,
            append_high_pivot=lambda *args, **kwargs: None,
            append_low_pivot=lambda *args, **kwargs: None,
            confirm_move_away_high=lambda *args, **kwargs: None,
            confirm_move_away_low=lambda *args, **kwargs: None,
            stop_extending_high=lambda *args, **kwargs: None,
            stop_extending_low=lambda *args, **kwargs: None,
            sweep_triggers=mock.Mock(side_effect=sweep_seq),
            _build_entry_spec_from_exit_override=fake_build_entry_spec_from_exit_override,
        ):
            _equity_df, trades_df, summary = liq_sweep_c.run_backtest(
                df,
                strategy_params=params,
                override_exit_builder=build_time_exit,
                override_exit_params={"hold_bars": 1},
            )

        self.assertEqual(captured["entry_open"], 1.1)
        self.assertEqual(captured["entry_index"], 0)
        self.assertGreater(captured["structural_atr"], 0.0)
        self.assertEqual(int(summary["trades"]), 1)
        self.assertEqual(len(trades_df), 1)
        self.assertEqual(pd.Timestamp(trades_df.iloc[0]["entry_time"]), idx[0])
        self.assertAlmostEqual(float(trades_df.iloc[0]["entry"]), 1.1000, places=8)

    def test_structural_gate_rejects_when_stop_is_more_than_five_atr(self) -> None:
        params = liq_sweep_c.Params(min_rr=1.5, max_rr=10.0, max_sl_atr_mult=5.0)
        high_pool = liq_sweep_c._LevelPool(max_size=10)  # pylint: disable=protected-access
        low_pool = liq_sweep_c._LevelPool(max_size=10)  # pylint: disable=protected-access

        high_pool.lvls = [1.1100]
        high_pool.drawn_act = [True]
        high_pool.state = [liq_sweep_c.ST_RED]
        low_pool.lvls = [1.0950]
        low_pool.drawn_act = [True]
        low_pool.state = [liq_sweep_c.ST_PURPLE]

        gate = liq_sweep_c._structural_entry_gate(  # pylint: disable=protected-access
            side="long",
            entry_price=1.1000,
            structural_atr=0.001,
            high_pool=high_pool,
            low_pool=low_pool,
            strategy_params=params,
        )

        self.assertIsNone(gate)

    def test_entry_plugin_exposes_full_system_rerun_bridge(self) -> None:
        captured: dict[str, object] = {}

        class FakeParams:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)

        class FakeModule:
            Params = FakeParams

            @staticmethod
            def compute_features(df, p):
                captured["params"] = p.kwargs
                return df

            @staticmethod
            def compute_signals(df):
                return df

            @staticmethod
            def run_backtest(df_sig, **kwargs):
                captured["run_backtest_kwargs"] = dict(kwargs)
                return pd.DataFrame(), pd.DataFrame(), {"trades": 0}

        def fake_exit_plugin(*args, **kwargs):
            return {"hold_bars": 1}

        def fake_sizing_plugin(**kwargs):
            return 1.0

        with mock.patch.object(interequity_liqsweepc_entry, "_load_strategy_module", return_value=FakeModule):
            interequity_liqsweepc_entry.run_full_system(
                _sample_ohlc(10),
                entry_params={"min_rr": 1.5},
                exit_plugin=fake_exit_plugin,
                exit_params={"hold_bars": 1},
                cfg=liq_sweep_c.BacktestConfig(),
                sizing_plugin=fake_sizing_plugin,
                sizing_params={"risk_pct": 0.01},
            )

        self.assertEqual(captured["params"], {"min_rr": 1.5})
        self.assertIs(captured["run_backtest_kwargs"]["override_exit_builder"], fake_exit_plugin)
        self.assertEqual(captured["run_backtest_kwargs"]["override_exit_params"], {"hold_bars": 1})
        self.assertIs(captured["run_backtest_kwargs"]["override_size_fn"], fake_sizing_plugin)


if __name__ == "__main__":
    unittest.main()
