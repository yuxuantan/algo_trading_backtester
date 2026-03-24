from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantbt.experiments.limited.runner import (
    _build_schedule_metrics_from_trades,
    _get_full_system_runner,
)


class LimitedRunnerFullSystemTests(unittest.TestCase):
    def test_get_full_system_runner_requires_entry_test_single_rule_and_plugin_hook(self) -> None:
        def fake_plugin(*args, **kwargs):
            return None

        def fake_runner(*args, **kwargs):
            return None

        fake_plugin.run_full_system = fake_runner

        runner = _get_full_system_runner(
            test_focus="entry_test",
            entry_mode="all",
            entry_combo=({"plugin": fake_plugin, "params": {}},),
        )
        self.assertIs(runner, fake_runner)
        self.assertIsNone(
            _get_full_system_runner(
                test_focus="exit_test",
                entry_mode="all",
                entry_combo=({"plugin": fake_plugin, "params": {}},),
            )
        )
        self.assertIsNone(
            _get_full_system_runner(
                test_focus="entry_test",
                entry_mode="vote",
                entry_combo=({"plugin": fake_plugin, "params": {}},),
            )
        )
        self.assertIsNone(
            _get_full_system_runner(
                test_focus="entry_test",
                entry_mode="all",
                entry_combo=({"plugin": fake_plugin, "params": {}}, {"plugin": fake_plugin, "params": {}}),
            )
        )

    def test_build_schedule_metrics_from_trades_uses_trade_times(self) -> None:
        idx = pd.to_datetime(
            [
                "2020-01-01 00:00:00+00:00",
                "2020-01-01 00:15:00+00:00",
                "2020-01-01 00:30:00+00:00",
                "2020-01-01 00:45:00+00:00",
            ],
            utc=True,
        )
        trades = pd.DataFrame(
            [
                {"entry_time": idx[0], "exit_time": idx[2], "side": "long"},
                {"entry_time": idx[1], "exit_time": idx[3], "side": "short"},
            ]
        )

        metrics = _build_schedule_metrics_from_trades(idx, trades)

        self.assertEqual(metrics["trades"], 2.0)
        self.assertEqual(metrics["long_trade_pct"], 50.0)
        self.assertEqual(metrics["avg_bars_held"], 2.0)


if __name__ == "__main__":
    unittest.main()
