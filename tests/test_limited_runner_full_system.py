from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantbt.experiments.limited.runner import _get_full_system_runner


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


if __name__ == "__main__":
    unittest.main()
