from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = PROJECT_ROOT / "scripts" / "fetch_mt5_live_data.py"
MODULE_SPEC = importlib.util.spec_from_file_location("fetch_mt5_live_data", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
fetch_mt5_live_data = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(fetch_mt5_live_data)


def _make_rates(start: str, periods: int, freq: str = "15min") -> list[dict]:
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    out: list[dict] = []
    for i, ts in enumerate(idx):
        base = 1.1000 + (i * 0.0001)
        out.append(
            {
                "time": int(ts.timestamp()),
                "open": base,
                "high": base + 0.0002,
                "low": base - 0.0002,
                "close": base + 0.0001,
                "tick_volume": 100 + i,
                "spread": 10,
                "real_volume": 0,
            }
        )
    return out


class FakeRangeMT5:
    def __init__(self, rates: list[dict]) -> None:
        self.backend = "silicon"
        self._rates = rates
        self.range_calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.pos_calls: list[tuple[int, int]] = []

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):  # noqa: ANN001
        del symbol, timeframe
        start = pd.Timestamp(date_from)
        end = pd.Timestamp(date_to)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        self.range_calls.append((start, end))
        return [
            row
            for row in self._rates
            if start <= pd.Timestamp(row["time"], unit="s", tz="UTC") <= end
        ]

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):  # noqa: ANN001
        del symbol, timeframe
        self.pos_calls.append((int(start_pos), int(count)))
        raise AssertionError("date-range fetch should prefer copy_rates_range when available")

    def last_error(self) -> tuple[int, str]:
        return -1, "Terminal: Call failed"


class FakePosMT5:
    def __init__(self, rates: list[dict]) -> None:
        self.backend = "native"
        self._rates = rates
        self.pos_calls: list[tuple[int, int]] = []

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):  # noqa: ANN001
        del symbol, timeframe
        start_pos = int(start_pos)
        count = int(count)
        self.pos_calls.append((start_pos, count))
        end_idx = len(self._rates) - start_pos
        if end_idx <= 0:
            return None
        start_idx = max(0, end_idx - count)
        return self._rates[start_idx:end_idx]

    def last_error(self) -> tuple[int, str]:
        return -1, "Terminal: Call failed"


class FetchMt5LiveDataTests(unittest.TestCase):
    def test_validate_requested_coverage_allows_small_leading_market_closure_gap(self) -> None:
        start = pd.Timestamp("2010-01-01T00:00:00Z")
        end = pd.Timestamp("2010-01-11T00:00:00Z")
        df = pd.DataFrame(_make_rates("2010-01-04T00:00:00Z", periods=5 * 24 * 12)).assign(
            timestamp=lambda x: pd.to_datetime(x["time"], unit="s", utc=True)
        )
        df = df.set_index("timestamp").sort_index()

        fetch_mt5_live_data._validate_requested_coverage_mt5(  # pylint: disable=protected-access
            df=df,
            symbol="EURUSD",
            timeframe="M5",
            start=start,
            end=end,
        )

    def test_validate_requested_coverage_reports_history_floor_directly(self) -> None:
        start = pd.Timestamp("2010-01-01T00:00:00Z")
        end = pd.Timestamp("2026-03-16T00:00:00Z")
        late_start = pd.Timestamp("2025-12-31T03:45:00Z")
        df = pd.DataFrame(_make_rates(late_start.isoformat(), periods=96)).assign(
            timestamp=lambda x: pd.to_datetime(x["time"], unit="s", utc=True)
        )
        df = df.set_index("timestamp").sort_index()

        with self.assertRaisesRegex(ValueError, r"Backend history currently starts at 2025-12-31T03:45:00\+00:00"):
            fetch_mt5_live_data._validate_requested_coverage_mt5(  # pylint: disable=protected-access
                df=df,
                symbol="EURUSD",
                timeframe="M15",
                start=start,
                end=end,
            )

    def test_date_range_fetch_prefers_time_windows_when_range_api_exists(self) -> None:
        start = pd.Timestamp("2025-12-30T00:00:00Z")
        end = pd.Timestamp("2026-01-01T00:00:00Z")
        rates = _make_rates(start.isoformat(), periods=192)
        mt5 = FakeRangeMT5(rates)

        df = fetch_mt5_live_data._fetch_rates_by_date_range(  # pylint: disable=protected-access
            mt5,
            symbol="EURUSD",
            timeframe="M15",
            timeframe_mt5=15,
            start_utc=start,
            end_utc=end,
            batch_size=50,
            max_backfill_batches=10,
            progress_every=0,
            request_cap=50,
        )

        self.assertEqual(len(df), 192)
        self.assertEqual(df.index.min(), start)
        self.assertEqual(df.index.max(), end - pd.Timedelta(minutes=15))
        self.assertGreater(len(mt5.range_calls), 1)
        self.assertEqual(mt5.pos_calls, [])
        fetch_mt5_live_data._validate_requested_coverage_mt5(  # pylint: disable=protected-access
            df=df,
            symbol="EURUSD",
            timeframe="M15",
            start=start,
            end=end,
        )

    def test_date_range_fetch_falls_back_to_pos_when_range_api_missing(self) -> None:
        start = pd.Timestamp("2025-12-31T00:00:00Z")
        end = pd.Timestamp("2026-01-01T00:00:00Z")
        rates = _make_rates("2025-12-30T00:00:00Z", periods=192)
        mt5 = FakePosMT5(rates)

        df = fetch_mt5_live_data._fetch_rates_by_date_range(  # pylint: disable=protected-access
            mt5,
            symbol="EURUSD",
            timeframe="M15",
            timeframe_mt5=15,
            start_utc=start,
            end_utc=end,
            batch_size=32,
            max_backfill_batches=10,
            progress_every=0,
            request_cap=32,
        )

        self.assertEqual(len(df), 96)
        self.assertEqual(df.index.min(), start)
        self.assertEqual(df.index.max(), end - pd.Timedelta(minutes=15))
        self.assertGreaterEqual(len(mt5.pos_calls), 3)


if __name__ == "__main__":
    unittest.main()
