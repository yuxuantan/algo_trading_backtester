"""Core backtest engines and metrics."""

from .engine import BacktestConfig, run_backtest_sma_cross
from .engine_limited import run_backtest_limited
from .metrics import max_drawdown, profit_factor
from .performance import cagr_pct, common_performance_metrics, sortino_ratio_from_equity

__all__ = [
    "BacktestConfig",
    "run_backtest_sma_cross",
    "run_backtest_limited",
    "max_drawdown",
    "profit_factor",
    "cagr_pct",
    "sortino_ratio_from_equity",
    "common_performance_metrics",
]
