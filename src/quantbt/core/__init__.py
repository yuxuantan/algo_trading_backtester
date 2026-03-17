"""Core backtest engines and metrics."""

from .engine import BacktestConfig, run_backtest_sma_cross
from .engine_limited import run_backtest_limited
from .indicators import simple_atr
from .metrics import max_drawdown, profit_factor
from .performance import build_backtest_summary, cagr_pct, common_performance_metrics, sortino_ratio_from_equity
from .trades import close_trade_with_costs

__all__ = [
    "BacktestConfig",
    "run_backtest_sma_cross",
    "run_backtest_limited",
    "simple_atr",
    "max_drawdown",
    "profit_factor",
    "build_backtest_summary",
    "cagr_pct",
    "sortino_ratio_from_equity",
    "common_performance_metrics",
    "close_trade_with_costs",
]
