from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import polars as pl

from btc_portfolio_mgr.backtest.metrics import compute_backtest_metrics


def _bars(equity_series: list[float], weight_series: list[float] | None = None) -> pl.DataFrame:
    n = len(equity_series)
    weights = weight_series or [0.5] * n
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n)],
            "weight": weights,
            "net_pnl": [
                (equity_series[i] - equity_series[i - 1]) if i > 0 else 0.0
                for i in range(n)
            ],
            "equity": equity_series,
        }
    )


def test_metrics_constant_equity_zero_sharpe() -> None:
    bars = _bars([1.0] * 365)
    m = compute_backtest_metrics(bars)
    assert m["annualized_return"] == 0.0
    assert math.isnan(m["sharpe"]) or m["sharpe"] == 0.0


def test_metrics_monotonic_growth_positive_sharpe() -> None:
    # 0.1% per day equity growth -> ~44% annualized
    n = 365
    equity = [1.0 * (1.001 ** i) for i in range(n)]
    bars = _bars(equity)
    m = compute_backtest_metrics(bars)
    assert m["annualized_return"] > 0.4
    assert m["annualized_return"] < 0.5
    assert m["sharpe"] > 10  # monotonic = essentially zero vol


def test_metrics_max_drawdown_known_value() -> None:
    # Equity: 1.0, 1.1, 0.9, 1.05 -> peak at 1.1, trough at 0.9, DD = 0.2/1.1 ≈ 0.182
    bars = _bars([1.0, 1.1, 0.9, 1.05])
    m = compute_backtest_metrics(bars)
    assert math.isclose(m["max_drawdown"], (1.1 - 0.9) / 1.1, abs_tol=1e-9)


def test_metrics_turnover_known_value() -> None:
    # Daily weight changes: 0.5, 0.3, 0.5, 0.3
    # Δweight sums: 0.0 + 0.2 + 0.2 + 0.2 = 0.6 over 4 days = ~54.75 annualized
    bars = _bars(
        equity_series=[1.0, 1.0, 1.0, 1.0],
        weight_series=[0.5, 0.3, 0.5, 0.3],
    )
    m = compute_backtest_metrics(bars)
    # 0.6 / 4 * 365 = 54.75
    assert math.isclose(m["turnover_annualized"], 54.75, abs_tol=0.01)


def test_metrics_time_in_market_known_value() -> None:
    bars = _bars(
        equity_series=[1.0] * 5,
        weight_series=[0.5, 0.0, 0.3, 0.0, 0.0],
    )
    m = compute_backtest_metrics(bars)
    assert math.isclose(m["time_in_market"], 2 / 5, abs_tol=1e-9)


def test_metrics_empty_returns_nan() -> None:
    bars = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "weight": pl.Float64(),
            "net_pnl": pl.Float64(),
            "equity": pl.Float64(),
        }
    )
    m = compute_backtest_metrics(bars)
    for key in ("annualized_return", "sharpe", "max_drawdown"):
        assert math.isnan(m[key])
