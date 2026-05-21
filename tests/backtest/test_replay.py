from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import polars as pl

from btc_portfolio_mgr.backtest.replay import replay_sizing
from btc_portfolio_mgr.sizing.params import SizingParams


def _make_forecasts(n: int) -> pl.DataFrame:
    """N daily forecast bars with constant positive mu, constant sigma."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n)],
            "mu": [0.005] * n,
            "sigma": [0.03] * n,
            "refit_anchor": [start] * n,
        },
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "mu": pl.Float64(),
            "sigma": pl.Float64(),
            "refit_anchor": pl.Datetime("us", "UTC"),
        },
    )


def _make_prices(n: int, drift: float) -> pl.DataFrame:
    """N+1 daily-cadence prices following geometric trend with `drift` log return per day."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100.0 * math.exp(drift * i) for i in range(n + 1)]
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n + 1)],
            "price": prices,
            "volume": [0.0] * (n + 1),
        },
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "price": pl.Float64(),
            "volume": pl.Float64(),
        },
    )


def test_replay_returns_bars_with_expected_columns() -> None:
    forecasts = _make_forecasts(10)
    prices = _make_prices(10, drift=0.001)
    params = SizingParams()
    bars, _ = replay_sizing(forecasts, prices, params, cost_bps=10.0)
    assert bars.columns == [
        "timestamp",
        "mu",
        "sigma",
        "target_weight",
        "weight",
        "realized_return",
        "gross_pnl",
        "costs",
        "net_pnl",
        "equity",
    ]
    # One bar per forecast row.
    assert bars.height == 10


def test_replay_positive_drift_grows_equity_when_long() -> None:
    """Constant positive μ + Kelly long position + positive realized drift → equity grows."""
    forecasts = _make_forecasts(50)
    prices = _make_prices(50, drift=0.005)  # +0.5% per day realized
    params = SizingParams()
    bars, summary = replay_sizing(forecasts, prices, params, cost_bps=0.0)
    final_equity = bars["equity"][-1]
    assert final_equity > 1.0, f"expected growth, got {final_equity}"
    assert summary["total_return"] == bars["equity"][-1] - 1.0


def test_replay_cost_reduces_net_pnl() -> None:
    """With identical forecasts/prices, higher cost level produces a lower equity."""
    forecasts = _make_forecasts(20)
    prices = _make_prices(20, drift=0.002)
    params = SizingParams()
    _, summary_free = replay_sizing(forecasts, prices, params, cost_bps=0.0)
    _, summary_costly = replay_sizing(forecasts, prices, params, cost_bps=50.0)
    assert summary_costly["total_return"] < summary_free["total_return"]


def test_replay_threshold_mutes_small_changes() -> None:
    """A high threshold should keep weight constant after the first rebalance,
    so total costs equal the initial rebalance only."""
    forecasts = _make_forecasts(20)
    prices = _make_prices(20, drift=0.001)
    # threshold of 0.05 allows initial entry (0 → 1.0 is 1.0 > 0.05) but blocks further rebalancing.
    params = SizingParams(rebalance_threshold=0.05)
    bars, _ = replay_sizing(forecasts, prices, params, cost_bps=10.0)
    nonzero_cost_bars = bars.filter(pl.col("costs") > 0)
    # Only the first bar should incur cost (entering the initial position).
    assert nonzero_cost_bars.height == 1


def test_replay_empty_forecasts_returns_empty() -> None:
    empty_forecasts = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "mu": pl.Float64(),
            "sigma": pl.Float64(),
            "refit_anchor": pl.Datetime("us", "UTC"),
        }
    )
    prices = _make_prices(5, drift=0.0)
    bars, summary = replay_sizing(
        empty_forecasts, prices, SizingParams(), cost_bps=10.0
    )
    assert bars.height == 0
    assert summary["total_return"] == 0.0
