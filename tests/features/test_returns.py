from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.returns import compute_log_return
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_log_return_constant_price_is_zero() -> None:
    prices = reindex_to_hourly(hourly([100.0] * 5))
    rets = compute_log_return(prices, lookback_hours=1)
    # First row has no prior price -> null; rest are 0.0
    values = rets.to_list()
    assert values[0] is None
    assert all(v == 0.0 for v in values[1:])


def test_log_return_known_values() -> None:
    prices = reindex_to_hourly(hourly([100.0, 110.0, 121.0]))
    rets = compute_log_return(prices, lookback_hours=1)
    values = rets.to_list()
    assert values[0] is None
    assert math.isclose(values[1], math.log(1.1))
    assert math.isclose(values[2], math.log(1.1))


def test_log_return_longer_lookback() -> None:
    # 4h return on a series that doubled over 4 hours
    prices = reindex_to_hourly(hourly([100.0, 105.0, 110.0, 115.0, 200.0]))
    rets = compute_log_return(prices, lookback_hours=4)
    values = rets.to_list()
    assert all(v is None for v in values[:4])
    assert math.isclose(values[4], math.log(2.0))


def test_log_return_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0], gap_hours=2, prices_after=[110.0, 111.0])
    )
    # Layout: [100, 101, null, null, 110, 111]
    # 1h returns: [None, log(1.01), None, None, None, log(111/110)]
    rets = compute_log_return(prices, lookback_hours=1)
    values = rets.to_list()
    assert values[0] is None
    assert math.isclose(values[1], math.log(101 / 100))
    assert values[2] is None
    assert values[3] is None
    assert values[4] is None  # prev row is null
    assert math.isclose(values[5], math.log(111 / 110))
