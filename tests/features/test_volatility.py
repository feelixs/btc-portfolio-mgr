from __future__ import annotations

import math

import numpy as np
import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.volatility import (
    compute_realized_kurt,
    compute_realized_skew,
    compute_realized_vol,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_realized_vol_constant_price_is_zero() -> None:
    # 1h returns are all 0, so realized vol = 0 once window is full.
    prices = reindex_to_hourly(hourly([100.0] * 10))
    vol = compute_realized_vol(prices, window_hours=3)
    values = vol.to_list()
    # rets: [None, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # rolling sum of ret^2 over 3 hours with min_samples=3:
    # idx 0,1,2: window has nulls or partial -> null
    # idx 3: full window [0,0,0] -> vol=0.0
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == 0.0


def test_realized_vol_known_value() -> None:
    # Two 1h log returns each = log(2); vol over window=2 = sqrt(2 * log(2)^2)
    prices = reindex_to_hourly(hourly([100.0, 200.0, 400.0]))
    vol = compute_realized_vol(prices, window_hours=2)
    values = vol.to_list()
    assert values[0] is None
    assert values[1] is None
    expected = math.sqrt(2 * (math.log(2) ** 2))
    assert math.isclose(values[2], expected)


def test_realized_vol_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0, 102.0], gap_hours=2, prices_after=[110.0, 111.0, 112.0])
    )
    # 1h returns: [None, r1, r2, None, None, None, r3, r4]
    # vol with window=2: idx 2 first valid; idx 3..6 have null in window; idx 7 valid again.
    vol = compute_realized_vol(prices, window_hours=2)
    values = vol.to_list()
    assert values[2] is not None
    assert values[3] is None
    assert values[4] is None
    assert values[5] is None
    assert values[6] is None
    assert values[7] is not None


def test_realized_skew_symmetric_is_near_zero() -> None:
    # Returns symmetric around 0: prices 100, 110, 100, 110, 100
    prices = reindex_to_hourly(hourly([100.0, 110.0, 100.0, 110.0, 100.0]))
    skew = compute_realized_skew(prices, window_hours=4)
    values = skew.to_list()
    assert values[4] is not None
    assert abs(values[4]) < 0.5  # roughly symmetric


def test_realized_kurt_constant_returns_is_undefined() -> None:
    # Constant returns => std=0 => kurtosis undefined; we return null.
    prices = reindex_to_hourly(hourly([100.0] * 10))
    kurt = compute_realized_kurt(prices, window_hours=4)
    values = kurt.to_list()
    assert values[5] is None  # std of all-zero returns is 0 => null


def test_realized_kurt_known_distribution() -> None:
    # Use a wider window for stable moment estimates.
    rng = np.random.default_rng(42)
    n = 200
    returns = rng.normal(0, 0.01, n)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * math.exp(r))
    df = reindex_to_hourly(hourly(prices))
    kurt = compute_realized_kurt(df, window_hours=100)
    last_kurt = kurt.to_list()[-1]
    assert last_kurt is not None
    # Excess kurtosis of N(0,1) is 0; finite-sample noise large -> tolerate ±1.5.
    assert abs(last_kurt) < 1.5
