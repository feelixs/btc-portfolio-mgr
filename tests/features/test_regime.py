from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.regime import (
    compute_kaufman_efficiency,
    compute_ma_zscore,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_zscore_constant_price_is_null() -> None:
    # std == 0 -> zscore undefined -> null
    prices = reindex_to_hourly(hourly([100.0] * 10))
    z = compute_ma_zscore(prices, window_hours=4)
    assert z.to_list()[5] is None


def test_zscore_one_sigma_above_mean() -> None:
    # Build a window where the latest price is meaningfully above mean.
    # Series: 100, 102, 100, 102, 100, 102, 100, 102, 104.
    # The 4-hour window ending at the last index = [100, 102, 100, 104].
    prices = reindex_to_hourly(hourly([100, 102, 100, 102, 100, 102, 100, 102, 104]))
    z = compute_ma_zscore(prices, window_hours=4)
    last = z.to_list()[-1]
    assert last is not None
    assert last > 0


def test_zscore_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0, 102.0, 103.0], gap_hours=2, prices_after=[110.0, 111.0])
    )
    # Reindexed: [100, 101, 102, 103, null, null, 110, 111]
    # zscore window=4: idx 3 first valid (window=[100,101,102,103])
    # idx 4..7 windows contain nulls -> null
    z = compute_ma_zscore(prices, window_hours=4)
    values = z.to_list()
    assert values[3] is not None
    assert values[4] is None
    assert values[5] is None
    assert values[6] is None
    assert values[7] is None


def test_kaufman_efficiency_monotonic_is_one() -> None:
    # Monotonic increase: |p_t - p_{t-n}| == sum(|p_i - p_{i-1}|) -> KER = 1
    prices = reindex_to_hourly(hourly([100, 101, 102, 103, 104, 105]))
    ker = compute_kaufman_efficiency(prices, window_hours=4)
    values = ker.to_list()
    # First valid at index = window_hours (needs window+1 prices for `window` diffs)
    assert math.isclose(values[4], 1.0)
    assert math.isclose(values[5], 1.0)


def test_kaufman_efficiency_oscillation_is_low() -> None:
    # Pure oscillation: net change = 0 -> KER = 0
    prices = reindex_to_hourly(hourly([100, 101, 100, 101, 100, 101]))
    ker = compute_kaufman_efficiency(prices, window_hours=4)
    values = ker.to_list()
    assert math.isclose(values[4], 0.0)
    assert math.isclose(values[5], 0.0)


def test_kaufman_efficiency_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0, 102.0], gap_hours=2, prices_after=[110.0, 111.0])
    )
    ker = compute_kaufman_efficiency(prices, window_hours=4)
    # Window=4 needs 5 contiguous non-null prices. The [100,101,102,null,null,110,111]
    # series of length 7 has no 5-prices-contiguous segment -> all KER null.
    assert all(v is None for v in ker.to_list())
