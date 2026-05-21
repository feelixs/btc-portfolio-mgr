from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from btc_portfolio_mgr.features.gaps import (
    MAX_INTERPOLATION_HOURS,
    find_gaps,
    interpolate_short_gaps,
    reindex_to_hourly,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_reindex_contiguous_is_unchanged() -> None:
    prices = hourly([100.0, 101.0, 102.0])
    reindexed = reindex_to_hourly(prices)
    assert reindexed.height == 3
    assert reindexed["price"].to_list() == [100.0, 101.0, 102.0]
    assert reindexed["price"].null_count() == 0


def test_reindex_fills_gap_with_nulls() -> None:
    prices = with_gap([100.0, 101.0], gap_hours=3, prices_after=[110.0, 111.0])
    reindexed = reindex_to_hourly(prices)
    # total span: 2 + 3 + 2 = 7 hours
    assert reindexed.height == 7
    assert reindexed["price"].to_list() == [100.0, 101.0, None, None, None, 110.0, 111.0]


def test_find_gaps_returns_empty_for_contiguous() -> None:
    prices = hourly([100.0, 101.0, 102.0])
    gaps = find_gaps(prices)
    assert gaps.height == 0


def test_find_gaps_reports_each_missing_block() -> None:
    prices = with_gap([100.0, 101.0], gap_hours=3, prices_after=[110.0])
    gaps = find_gaps(prices)
    assert gaps.height == 1
    row = gaps.row(0, named=True)
    assert row["gap_start"] == datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    assert row["gap_end"] == datetime(2024, 1, 1, 4, 0, tzinfo=timezone.utc)
    assert row["missing_hours"] == 3


# --- interpolate_short_gaps ---


def test_interpolate_single_hour_gap_uses_midpoint() -> None:
    # Gap of 1 hour between 100 and 102 -> interpolated to 101.
    prices = with_gap([100.0], gap_hours=1, prices_after=[102.0])
    reindexed = reindex_to_hourly(prices)
    filled = interpolate_short_gaps(reindexed)
    assert filled["price"].to_list() == [100.0, 101.0, 102.0]


def test_interpolate_three_hour_gap_builds_linear_ramp() -> None:
    # 3-hour gap between 100 and 104 -> 101, 102, 103.
    prices = with_gap([100.0], gap_hours=3, prices_after=[104.0])
    reindexed = reindex_to_hourly(prices)
    filled = interpolate_short_gaps(reindexed)
    assert filled["price"].to_list() == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_interpolate_gap_at_exactly_max_threshold_is_filled() -> None:
    # Gap == MAX_INTERPOLATION_HOURS should still be interpolated.
    prices = with_gap(
        [100.0], gap_hours=MAX_INTERPOLATION_HOURS, prices_after=[107.0]
    )
    reindexed = reindex_to_hourly(prices)
    filled = interpolate_short_gaps(reindexed)
    assert filled["price"].null_count() == 0


def test_interpolate_gap_longer_than_max_stays_null() -> None:
    # 10-hour gap exceeds default max of 6 -> stays null.
    prices = with_gap([100.0], gap_hours=10, prices_after=[110.0])
    reindexed = reindex_to_hourly(prices)
    filled = interpolate_short_gaps(reindexed)
    # First row is 100, last is 110; middle 10 are null.
    values = filled["price"].to_list()
    assert values[0] == 100.0
    assert values[-1] == 110.0
    assert all(v is None for v in values[1:-1])


def test_interpolate_contiguous_input_is_unchanged() -> None:
    prices = hourly([100.0, 101.0, 102.0])
    reindexed = reindex_to_hourly(prices)
    filled = interpolate_short_gaps(reindexed)
    assert filled["price"].to_list() == [100.0, 101.0, 102.0]


def test_interpolate_volume_not_interpolated() -> None:
    # Volume should remain null in interpolated rows.
    prices = with_gap([100.0], gap_hours=2, prices_after=[103.0])
    reindexed = reindex_to_hourly(prices)
    filled = interpolate_short_gaps(reindexed)
    volumes = filled["volume"].to_list()
    # First and last are 0.0 (set by synthetic_prices.hourly); middle two are null.
    assert volumes[0] == 0.0
    assert volumes[1] is None
    assert volumes[2] is None
    assert volumes[3] == 0.0


def test_interpolate_negative_max_raises() -> None:
    prices = hourly([100.0, 101.0])
    reindexed = reindex_to_hourly(prices)
    with pytest.raises(ValueError):
        interpolate_short_gaps(reindexed, max_gap_hours=-1)


def test_interpolate_empty_input_returns_empty() -> None:
    empty = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "price": pl.Float64,
            "volume": pl.Float64,
        }
    )
    filled = interpolate_short_gaps(empty)
    assert filled.height == 0
