from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from btc_portfolio_mgr.features.gaps import find_gaps, reindex_to_hourly
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
