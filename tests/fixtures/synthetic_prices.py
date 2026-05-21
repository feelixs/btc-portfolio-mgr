"""Deterministic price-series builders for feature-engineering tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from btc_portfolio_mgr.data.storage import SCHEMA


def hourly(prices: list[float], start: datetime | None = None) -> pl.DataFrame:
    """Build a contiguous hourly price DataFrame from a price list."""
    start = start or datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i, p in enumerate(prices):
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "price": float(p),
                "volume": 0.0,
            }
        )
    return pl.DataFrame(rows, schema=SCHEMA)


def with_gap(
    prices_before: list[float],
    gap_hours: int,
    prices_after: list[float],
    start: datetime | None = None,
) -> pl.DataFrame:
    """Two contiguous hourly blocks separated by `gap_hours` missing hours."""
    start = start or datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    before = hourly(prices_before, start)
    after_start = start + timedelta(hours=len(prices_before) + gap_hours)
    after = hourly(prices_after, after_start)
    return pl.concat([before, after])
