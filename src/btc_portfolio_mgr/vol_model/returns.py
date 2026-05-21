"""1h log return extraction from hourly prices for GARCH fitting."""
from __future__ import annotations

import polars as pl

from btc_portfolio_mgr.features.gaps import interpolate_short_gaps, reindex_to_hourly
from btc_portfolio_mgr.features.returns import compute_log_return


def extract_log_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Reindex prices to hourly grid, interpolate short gaps, compute log returns.

    Short null gaps (≤ MAX_INTERPOLATION_HOURS in gaps.py) are linear-interpolated
    so a single missing sample doesn't poison downstream GARCH fitting with a
    fake 0% return. Longer gaps stay null and produce null log returns, which
    are dropped from the output.

    Returns a DataFrame with columns (timestamp: Datetime us UTC, log_return: Float64).
    """
    schema = {
        "timestamp": pl.Datetime("us", "UTC"),
        "log_return": pl.Float64(),
    }
    if prices.height == 0:
        return pl.DataFrame(schema=schema)
    reindexed = interpolate_short_gaps(reindex_to_hourly(prices))
    returns = compute_log_return(reindexed, lookback_hours=1)
    return (
        reindexed.select("timestamp")
        .with_columns(log_return=returns)
        .drop_nulls()
    )
