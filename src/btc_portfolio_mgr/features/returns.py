"""Multi-lookback log-return features."""
from __future__ import annotations

import polars as pl


def compute_log_return(prices: pl.DataFrame, lookback_hours: int) -> pl.Series:
    """log(price_t / price_{t-lookback_hours}).

    Assumes `prices` is reindexed to a complete hourly grid (use
    `gaps.reindex_to_hourly` first). Returns null where either endpoint
    of the lookback window is missing.
    """
    if lookback_hours <= 0:
        raise ValueError(f"lookback_hours must be positive, got {lookback_hours}")
    expr = pl.col("price").log() - pl.col("price").shift(lookback_hours).log()
    return prices.select(expr.alias("ret")).get_column("ret")
