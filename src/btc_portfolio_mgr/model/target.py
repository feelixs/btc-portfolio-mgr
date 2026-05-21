"""Forward-return target construction and dataset assembly."""
from __future__ import annotations

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA

DATASET_SCHEMA: dict[str, pl.DataType] = {
    **FEATURE_SCHEMA,
    "target": pl.Float64(),
}


def compute_forward_log_return(
    prices_reindexed: pl.DataFrame, horizon_hours: int
) -> pl.Series:
    """log(price_{t+horizon} / price_t).

    `prices_reindexed` MUST be on a complete hourly grid (use
    `features.gaps.reindex_to_hourly`). Returns null where the future
    price is missing.
    """
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive, got {horizon_hours}")
    expr = pl.col("price").shift(-horizon_hours).log() - pl.col("price").log()
    return prices_reindexed.select(expr.alias("target")).get_column("target")


def build_dataset(
    features: pl.DataFrame, prices: pl.DataFrame, horizon_hours: int
) -> pl.DataFrame:
    """Join features + forward-return target by timestamp, drop null rows.

    Returns a DataFrame matching DATASET_SCHEMA (timestamp + 15 features + target).
    Rows where any feature or the target is null are dropped — those rows
    can't train the model.
    """
    prices_reindexed = reindex_to_hourly(prices)
    if prices_reindexed.height == 0:
        return pl.DataFrame(schema=DATASET_SCHEMA)
    target = compute_forward_log_return(prices_reindexed, horizon_hours)
    target_df = prices_reindexed.select("timestamp").with_columns(target=target)
    joined = features.join(target_df, on="timestamp", how="inner")
    cleaned = joined.drop_nulls()
    # Enforce column order matches DATASET_SCHEMA
    return cleaned.select(list(DATASET_SCHEMA.keys()))
