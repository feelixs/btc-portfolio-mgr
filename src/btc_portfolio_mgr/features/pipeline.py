"""Composes all feature families into the canonical feature matrix."""
from __future__ import annotations

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.regime import (
    compute_kaufman_efficiency,
    compute_ma_zscore,
)
from btc_portfolio_mgr.features.returns import compute_log_return
from btc_portfolio_mgr.features.schema import FEATURE_SCHEMA, assert_feature_schema
from btc_portfolio_mgr.features.volatility import (
    compute_realized_kurt,
    compute_realized_skew,
    compute_realized_vol,
)

# Lookbacks expressed in hours.
_RETURN_LOOKBACKS: dict[str, int] = {
    "ret_1h": 1,
    "ret_4h": 4,
    "ret_24h": 24,
    "ret_7d": 168,
    "ret_30d": 720,
}
_VOL_WINDOWS: dict[str, int] = {
    "vol_24h": 24,
    "vol_7d": 168,
    "vol_30d": 720,
}
_ZSCORE_WINDOWS: dict[str, int] = {
    "zscore_30d": 720,
    "zscore_90d": 2160,
}
_SKEW_WINDOWS: dict[str, int] = {
    "skew_24h": 24,
    "skew_7d": 168,
}
_KURT_WINDOWS: dict[str, int] = {
    "kurt_24h": 24,
    "kurt_7d": 168,
}
_KER_WINDOWS: dict[str, int] = {
    "ker_24h": 24,
}


def compose_features(prices: pl.DataFrame) -> pl.DataFrame:
    """Build the full feature matrix from a hourly price DataFrame.

    Input: prices with canonical storage SCHEMA. Output: DataFrame with
    FEATURE_SCHEMA. One row per hour in the reindexed grid. Null entries
    where the relevant rolling window includes a data gap.
    """
    reindexed = reindex_to_hourly(prices)
    if reindexed.height == 0:
        return pl.DataFrame(schema=FEATURE_SCHEMA)
    columns: dict[str, pl.Series] = {"timestamp": reindexed["timestamp"]}
    for name, lb in _RETURN_LOOKBACKS.items():
        columns[name] = compute_log_return(reindexed, lb).alias(name)
    for name, w in _VOL_WINDOWS.items():
        columns[name] = compute_realized_vol(reindexed, w).alias(name)
    for name, w in _ZSCORE_WINDOWS.items():
        columns[name] = compute_ma_zscore(reindexed, w).alias(name)
    for name, w in _SKEW_WINDOWS.items():
        columns[name] = compute_realized_skew(reindexed, w).alias(name)
    for name, w in _KURT_WINDOWS.items():
        columns[name] = compute_realized_kurt(reindexed, w).alias(name)
    for name, w in _KER_WINDOWS.items():
        columns[name] = compute_kaufman_efficiency(reindexed, w).alias(name)
    out = pl.DataFrame(columns, schema=FEATURE_SCHEMA)
    assert_feature_schema(out)
    return out
