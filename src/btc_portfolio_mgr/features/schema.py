"""Feature matrix schema — the contract consumed by Phase 3+ trainers."""
from __future__ import annotations

import polars as pl

FEATURE_COLUMNS: list[str] = [
    "ret_1h", "ret_4h", "ret_24h", "ret_7d", "ret_30d",
    "vol_24h", "vol_7d", "vol_30d",
    "zscore_30d", "zscore_90d",
    "skew_24h", "skew_7d",
    "kurt_24h", "kurt_7d",
    "ker_24h",
]

FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    **{col: pl.Float64() for col in FEATURE_COLUMNS},
}


class FeatureSchemaMismatchError(ValueError):
    """Raised when a DataFrame's schema does not match FEATURE_SCHEMA."""


def assert_feature_schema(df: pl.DataFrame) -> None:
    if df.schema != FEATURE_SCHEMA:
        raise FeatureSchemaMismatchError(
            f"expected {FEATURE_SCHEMA}, got {df.schema}"
        )
