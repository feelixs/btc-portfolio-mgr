from __future__ import annotations

import math

import polars as pl
import pytest

from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.features.schema import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    FeatureSchemaMismatchError,
    assert_feature_schema,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_compose_features_returns_canonical_schema() -> None:
    # Need at least 90d = 2160h of data for the longest lookback (zscore_90d).
    prices = hourly([100.0 + i * 0.01 for i in range(2200)])
    features = compose_features(prices)
    assert features.schema == FEATURE_SCHEMA
    assert features.columns == ["timestamp"] + FEATURE_COLUMNS


def test_compose_features_emits_one_row_per_input_hour() -> None:
    prices = hourly([100.0 + i * 0.01 for i in range(2200)])
    features = compose_features(prices)
    assert features.height == 2200


def test_compose_features_early_rows_are_null_for_long_lookbacks() -> None:
    prices = hourly([100.0 + i * 0.01 for i in range(2200)])
    features = compose_features(prices)
    # ret_30d needs 720 prior hours; first 720 rows of ret_30d are null.
    ret_30d = features["ret_30d"].to_list()
    assert all(v is None for v in ret_30d[:720])
    assert ret_30d[720] is not None
    # zscore_90d needs 2160 prior hours.
    z90 = features["zscore_90d"].to_list()
    assert all(v is None for v in z90[:2159])


def test_compose_features_propagates_gap_nulls() -> None:
    # 50 hours, then a long gap (> MAX_INTERPOLATION_HOURS), then 50 hours.
    # The gap must exceed the interpolation threshold so prices stay null.
    GAP = 12
    prices = with_gap(
        prices_before=[100.0 + i * 0.01 for i in range(50)],
        gap_hours=GAP,
        prices_after=[110.0 + i * 0.01 for i in range(50)],
    )
    features = compose_features(prices)
    # Reindexed length: 50 + GAP + 50 = 50 + 12 + 50 = 112.
    assert features.height == 50 + GAP + 50
    # Rows inside the gap have null prices -> all features null.
    for idx in range(50, 50 + GAP):
        for col in FEATURE_COLUMNS:
            assert features[col].to_list()[idx] is None, f"col {col} idx {idx}"


def test_assert_feature_schema_rejects_bad_input() -> None:
    bad = pl.DataFrame({"timestamp": [1], "ret_1h": [2.0]})
    with pytest.raises(FeatureSchemaMismatchError):
        assert_feature_schema(bad)


def test_compose_features_empty_input_returns_empty_with_schema() -> None:
    empty = pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC"), "price": pl.Float64(), "volume": pl.Float64()})
    features = compose_features(empty)
    assert features.height == 0
    assert features.schema == FEATURE_SCHEMA
