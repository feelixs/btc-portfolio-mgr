from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.target import (
    DATASET_SCHEMA,
    build_dataset,
    compute_forward_log_return,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_forward_return_constant_price_is_zero() -> None:
    prices = reindex_to_hourly(hourly([100.0] * 10))
    target = compute_forward_log_return(prices, horizon_hours=2)
    values = target.to_list()
    # First 8 rows have a future price -> 0.0; last 2 don't -> null.
    assert all(v == 0.0 for v in values[:8])
    assert values[8] is None
    assert values[9] is None


def test_forward_return_known_value() -> None:
    prices = reindex_to_hourly(hourly([100.0, 110.0, 121.0, 133.1]))
    target = compute_forward_log_return(prices, horizon_hours=2)
    # target[0] = log(121/100), target[1] = log(133.1/110), target[2..3] = null
    values = target.to_list()
    assert math.isclose(values[0], math.log(121 / 100))
    assert math.isclose(values[1], math.log(133.1 / 110))
    assert values[2] is None
    assert values[3] is None


def test_forward_return_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0, 102.0], gap_hours=2, prices_after=[110.0, 111.0])
    )
    # Reindexed: [100, 101, 102, null, null, 110, 111]
    # horizon=1: target[i] = log(price[i+1]/price[i])
    # i=0: log(101/100) ✓
    # i=1: log(102/101) ✓
    # i=2: log(null/102) = null
    # i=3: null source price -> null
    # i=4: log(110/null) = null
    # i=5: log(111/110) ✓
    # i=6: no next price -> null
    target = compute_forward_log_return(prices, horizon_hours=1)
    values = target.to_list()
    assert math.isclose(values[0], math.log(101 / 100))
    assert math.isclose(values[1], math.log(102 / 101))
    assert values[2] is None
    assert values[3] is None
    assert values[4] is None
    assert math.isclose(values[5], math.log(111 / 110))
    assert values[6] is None


def test_build_dataset_drops_null_rows_and_returns_schema() -> None:
    # Need at least 2160 + 24 = 2184 hours so some rows have both full features and a valid target.
    prices = hourly([100.0 + i * 0.01 for i in range(2300)])
    features = compose_features(prices)
    dataset = build_dataset(features=features, prices=prices, horizon_hours=24)
    # Schema must match contract.
    assert dataset.schema == DATASET_SCHEMA
    assert dataset.columns == ["timestamp"] + FEATURE_COLUMNS + ["target"]
    # No nulls remain after drop.
    for col in dataset.columns:
        assert dataset[col].null_count() == 0, f"{col} has nulls"
    # Sanity: row count is between 0 and input height, and timestamps are sorted ascending.
    assert 0 < dataset.height < 2300
    assert dataset["timestamp"].is_sorted()


def test_build_dataset_empty_input_returns_empty() -> None:
    empty_prices = pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "price": pl.Float64(), "volume": pl.Float64()}
    )
    empty_features = compose_features(empty_prices)
    dataset = build_dataset(features=empty_features, prices=empty_prices, horizon_hours=24)
    assert dataset.height == 0
    assert dataset.schema == DATASET_SCHEMA
