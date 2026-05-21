from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.target import DATASET_SCHEMA
from btc_portfolio_mgr.model.train import (
    DEFAULT_PARAMS,
    CVResult,
    cross_validate,
    train_lightgbm,
)


def _synthetic_dataset(n: int, seed: int = 42) -> pl.DataFrame:
    """Build a dataset where target = 0.3 * ret_1h + 0.2 * vol_24h + noise."""
    from datetime import datetime, timezone, timedelta
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=n - 1)
    cols: dict[str, Any] = {"timestamp": pl.datetime_range(
        start=start, end=end, interval="1h", time_zone="UTC", eager=True,
    )}
    # Random features
    feature_values: dict[str, np.ndarray] = {}
    for col in FEATURE_COLUMNS:
        feature_values[col] = rng.normal(0, 1, n).astype(np.float64)
    # Target driven by two features
    target = (
        0.3 * feature_values["ret_1h"]
        + 0.2 * feature_values["vol_24h"]
        + rng.normal(0, 0.1, n)
    )
    cols.update(feature_values)
    cols["target"] = target
    return pl.DataFrame(cols, schema=DATASET_SCHEMA)


def test_train_lightgbm_learns_known_signal() -> None:
    df = _synthetic_dataset(n=1000)
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["target"].to_numpy()
    booster = train_lightgbm(X, y)
    preds = cast(np.ndarray, booster.predict(X))
    # On training data with a strong linear signal, IC should be very high.
    ic = float(np.corrcoef(y, preds)[0, 1])
    assert ic > 0.8, f"expected IC > 0.8, got {ic}"


def test_train_lightgbm_uses_default_params() -> None:
    assert DEFAULT_PARAMS["objective"] == "huber"
    assert DEFAULT_PARAMS["alpha"] == 0.02
    assert DEFAULT_PARAMS["num_leaves"] == 31
    assert DEFAULT_PARAMS["learning_rate"] == 0.05
    assert DEFAULT_PARAMS["min_data_in_leaf"] == 200
    assert DEFAULT_PARAMS["lambda_l1"] == 0.1
    assert DEFAULT_PARAMS["lambda_l2"] == 0.1
    assert DEFAULT_PARAMS["seed"] == 42


def test_train_lightgbm_param_override() -> None:
    df = _synthetic_dataset(n=500)
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["target"].to_numpy()
    # Override learning_rate; expect a booster still produced
    booster = train_lightgbm(X, y, params={"learning_rate": 0.01}, num_boost_round=100)
    preds = cast(np.ndarray, booster.predict(X))
    assert preds.shape == (500,)


def test_cross_validate_returns_per_fold_metrics() -> None:
    df = _synthetic_dataset(n=2000)
    result = cross_validate(
        df,
        n_folds=5,
        label_horizon_hours=24,
        embargo_hours=24,
    )
    assert isinstance(result, CVResult)
    # Per-fold arrays of size 5
    assert len(result.fold_ic) == 5
    assert len(result.fold_hit_rate) == 5
    assert len(result.fold_rmse) == 5
    assert len(result.fold_r_squared) == 5
    # Aggregated mean fields
    assert result.mean_ic == pytest.approx(np.nanmean(result.fold_ic))
    # OOF predictions cover (some fraction of) the dataset
    assert result.oof_predictions.shape == (df.height,)
    # OOF mask: True at indices that were in any test fold
    assert result.oof_mask.shape == (df.height,)
    assert result.oof_mask.sum() == df.height  # every row is in exactly one test fold
    # Sanity: IC on the known linear signal should be positive on average.
    assert result.mean_ic > 0.2, f"expected mean IC > 0.2, got {result.mean_ic}"
