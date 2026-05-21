from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.inference import (
    ModelArtifact,
    load_artifact,
    predict,
    save_artifact,
)
from btc_portfolio_mgr.model.target import DATASET_SCHEMA
from btc_portfolio_mgr.model.train import train_lightgbm


def _toy_artifact() -> ModelArtifact:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (500, len(FEATURE_COLUMNS)))
    y = 0.3 * X[:, 0] + rng.normal(0, 0.1, 500)
    booster = train_lightgbm(X, y, num_boost_round=50)
    return ModelArtifact(
        booster=booster,
        feature_columns=list(FEATURE_COLUMNS),
        target_horizon_hours=24,
        trained_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
    )


def test_save_and_load_artifact_roundtrip(tmp_path: Path) -> None:
    artifact = _toy_artifact()
    model_path = tmp_path / "btc_24h.txt"
    meta_path = tmp_path / "btc_24h.metadata.json"
    save_artifact(artifact, model_path, meta_path)
    assert model_path.exists()
    assert meta_path.exists()
    loaded = load_artifact(model_path, meta_path)
    assert loaded.feature_columns == artifact.feature_columns
    assert loaded.target_horizon_hours == artifact.target_horizon_hours
    assert loaded.trained_at == artifact.trained_at
    # Predictions match
    rng = np.random.default_rng(0)
    X_test = rng.normal(0, 1, (10, len(FEATURE_COLUMNS)))
    original_preds = artifact.booster.predict(X_test)
    loaded_preds = loaded.booster.predict(X_test)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)


def test_predict_returns_series_of_right_length() -> None:
    from datetime import timedelta
    artifact = _toy_artifact()
    # Build a feature DataFrame
    rng = np.random.default_rng(0)
    rows: dict = {col: rng.normal(0, 1, 50).tolist() for col in FEATURE_COLUMNS}
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=49)
    rows["timestamp"] = pl.datetime_range(
        start=start, end=end, interval="1h", time_zone="UTC", eager=True,
    ).to_list()
    features = pl.DataFrame(rows)
    preds = predict(artifact, features)
    assert preds.dtype == pl.Float64
    assert preds.len() == 50


def test_predict_raises_when_feature_columns_missing() -> None:
    artifact = _toy_artifact()
    bad = pl.DataFrame({"timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)], "ret_1h": [0.0]})
    with pytest.raises(KeyError):
        predict(artifact, bad)
