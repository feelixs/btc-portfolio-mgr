from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from btc_portfolio_mgr.data.storage import SCHEMA, write_parquet
from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.features.schema import FEATURE_SCHEMA
from tests.fixtures.synthetic_prices import hourly


def test_train_script_produces_artifact(tmp_path: Path) -> None:
    from scripts import train_model as tm

    # Need:
    #   - 2160h zscore_90d warmup
    #   - 168h forward target window
    #   - enough daily samples (post-resampling) so LightGBM has data per fold
    # 8000 hourly rows -> ~333 days; after warmup ~243 days; with 3 folds = ~81/fold.
    n = 8000
    prices = hourly([100.0 + i * 0.01 + (i % 7) * 0.05 for i in range(n)])
    prices_path = tmp_path / "btc_hourly.parquet"
    features_path = tmp_path / "btc_features.parquet"
    model_path = tmp_path / "btc_7d.txt"
    metadata_path = tmp_path / "btc_7d.metadata.json"

    write_parquet(prices, prices_path)
    features = compose_features(prices)
    features.write_parquet(features_path)

    result = tm.run(
        prices_path=prices_path,
        features_path=features_path,
        model_path=model_path,
        metadata_path=metadata_path,
        n_folds=3,
        num_boost_round=50,
    )

    # The model + metadata files exist
    assert model_path.exists()
    assert metadata_path.exists()
    # Metadata contains the right keys
    metadata = json.loads(metadata_path.read_text())
    assert "feature_columns" in metadata
    assert metadata["target_horizon_hours"] == 168  # 7d
    assert "trained_at" in metadata
    assert "git_sha" in metadata
    assert "cv_metrics" in metadata
    assert "mean_ic" in metadata["cv_metrics"]
    assert "mean_hit_rate" in metadata["cv_metrics"]
    assert "mean_rmse" in metadata["cv_metrics"]
    assert "mean_r_squared" in metadata["cv_metrics"]
    # The CV result is exposed
    assert "mean_ic" in result
    assert "mean_hit_rate" in result
    assert "mean_rmse" in result
    assert "mean_r_squared" in result
