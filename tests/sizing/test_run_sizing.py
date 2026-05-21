from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from btc_portfolio_mgr.data.storage import write_parquet
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.model.inference import ModelArtifact, save_artifact
from btc_portfolio_mgr.model.train import train_lightgbm
from btc_portfolio_mgr.vol_model.garch import fit_gjr_garch
from btc_portfolio_mgr.vol_model.inference import VolArtifact, save_vol_artifact
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR
from tests.fixtures.synthetic_prices import hourly


def test_run_sizing_prints_target_weight(tmp_path: Path) -> None:
    from scripts import run_sizing as rs

    # Synthetic prices long enough for features (2160h lookback) + GARCH.
    n = 2500
    rng = np.random.default_rng(13)
    returns = rng.normal(0, 0.01, n - 1)
    prices_list = [100.0]
    for r in returns:
        prices_list.append(prices_list[-1] * float(math.exp(r)))
    prices = hourly(prices_list)

    prices_path = tmp_path / "btc_hourly.parquet"
    features_path = tmp_path / "btc_features.parquet"
    return_model_path = tmp_path / "btc_24h.txt"
    return_meta_path = tmp_path / "btc_24h.metadata.json"
    vol_path = tmp_path / "btc_vol.json"

    write_parquet(prices, prices_path)
    features = compose_features(prices)
    features.write_parquet(features_path)

    # Train a toy return model on the features that are non-null
    valid_features = features.drop_nulls()
    rng2 = np.random.default_rng(0)
    X = valid_features.select(FEATURE_COLUMNS).to_numpy()
    y = rng2.normal(0, 0.01, X.shape[0])
    booster = train_lightgbm(X, y, num_boost_round=50)
    return_artifact = ModelArtifact(
        booster=booster,
        feature_columns=list(FEATURE_COLUMNS),
        target_horizon_hours=24,
        trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        git_sha="aaaaaaa",
        cv_metrics={"mean_ic": 0.0, "mean_hit_rate": 0.5, "mean_rmse": 0.01, "mean_r_squared": 0.0},
    )
    save_artifact(return_artifact, return_model_path, return_meta_path)

    # Train a toy vol model
    from btc_portfolio_mgr.vol_model.returns import extract_log_returns

    log_returns = extract_log_returns(prices)["log_return"]
    vol_params = fit_gjr_garch(log_returns)
    vol_artifact = VolArtifact(
        params=vol_params,
        spec=DEFAULT_SPEC,
        scale_factor=SCALE_FACTOR,
        trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        git_sha="bbbbbbb",
        eval_metrics={"qlike": -8.0, "mse_log": 0.05, "correlation": 0.5, "mae": 0.006},
        n_training_returns=int(log_returns.len()),
    )
    save_vol_artifact(vol_artifact, vol_path)

    result = rs.run(
        prices_path=prices_path,
        features_path=features_path,
        return_model_path=return_model_path,
        return_metadata_path=return_meta_path,
        vol_artifact_path=vol_path,
        current_weight=0.0,
    )

    assert "target_weight" in result
    assert isinstance(result["target_weight"], float)
    assert -1.0 <= result["target_weight"] <= 1.0
    assert "mu" in result
    assert "sigma" in result
    assert result["sigma"] > 0
