from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.inference import ModelArtifact
from btc_portfolio_mgr.model.train import train_lightgbm
from btc_portfolio_mgr.sizing.params import DEFAULT_PARAMS, SizingParams
from btc_portfolio_mgr.sizing.pipeline import (
    compute_target_weight,
    compute_target_weight_from_paths,
)
from btc_portfolio_mgr.vol_model.garch import fit_gjr_garch
from btc_portfolio_mgr.vol_model.inference import (
    VolArtifact,
    save_vol_artifact,
)
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR


def _toy_return_artifact() -> ModelArtifact:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (500, len(FEATURE_COLUMNS)))
    y = 0.3 * X[:, 0] + rng.normal(0, 0.1, 500)
    booster = train_lightgbm(X, y, num_boost_round=50)
    return ModelArtifact(
        booster=booster,
        feature_columns=list(FEATURE_COLUMNS),
        target_horizon_hours=24,
        trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        git_sha="aaaaaaa",
        cv_metrics={"mean_ic": 0.05, "mean_hit_rate": 0.52, "mean_rmse": 0.015, "mean_r_squared": 0.0},
    )


def _toy_vol_artifact_and_returns() -> tuple[VolArtifact, pl.Series]:
    rng = np.random.default_rng(7)
    rets = pl.Series(rng.normal(0, 0.01, 1500), dtype=pl.Float64)
    params = fit_gjr_garch(rets)
    return (
        VolArtifact(
            params=params,
            spec=DEFAULT_SPEC,
            scale_factor=SCALE_FACTOR,
            trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
            git_sha="bbbbbbb",
            eval_metrics={"qlike": -8.0, "mse_log": 0.05, "correlation": 0.5, "mae": 0.006},
            n_training_returns=1500,
        ),
        rets,
    )


def _toy_features_row() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    row = {col: [rng.normal(0, 1)] for col in FEATURE_COLUMNS}
    return pl.DataFrame(row)


def test_compute_target_weight_returns_float_in_bounds() -> None:
    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    features = _toy_features_row()
    w = compute_target_weight(
        return_artifact=return_artifact,
        vol_artifact=vol_artifact,
        features_row=features,
        log_returns=rets,
        current_weight=0.0,
        params=DEFAULT_PARAMS,
    )
    assert isinstance(w, float)
    assert -1.0 <= w <= 1.0


def test_compute_target_weight_respects_threshold() -> None:
    """A small change between target and current should keep current."""
    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    features = _toy_features_row()
    w_initial = compute_target_weight(
        return_artifact=return_artifact,
        vol_artifact=vol_artifact,
        features_row=features,
        log_returns=rets,
        current_weight=0.0,
        params=DEFAULT_PARAMS,
    )
    # Call again with current = w_initial; same inputs should not change it (diff = 0).
    w_repeat = compute_target_weight(
        return_artifact=return_artifact,
        vol_artifact=vol_artifact,
        features_row=features,
        log_returns=rets,
        current_weight=w_initial,
        params=DEFAULT_PARAMS,
    )
    assert w_repeat == w_initial


def test_compute_target_weight_features_row_must_be_single_row() -> None:
    import pytest

    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    rng = np.random.default_rng(0)
    multi_row = pl.DataFrame({col: rng.normal(0, 1, 3).tolist() for col in FEATURE_COLUMNS})
    with pytest.raises(ValueError):
        compute_target_weight(
            return_artifact=return_artifact,
            vol_artifact=vol_artifact,
            features_row=multi_row,
            log_returns=rets,
            current_weight=0.0,
            params=DEFAULT_PARAMS,
        )


def test_compute_target_weight_from_paths(tmp_path: Path) -> None:
    from btc_portfolio_mgr.model.inference import save_artifact

    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    features = _toy_features_row()

    return_model_path = tmp_path / "ret.txt"
    return_meta_path = tmp_path / "ret.metadata.json"
    vol_path = tmp_path / "vol.json"
    save_artifact(return_artifact, return_model_path, return_meta_path)
    save_vol_artifact(vol_artifact, vol_path)

    w = compute_target_weight_from_paths(
        return_model_path=return_model_path,
        return_metadata_path=return_meta_path,
        vol_artifact_path=vol_path,
        features_row=features,
        log_returns=rets,
        current_weight=0.0,
        params=DEFAULT_PARAMS,
    )
    assert isinstance(w, float)
    assert -1.0 <= w <= 1.0


def test_compute_target_weight_raises_on_nan_mu() -> None:
    import pytest

    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    features = _toy_features_row()

    # Patch predict to return a NaN Series
    import btc_portfolio_mgr.sizing.pipeline as pipeline_module

    def fake_predict(_artifact, _features):
        return pl.Series("prediction", [float("nan")], dtype=pl.Float64)

    original_predict = pipeline_module.predict
    pipeline_module.predict = fake_predict
    try:
        with pytest.raises(ValueError, match="non-finite prediction"):
            compute_target_weight(
                return_artifact=return_artifact,
                vol_artifact=vol_artifact,
                features_row=features,
                log_returns=rets,
                current_weight=0.0,
                params=DEFAULT_PARAMS,
            )
    finally:
        pipeline_module.predict = original_predict
