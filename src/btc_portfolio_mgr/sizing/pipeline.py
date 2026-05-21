"""End-to-end target-weight computation from artifacts + current state."""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from btc_portfolio_mgr.model.inference import (
    ModelArtifact,
    load_artifact,
    predict,
)
from btc_portfolio_mgr.sizing.params import SizingParams
from btc_portfolio_mgr.sizing.sizer import target_weight
from btc_portfolio_mgr.vol_model.inference import (
    VolArtifact,
    load_vol_artifact,
    predict_24h_vol,
)


def compute_target_weight(
    return_artifact: ModelArtifact,
    vol_artifact: VolArtifact,
    features_row: pl.DataFrame,
    log_returns: pl.Series,
    current_weight: float,
    params: SizingParams,
) -> float:
    """Compute target weight from already-loaded artifacts.

    `features_row` MUST be a 1-row polars DataFrame matching the return
    artifact's `feature_columns`. `log_returns` is the 1h log-return history
    up to (and including) the forecast point — used by the vol artifact.
    `current_weight` is the actual current position (caller-tracked).
    """
    if features_row.height != 1:
        raise ValueError(
            f"features_row must have exactly 1 row, got {features_row.height}"
        )
    mu_series = predict(return_artifact, features_row)
    mu = float(mu_series.to_numpy()[0])
    if not math.isfinite(mu):
        raise ValueError(f"return model produced non-finite prediction: mu={mu}")
    sigma = predict_24h_vol(vol_artifact, log_returns)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"vol model produced invalid sigma: {sigma}")
    return target_weight(
        mu=mu,
        sigma=sigma,
        current_weight=current_weight,
        params=params,
    )


def compute_target_weight_from_paths(
    return_model_path: Path,
    return_metadata_path: Path,
    vol_artifact_path: Path,
    features_row: pl.DataFrame,
    log_returns: pl.Series,
    current_weight: float,
    params: SizingParams,
) -> float:
    """Convenience wrapper: load both artifacts from disk, then compute."""
    return_artifact = load_artifact(return_model_path, return_metadata_path)
    vol_artifact = load_vol_artifact(vol_artifact_path)
    return compute_target_weight(
        return_artifact=return_artifact,
        vol_artifact=vol_artifact,
        features_row=features_row,
        log_returns=log_returns,
        current_weight=current_weight,
        params=params,
    )
