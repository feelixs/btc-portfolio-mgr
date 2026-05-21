from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from btc_portfolio_mgr.vol_model.garch import fit_gjr_garch
from btc_portfolio_mgr.vol_model.inference import (
    VolArtifact,
    load_vol_artifact,
    predict_24h_vol,
    save_vol_artifact,
)
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR


def _toy_artifact_and_returns() -> tuple[VolArtifact, pl.Series]:
    rng = np.random.default_rng(7)
    rets = pl.Series(rng.normal(0, 0.01, 1500), dtype=pl.Float64)
    params = fit_gjr_garch(rets)
    artifact = VolArtifact(
        params=params,
        spec=DEFAULT_SPEC,
        scale_factor=SCALE_FACTOR,
        trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        git_sha="deadbee",
        eval_metrics={"qlike": -8.5, "mse_log": 0.04, "correlation": 0.6, "mae": 0.005},
        n_training_returns=1500,
    )
    return artifact, rets


def test_save_and_load_artifact_roundtrip(tmp_path: Path) -> None:
    artifact, _ = _toy_artifact_and_returns()
    path = tmp_path / "btc_vol.json"
    save_vol_artifact(artifact, path)
    assert path.exists()
    loaded = load_vol_artifact(path)
    # All fields round-trip
    assert loaded.params == artifact.params
    assert loaded.spec == artifact.spec
    assert loaded.scale_factor == artifact.scale_factor
    assert loaded.trained_at == artifact.trained_at
    assert loaded.git_sha == artifact.git_sha
    assert loaded.eval_metrics == artifact.eval_metrics
    assert loaded.n_training_returns == artifact.n_training_returns


def test_save_writes_human_readable_json(tmp_path: Path) -> None:
    artifact, _ = _toy_artifact_and_returns()
    path = tmp_path / "btc_vol.json"
    save_vol_artifact(artifact, path)
    data = json.loads(path.read_text())
    assert "params" in data
    assert "spec" in data
    assert "scale_factor" in data
    assert "trained_at" in data
    assert "git_sha" in data
    assert "eval_metrics" in data
    assert "n_training_returns" in data
    assert data["spec"]["vol"] == "GARCH"
    assert data["spec"]["o"] == 1


def test_predict_24h_vol_returns_positive_float() -> None:
    artifact, rets = _toy_artifact_and_returns()
    vol = predict_24h_vol(artifact, rets)
    assert isinstance(vol, float)
    assert vol > 0
    assert math.isfinite(vol)


def test_predict_24h_vol_at_specific_index() -> None:
    artifact, rets = _toy_artifact_and_returns()
    vol = predict_24h_vol(artifact, rets, last_obs_index=1000)
    assert vol > 0
    assert math.isfinite(vol)


def test_predict_after_roundtrip_matches(tmp_path: Path) -> None:
    artifact, rets = _toy_artifact_and_returns()
    path = tmp_path / "btc_vol.json"
    save_vol_artifact(artifact, path)
    loaded = load_vol_artifact(path)
    vol_original = predict_24h_vol(artifact, rets)
    vol_loaded = predict_24h_vol(loaded, rets)
    assert math.isclose(vol_original, vol_loaded, abs_tol=1e-9)
