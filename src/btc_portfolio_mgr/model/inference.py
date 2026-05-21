"""Model artifact persistence and inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import polars as pl


@dataclass(frozen=True)
class ModelArtifact:
    booster: lgb.Booster
    feature_columns: list[str]
    target_horizon_hours: int
    trained_at: datetime


def save_artifact(
    artifact: ModelArtifact, model_path: Path, metadata_path: Path
) -> None:
    """Save booster + JSON metadata sidecar."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.booster.save_model(str(model_path))
    metadata = {
        "feature_columns": artifact.feature_columns,
        "target_horizon_hours": artifact.target_horizon_hours,
        "trained_at": artifact.trained_at.isoformat(),
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def load_artifact(model_path: Path, metadata_path: Path) -> ModelArtifact:
    """Load booster + JSON metadata into a ModelArtifact."""
    booster = lgb.Booster(model_file=str(model_path))
    with metadata_path.open() as f:
        metadata = json.load(f)
    return ModelArtifact(
        booster=booster,
        feature_columns=metadata["feature_columns"],
        target_horizon_hours=int(metadata["target_horizon_hours"]),
        trained_at=datetime.fromisoformat(metadata["trained_at"]),
    )


def predict(artifact: ModelArtifact, features: pl.DataFrame) -> pl.Series:
    """Predict on a feature DataFrame. Columns must match training feature set."""
    missing = [c for c in artifact.feature_columns if c not in features.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    X = features.select(artifact.feature_columns).to_numpy()
    preds = artifact.booster.predict(X)
    return pl.Series("prediction", preds, dtype=pl.Float64)
