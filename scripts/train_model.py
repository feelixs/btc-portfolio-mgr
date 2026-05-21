"""Train the BTC 24h-ahead return model: purged CV, then final fit on all data."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.inference import ModelArtifact, save_artifact
from btc_portfolio_mgr.model.target import build_dataset
from btc_portfolio_mgr.model.train import cross_validate, train_lightgbm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "btc_features.parquet"
DEFAULT_MODEL = REPO_ROOT / "models" / "btc_24h.txt"
DEFAULT_METADATA = REPO_ROOT / "models" / "btc_24h.metadata.json"

TARGET_HORIZON_HOURS = 24
EMBARGO_HOURS = 24


def run(
    prices_path: Path = DEFAULT_PRICES,
    features_path: Path = DEFAULT_FEATURES,
    model_path: Path = DEFAULT_MODEL,
    metadata_path: Path = DEFAULT_METADATA,
    n_folds: int = 5,
    num_boost_round: int = 500,
) -> dict[str, Any]:
    """Build dataset → purged CV → final fit on all data → save artifact."""
    prices = read_parquet(prices_path)
    features = pl.read_parquet(features_path)
    dataset = build_dataset(features, prices, horizon_hours=TARGET_HORIZON_HOURS)
    print(f"dataset: {dataset.height} rows after null-drop")

    cv = cross_validate(
        dataset,
        n_folds=n_folds,
        label_horizon_hours=TARGET_HORIZON_HOURS,
        embargo_hours=EMBARGO_HOURS,
        num_boost_round=num_boost_round,
    )
    print(f"CV (n_folds={n_folds}):")
    print(f"  IC:        per-fold {[f'{x:.4f}' for x in cv.fold_ic]} mean={cv.mean_ic:.4f}")
    print(f"  hit_rate:  per-fold {[f'{x:.4f}' for x in cv.fold_hit_rate]} mean={cv.mean_hit_rate:.4f}")
    print(f"  RMSE:      per-fold {[f'{x:.4f}' for x in cv.fold_rmse]} mean={cv.mean_rmse:.4f}")
    print(f"  R^2:       per-fold {[f'{x:.4f}' for x in cv.fold_r_squared]} mean={cv.mean_r_squared:.4f}")

    # Final model: train on ALL data.
    X = dataset.select(FEATURE_COLUMNS).to_numpy()
    y = dataset["target"].to_numpy()
    booster = train_lightgbm(X, y, num_boost_round=num_boost_round)

    artifact = ModelArtifact(
        booster=booster,
        feature_columns=list(FEATURE_COLUMNS),
        target_horizon_hours=TARGET_HORIZON_HOURS,
        trained_at=datetime.now(tz=timezone.utc),
    )
    save_artifact(artifact, model_path, metadata_path)
    print(f"saved model to {model_path}")
    print(f"saved metadata to {metadata_path}")

    return {
        "mean_ic": cv.mean_ic,
        "mean_hit_rate": cv.mean_hit_rate,
        "mean_rmse": cv.mean_rmse,
        "mean_r_squared": cv.mean_r_squared,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
