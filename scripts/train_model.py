"""Train the BTC 7d-ahead return model on daily-resampled features.

The target is the forward 7-day log return. To break label autocorrelation
(adjacent hourly rows share 167/168 hours of the same 7d label window), we
resample the feature dataset down to one row per day before purged CV. This
produces a statistically honest IC estimate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.git_info import current_git_sha
from btc_portfolio_mgr.model.inference import ModelArtifact, save_artifact
from btc_portfolio_mgr.model.target import build_dataset
from btc_portfolio_mgr.model.train import cross_validate, train_lightgbm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "btc_features.parquet"
DEFAULT_MODEL = REPO_ROOT / "models" / "btc_7d.txt"
DEFAULT_METADATA = REPO_ROOT / "models" / "btc_7d.metadata.json"

TARGET_HORIZON_HOURS = 168  # 7 days — the actual prediction horizon
SAMPLE_PERIOD_HOURS = 24  # train on one row per day to break label autocorrelation
# Purged-CV offsets expressed in ROW positions, not wall-clock hours. After
# daily resampling, 1 row == 1 day, so 7 rows = 7 days of label coverage.
PURGE_ROWS = TARGET_HORIZON_HOURS // SAMPLE_PERIOD_HOURS  # = 7
EMBARGO_ROWS = TARGET_HORIZON_HOURS // SAMPLE_PERIOD_HOURS  # = 7


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
    print(f"dataset: {dataset.height} hourly rows after null-drop")

    # Resample to one row per day (midnight UTC) so adjacent samples don't share
    # 167/168 hours of the same 7d forward label.
    dataset = dataset.filter(pl.col("timestamp").dt.hour() == 0)
    print(f"dataset: {dataset.height} daily rows after resampling")

    cv = cross_validate(
        dataset,
        n_folds=n_folds,
        label_horizon_hours=PURGE_ROWS,
        embargo_hours=EMBARGO_ROWS,
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
        git_sha=current_git_sha(REPO_ROOT),
        cv_metrics={
            "mean_ic": cv.mean_ic,
            "mean_hit_rate": cv.mean_hit_rate,
            "mean_rmse": cv.mean_rmse,
            "mean_r_squared": cv.mean_r_squared,
        },
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
