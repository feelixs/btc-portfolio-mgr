"""LightGBM training wrapper and purged-CV evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import polars as pl

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.cv import purged_kfold_indices
from btc_portfolio_mgr.model.metrics import (
    compute_hit_rate,
    compute_ic,
    compute_r_squared,
    compute_rmse,
)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "huber",
    "alpha": 0.02,  # threshold in residual-space; ~1x typical 24h log-return magnitude
    "metric": "huber",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
) -> lgb.Booster:
    """Fit a LightGBM regressor with Huber loss. Returns the booster."""
    merged = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(X_train, label=y_train)
    return lgb.train(
        merged,
        train_set,
        num_boost_round=num_boost_round,
    )


@dataclass(frozen=True)
class CVResult:
    fold_ic: list[float]
    fold_hit_rate: list[float]
    fold_rmse: list[float]
    fold_r_squared: list[float]
    mean_ic: float
    mean_hit_rate: float
    mean_rmse: float
    mean_r_squared: float
    oof_predictions: np.ndarray
    oof_mask: np.ndarray


def cross_validate(
    dataset: pl.DataFrame,
    n_folds: int = 5,
    label_horizon_hours: int = 24,
    embargo_hours: int = 24,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
) -> CVResult:
    """Run purged k-fold CV. Returns per-fold metrics + OOF predictions."""
    X = dataset.select(FEATURE_COLUMNS).to_numpy()
    y = dataset["target"].to_numpy()
    n = X.shape[0]
    oof_preds = np.full(n, np.nan)
    oof_mask = np.zeros(n, dtype=bool)
    fold_ic: list[float] = []
    fold_hit_rate: list[float] = []
    fold_rmse: list[float] = []
    fold_r_squared: list[float] = []
    for train_idx, test_idx in purged_kfold_indices(
        n=n,
        n_folds=n_folds,
        label_horizon_hours=label_horizon_hours,
        embargo_hours=embargo_hours,
    ):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        booster = train_lightgbm(X_train, y_train, params=params, num_boost_round=num_boost_round)
        preds = cast(np.ndarray, booster.predict(X_test))
        oof_preds[test_idx] = preds
        oof_mask[test_idx] = True
        fold_ic.append(compute_ic(y_test, preds))
        fold_hit_rate.append(compute_hit_rate(y_test, preds))
        fold_rmse.append(compute_rmse(y_test, preds))
        fold_r_squared.append(compute_r_squared(y_test, preds))
    return CVResult(
        fold_ic=fold_ic,
        fold_hit_rate=fold_hit_rate,
        fold_rmse=fold_rmse,
        fold_r_squared=fold_r_squared,
        mean_ic=float(np.nanmean(fold_ic)),
        mean_hit_rate=float(np.nanmean(fold_hit_rate)),
        mean_rmse=float(np.nanmean(fold_rmse)),
        mean_r_squared=float(np.nanmean(fold_r_squared)),
        oof_predictions=oof_preds,
        oof_mask=oof_mask,
    )
