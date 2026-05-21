"""Volatility-forecast evaluation metrics.

All accept (realized_vol, predicted_vol) 1-D numpy arrays. Edge cases
(non-positive vols, empty, constant) return NaN.
"""
from __future__ import annotations

import numpy as np


def _valid(realized: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return (
        (realized > 0)
        & (predicted > 0)
        & np.isfinite(realized)
        & np.isfinite(predicted)
    )


def compute_qlike(realized: np.ndarray, predicted: np.ndarray) -> float:
    """QLIKE = mean(log(σ̂²) + σ²_realized / σ̂²). Lower is better.

    Penalizes underestimation more than overestimation.
    """
    mask = _valid(realized, predicted)
    if mask.sum() != len(realized):
        return float("nan")
    r = realized[mask]
    p = predicted[mask]
    return float(np.mean(np.log(p ** 2) + (r ** 2) / (p ** 2)))


def compute_mse_log(realized: np.ndarray, predicted: np.ndarray) -> float:
    """MSE of log(vol). Symmetric in log-space."""
    mask = _valid(realized, predicted)
    if mask.sum() != len(realized):
        return float("nan")
    return float(np.mean((np.log(realized[mask]) - np.log(predicted[mask])) ** 2))


def compute_vol_correlation(realized: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation between realized and predicted vol."""
    mask = _valid(realized, predicted)
    if mask.sum() < 2:
        return float("nan")
    r = realized[mask]
    p = predicted[mask]
    if r.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(r, p)[0, 1])


def compute_mae_vol(realized: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error in vol units."""
    mask = _valid(realized, predicted)
    if mask.sum() != len(realized):
        return float("nan")
    return float(np.mean(np.abs(realized[mask] - predicted[mask])))
