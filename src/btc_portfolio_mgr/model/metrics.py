"""Evaluation metrics for return-prediction models.

All functions accept (y_true, y_pred) as 1-D numpy arrays and return float.
Edge cases (constant inputs, empty arrays) return NaN rather than raising.
"""
from __future__ import annotations

import numpy as np


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between predictions and realized values.

    Returns NaN if either array is constant.
    """
    if y_true.size < 2:
        return float("nan")
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions whose sign matches the realized value's sign.

    Rows where y_true == 0 are excluded from the denominator.
    Returns NaN if no non-zero rows.
    """
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination on out-of-fold predictions.

    1 - SS_res / SS_tot. Negative when worse than mean baseline.
    Returns NaN if y_true is constant.
    """
    if y_true.size < 2:
        return float("nan")
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot
