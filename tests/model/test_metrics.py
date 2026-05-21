from __future__ import annotations

import math

import numpy as np
import pytest

from btc_portfolio_mgr.model.metrics import (
    compute_hit_rate,
    compute_ic,
    compute_r_squared,
    compute_rmse,
)


def test_ic_perfect_correlation_is_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # scaled by 2
    assert math.isclose(compute_ic(y_true, y_pred), 1.0, abs_tol=1e-9)


def test_ic_anti_correlation_is_negative_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert math.isclose(compute_ic(y_true, y_pred), -1.0, abs_tol=1e-9)


def test_ic_constant_pred_is_nan() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    assert math.isnan(compute_ic(y_true, y_pred))


def test_hit_rate_all_same_sign() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.04])
    y_pred = np.array([0.005, -0.01, 0.02, -0.03])  # all same sign
    assert compute_hit_rate(y_true, y_pred) == 1.0


def test_hit_rate_half_correct() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.04])
    y_pred = np.array([0.005, 0.01, 0.02, 0.03])  # 2 of 4 same sign
    assert compute_hit_rate(y_true, y_pred) == 0.5


def test_hit_rate_zeros_excluded() -> None:
    y_true = np.array([0.0, 0.01, -0.01])
    y_pred = np.array([0.5, 0.005, -0.005])  # row 0 excluded (y_true is 0)
    assert compute_hit_rate(y_true, y_pred) == 1.0


def test_rmse_perfect_prediction_is_zero() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert compute_rmse(y_true, y_pred) == 0.0


def test_rmse_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    # squared errors: 1, 0, 1 -> mean 2/3 -> sqrt = sqrt(2/3)
    assert math.isclose(compute_rmse(y_true, y_pred), math.sqrt(2 / 3))


def test_r_squared_perfect_is_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_r_squared(y_true, y_pred) == 1.0


def test_r_squared_mean_baseline_is_zero() -> None:
    # If pred = mean(y_true), R^2 = 0
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert math.isclose(compute_r_squared(y_true, y_pred), 0.0, abs_tol=1e-9)


def test_r_squared_can_be_negative() -> None:
    # Predictions worse than mean baseline -> negative R^2
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # anti-correlated
    assert compute_r_squared(y_true, y_pred) < 0


def test_r_squared_constant_true_is_nan() -> None:
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 1.0, 1.1])
    assert math.isnan(compute_r_squared(y_true, y_pred))
