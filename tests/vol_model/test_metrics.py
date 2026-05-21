from __future__ import annotations

import math

import numpy as np

from btc_portfolio_mgr.vol_model.metrics import (
    compute_qlike,
    compute_mse_log,
    compute_vol_correlation,
    compute_mae_vol,
)


def test_qlike_perfect_prediction() -> None:
    realized = np.array([0.01, 0.02, 0.03])
    predicted = np.array([0.01, 0.02, 0.03])
    expected = float(np.mean(np.log(realized ** 2) + 1))
    assert math.isclose(compute_qlike(realized, predicted), expected)


def test_qlike_penalizes_underestimation_more_than_overestimation() -> None:
    realized = np.array([0.02])
    under = np.array([0.01])
    over = np.array([0.04])
    qlike_under = compute_qlike(realized, under)
    qlike_over = compute_qlike(realized, over)
    assert qlike_under > qlike_over


def test_qlike_invalid_vol_is_nan() -> None:
    realized = np.array([0.02, 0.03])
    bad_pred = np.array([0.02, 0.0])
    assert math.isnan(compute_qlike(realized, bad_pred))


def test_mse_log_perfect_is_zero() -> None:
    realized = np.array([0.01, 0.02, 0.03])
    predicted = realized.copy()
    assert compute_mse_log(realized, predicted) == 0.0


def test_mse_log_known_value() -> None:
    realized = np.array([0.01, 0.02])
    predicted = np.array([0.02, 0.01])
    expected = math.log(2) ** 2
    assert math.isclose(compute_mse_log(realized, predicted), expected)


def test_mse_log_invalid_vol_is_nan() -> None:
    realized = np.array([0.0, 0.01])
    predicted = np.array([0.01, 0.01])
    assert math.isnan(compute_mse_log(realized, predicted))


def test_vol_correlation_perfect() -> None:
    realized = np.array([0.01, 0.02, 0.03, 0.04])
    predicted = realized * 2
    assert math.isclose(compute_vol_correlation(realized, predicted), 1.0, abs_tol=1e-9)


def test_vol_correlation_constant_is_nan() -> None:
    realized = np.array([0.01, 0.02, 0.03])
    predicted = np.array([0.02, 0.02, 0.02])
    assert math.isnan(compute_vol_correlation(realized, predicted))


def test_mae_vol_perfect_is_zero() -> None:
    realized = np.array([0.01, 0.02, 0.03])
    predicted = realized.copy()
    assert compute_mae_vol(realized, predicted) == 0.0


def test_mae_vol_known_value() -> None:
    realized = np.array([0.01, 0.02, 0.03])
    predicted = np.array([0.02, 0.02, 0.02])
    expected = (0.01 + 0.0 + 0.01) / 3
    assert math.isclose(compute_mae_vol(realized, predicted), expected)
