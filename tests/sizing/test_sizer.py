from __future__ import annotations

import math

import pytest

from btc_portfolio_mgr.sizing.params import DEFAULT_PARAMS, SizingParams
from btc_portfolio_mgr.sizing.sizer import (
    apply_rebalance_threshold,
    compute_kelly_weight,
    target_weight,
)


# --- compute_kelly_weight ---


def test_kelly_weight_known_value() -> None:
    # mu = 0.005, sigma = 0.02, k = 0.20
    # raw = 0.20 * 0.005 / 0.02^2 = 2.5 -> clipped to 1.0
    w = compute_kelly_weight(
        mu=0.005, sigma=0.02, kelly_fraction=0.20, lower_bound=-1.0, upper_bound=1.0
    )
    assert w == 1.0


def test_kelly_weight_negative_mu_gives_negative_weight() -> None:
    # mu = -0.001, sigma = 0.02 -> raw = 0.20 * -0.001 / 0.0004 = -0.5 (within bounds)
    w = compute_kelly_weight(
        mu=-0.001, sigma=0.02, kelly_fraction=0.20, lower_bound=-1.0, upper_bound=1.0
    )
    assert math.isclose(w, -0.5)


def test_kelly_weight_zero_mu_gives_zero() -> None:
    w = compute_kelly_weight(
        mu=0.0, sigma=0.02, kelly_fraction=0.20, lower_bound=-1.0, upper_bound=1.0
    )
    assert w == 0.0


def test_kelly_weight_clips_at_upper_bound() -> None:
    w = compute_kelly_weight(
        mu=0.05, sigma=0.01, kelly_fraction=0.20, lower_bound=-1.0, upper_bound=1.0
    )
    assert w == 1.0


def test_kelly_weight_clips_at_lower_bound() -> None:
    w = compute_kelly_weight(
        mu=-0.05, sigma=0.01, kelly_fraction=0.20, lower_bound=-1.0, upper_bound=1.0
    )
    assert w == -1.0


def test_kelly_weight_kelly_fraction_scales_linearly() -> None:
    args = dict(mu=0.001, sigma=0.05, lower_bound=-1.0, upper_bound=1.0)
    w_full = compute_kelly_weight(kelly_fraction=1.0, **args)
    w_half = compute_kelly_weight(kelly_fraction=0.5, **args)
    assert math.isclose(w_half, w_full / 2)


def test_kelly_weight_sigma_zero_raises() -> None:
    with pytest.raises(ValueError):
        compute_kelly_weight(
            mu=0.005, sigma=0.0, kelly_fraction=0.20,
            lower_bound=-1.0, upper_bound=1.0,
        )


def test_kelly_weight_sigma_negative_raises() -> None:
    with pytest.raises(ValueError):
        compute_kelly_weight(
            mu=0.005, sigma=-0.01, kelly_fraction=0.20,
            lower_bound=-1.0, upper_bound=1.0,
        )


def test_kelly_weight_bounds_inverted_raises() -> None:
    with pytest.raises(ValueError):
        compute_kelly_weight(
            mu=0.001, sigma=0.02, kelly_fraction=0.20,
            lower_bound=1.0, upper_bound=-1.0,
        )


# --- apply_rebalance_threshold ---


def test_threshold_small_change_keeps_current() -> None:
    w = apply_rebalance_threshold(new_weight=0.51, current_weight=0.50, threshold=0.02)
    assert w == 0.50


def test_threshold_large_change_uses_new() -> None:
    w = apply_rebalance_threshold(new_weight=0.55, current_weight=0.50, threshold=0.02)
    assert w == 0.55


def test_threshold_exact_boundary_keeps_current() -> None:
    # |diff| == threshold (not strictly greater) -> keep current.
    w = apply_rebalance_threshold(new_weight=0.52, current_weight=0.50, threshold=0.02)
    assert w == 0.50


def test_threshold_zero_always_rebalances() -> None:
    w = apply_rebalance_threshold(new_weight=0.501, current_weight=0.50, threshold=0.0)
    assert w == 0.501


def test_threshold_negative_raises() -> None:
    with pytest.raises(ValueError):
        apply_rebalance_threshold(new_weight=0.5, current_weight=0.4, threshold=-0.01)


# --- target_weight (composer) ---


def test_target_weight_uses_default_params() -> None:
    # raw = 0.20 * 0.001 / 0.0004 = 0.5; current 0.0; diff 0.5 > 0.02 -> use 0.5.
    w = target_weight(mu=0.001, sigma=0.02, current_weight=0.0, params=DEFAULT_PARAMS)
    assert math.isclose(w, 0.5)


def test_target_weight_respects_threshold() -> None:
    # raw new = 0.5, current = 0.49, diff = 0.01 < 0.02 -> stay.
    w = target_weight(mu=0.001, sigma=0.02, current_weight=0.49, params=DEFAULT_PARAMS)
    assert w == 0.49


def test_target_weight_respects_bounds() -> None:
    w = target_weight(mu=1.0, sigma=0.01, current_weight=0.0, params=DEFAULT_PARAMS)
    assert w == 1.0


def test_target_weight_custom_params() -> None:
    # k=1.0 -> raw = 1.0 * 0.001 / 0.0004 = 2.5 -> clipped at 1.0
    full_kelly = SizingParams(kelly_fraction=1.0)
    w = target_weight(mu=0.001, sigma=0.02, current_weight=0.0, params=full_kelly)
    assert w == 1.0
