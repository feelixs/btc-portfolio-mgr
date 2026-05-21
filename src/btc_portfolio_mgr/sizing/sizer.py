"""Fractional Kelly weight + rebalance threshold + composition."""
from __future__ import annotations

from btc_portfolio_mgr.sizing.params import SizingParams


def compute_kelly_weight(
    mu: float,
    sigma: float,
    kelly_fraction: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """Fractional Kelly weight: clip(k · μ / σ², [lower, upper]).

    μ and σ must be on the same time horizon (i.e. σ² is the variance over the
    same window as μ). σ must be positive; lower_bound < upper_bound.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if lower_bound >= upper_bound:
        raise ValueError(
            f"lower_bound ({lower_bound}) must be < upper_bound ({upper_bound})"
        )
    raw = kelly_fraction * mu / (sigma ** 2)
    if raw < lower_bound:
        return lower_bound
    if raw > upper_bound:
        return upper_bound
    return raw


def apply_rebalance_threshold(
    new_weight: float, current_weight: float, threshold: float
) -> float:
    """Return new_weight if |new - current| > threshold; otherwise current_weight."""
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    # Special case: threshold=0 means always rebalance
    if threshold == 0:
        return new_weight
    # For threshold > 0: rebalance only if diff strictly exceeds threshold.
    # Use 1e-9 tolerance for floating point comparison.
    if abs(new_weight - current_weight) > threshold + 1e-9:
        return new_weight
    return current_weight


def target_weight(
    mu: float, sigma: float, current_weight: float, params: SizingParams
) -> float:
    """Compose Kelly weight + bound clipping + rebalance threshold."""
    raw = compute_kelly_weight(
        mu=mu,
        sigma=sigma,
        kelly_fraction=params.kelly_fraction,
        lower_bound=params.lower_bound,
        upper_bound=params.upper_bound,
    )
    return apply_rebalance_threshold(
        new_weight=raw,
        current_weight=current_weight,
        threshold=params.rebalance_threshold,
    )
