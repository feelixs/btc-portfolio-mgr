"""Backward-looking realized 24h vol — used as the evaluation target."""
from __future__ import annotations

import polars as pl


def compute_realized_24h_vol(
    returns: pl.Series, window_hours: int = 24
) -> pl.Series:
    """sqrt(sum of squared 1h returns) over rolling backward window.

    Returns null where the window contains any null return (gap) or fewer
    than `window_hours` observations.
    """
    if window_hours <= 0:
        raise ValueError(f"window_hours must be positive, got {window_hours}")
    return (
        returns.pow(2)
        .rolling_sum(window_size=window_hours, min_samples=window_hours)
        .sqrt()
    )
