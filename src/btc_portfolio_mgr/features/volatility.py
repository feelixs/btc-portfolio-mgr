"""Realized vol, skew, and excess kurtosis from 1h log returns."""
from __future__ import annotations

import numpy as np
import polars as pl

from btc_portfolio_mgr.features.returns import compute_log_return


def _validate_window(window_hours: int) -> None:
    if window_hours <= 0:
        raise ValueError(f"window_hours must be positive, got {window_hours}")


def compute_realized_vol(prices: pl.DataFrame, window_hours: int) -> pl.Series:
    """sqrt(sum of squared 1h log returns) over the rolling window.

    Returns null where the window does not contain `window_hours` non-null
    1h returns (i.e. across gaps).
    """
    _validate_window(window_hours)
    rets = compute_log_return(prices, lookback_hours=1)
    return (
        rets.pow(2)
        .rolling_sum(window_size=window_hours, min_samples=window_hours)
        .sqrt()
    )


def compute_realized_skew(prices: pl.DataFrame, window_hours: int) -> pl.Series:
    """Sample skew of 1h log returns over the window. Null when incomplete.

    `min_samples=window_hours` ensures null when the window contains any
    null return (i.e. across gaps).
    """
    _validate_window(window_hours)
    rets = compute_log_return(prices, lookback_hours=1)
    return rets.rolling_skew(
        window_size=window_hours, bias=False, min_samples=window_hours
    )


def compute_realized_kurt(prices: pl.DataFrame, window_hours: int) -> pl.Series:
    """Sample excess kurtosis of 1h log returns over the window.

    Polars has no rolling_kurt, so this uses a numpy loop. O(n * window).
    Null when the window contains any null return or when stdev == 0.
    """
    _validate_window(window_hours)
    rets = compute_log_return(prices, lookback_hours=1).to_numpy()
    n = len(rets)
    out = np.full(n, np.nan)
    for i in range(window_hours - 1, n):
        w = rets[i - window_hours + 1 : i + 1]
        if np.isnan(w).any():
            continue
        std = w.std(ddof=1)
        if std == 0 or not np.isfinite(std):
            continue
        z = (w - w.mean()) / std
        out[i] = float(np.power(z, 4).mean() - 3.0)
    return pl.Series(out, dtype=pl.Float64).fill_nan(None)
