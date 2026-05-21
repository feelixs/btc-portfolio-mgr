"""Regime features: MA z-score and Kaufman Efficiency Ratio."""
from __future__ import annotations

import numpy as np
import polars as pl


def _validate_window(window_hours: int) -> None:
    if window_hours <= 0:
        raise ValueError(f"window_hours must be positive, got {window_hours}")


def compute_ma_zscore(prices: pl.DataFrame, window_hours: int) -> pl.Series:
    """(price - rolling_mean) / rolling_std over the window.

    Null where rolling stats are null (gap) or std == 0 (constant prices,
    which would give 0/0 = NaN or x/0 = inf — both are mapped to null).
    """
    _validate_window(window_hours)
    p = prices["price"]
    mean = p.rolling_mean(window_size=window_hours, min_samples=window_hours)
    std = p.rolling_std(window_size=window_hours, min_samples=window_hours, ddof=1)
    raw = ((p - mean) / std).to_numpy()
    raw = np.where(np.isfinite(raw), raw, np.nan)
    return pl.Series(raw, dtype=pl.Float64).fill_nan(None)


def compute_kaufman_efficiency(
    prices: pl.DataFrame, window_hours: int
) -> pl.Series:
    """Kaufman Efficiency Ratio over the window.

    KER = |price_t - price_{t-window}| / sum(|price_i - price_{i-1}|)
    over i in (t-window+1 .. t). Range [0, 1]. Null when any input in the
    window is null. When net == 0 AND path == 0, returns 0.0 (no movement).

    Implementation: numpy loop. O(n * window).
    """
    _validate_window(window_hours)
    p = prices["price"].to_numpy().astype(np.float64)
    n = len(p)
    out = np.full(n, np.nan)
    # Need window_hours + 1 contiguous non-null prices to compute window_hours diffs.
    for i in range(window_hours, n):
        w = p[i - window_hours : i + 1]
        if np.isnan(w).any():
            continue
        net = abs(w[-1] - w[0])
        path = float(np.abs(np.diff(w)).sum())
        if path == 0:
            out[i] = 0.0
            continue
        out[i] = net / path
    return pl.Series(out, dtype=pl.Float64).fill_nan(None)
