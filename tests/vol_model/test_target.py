from __future__ import annotations

import math

import polars as pl
import pytest

from btc_portfolio_mgr.vol_model.target import compute_realized_24h_vol


def test_realized_24h_vol_constant_returns_is_constant() -> None:
    returns = pl.Series([0.01] * 30, dtype=pl.Float64)
    vol = compute_realized_24h_vol(returns, window_hours=24)
    values = vol.to_list()
    assert values[22] is None
    expected = 0.01 * math.sqrt(24)
    assert math.isclose(values[23], expected)
    assert math.isclose(values[29], expected)


def test_realized_24h_vol_zero_returns_is_zero() -> None:
    returns = pl.Series([0.0] * 30, dtype=pl.Float64)
    vol = compute_realized_24h_vol(returns, window_hours=24)
    values = vol.to_list()
    assert values[23] == 0.0


def test_realized_24h_vol_null_propagates() -> None:
    returns = pl.Series([0.01] * 5 + [None] + [0.01] * 24, dtype=pl.Float64)
    vol = compute_realized_24h_vol(returns, window_hours=24)
    values = vol.to_list()
    # Window at index 23 contains null at index 5 -> null.
    assert values[23] is None
    # By index 29, window is [6..29] which excludes the null -> valid.
    assert values[29] is not None


def test_realized_24h_vol_invalid_window_raises() -> None:
    returns = pl.Series([0.01] * 10, dtype=pl.Float64)
    with pytest.raises(ValueError):
        compute_realized_24h_vol(returns, window_hours=0)
    with pytest.raises(ValueError):
        compute_realized_24h_vol(returns, window_hours=-1)
