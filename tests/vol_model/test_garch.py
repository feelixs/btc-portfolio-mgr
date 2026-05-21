from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from btc_portfolio_mgr.vol_model.garch import (
    fit_gjr_garch,
    forecast_24h_vol,
)
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC


def _simulated_garch_returns(n: int = 3000, seed: int = 42) -> pl.Series:
    """Simulate a series with GARCH-like persistence so the fitter can recover it."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.1, 0.85
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    rets = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = omega + alpha * rets[t - 1] ** 2 + beta * sigma2[t - 1]
        rets[t] = rng.normal(0, math.sqrt(sigma2[t]))
    return pl.Series(rets, dtype=pl.Float64)


def test_fit_gjr_garch_returns_param_dict_with_expected_keys() -> None:
    rets = _simulated_garch_returns()
    params = fit_gjr_garch(rets)
    assert isinstance(params, dict)
    for key in ("mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"):
        assert key in params, f"missing key {key} in params {list(params.keys())}"
    # Persistence (alpha + beta) is plausible
    persistence = params["alpha[1]"] + params["beta[1]"]
    assert 0.5 < persistence < 1.0


def test_fit_gjr_garch_rejects_empty_input() -> None:
    empty = pl.Series([], dtype=pl.Float64)
    with pytest.raises(ValueError):
        fit_gjr_garch(empty)


def test_fit_gjr_garch_rejects_too_short_input() -> None:
    short = pl.Series([0.01, 0.02, 0.03], dtype=pl.Float64)
    with pytest.raises(ValueError):
        fit_gjr_garch(short)


def test_forecast_24h_vol_returns_positive_finite_float() -> None:
    rets = _simulated_garch_returns()
    params = fit_gjr_garch(rets)
    vol = forecast_24h_vol(params, rets, spec=DEFAULT_SPEC)
    assert isinstance(vol, float)
    assert vol > 0
    assert math.isfinite(vol)


def test_forecast_24h_vol_at_specific_index() -> None:
    rets = _simulated_garch_returns(n=2000)
    params = fit_gjr_garch(rets)
    vol_end = forecast_24h_vol(params, rets, spec=DEFAULT_SPEC, last_obs_index=1999)
    vol_mid = forecast_24h_vol(params, rets, spec=DEFAULT_SPEC, last_obs_index=1000)
    assert vol_end > 0 and math.isfinite(vol_end)
    assert vol_mid > 0 and math.isfinite(vol_mid)


def test_forecast_reproduces_after_refit_roundtrip() -> None:
    """Fit, extract params, then reconstruct via fix() — forecasts must match."""
    rets = _simulated_garch_returns(n=2000)
    params = fit_gjr_garch(rets)
    vol1 = forecast_24h_vol(params, rets, spec=DEFAULT_SPEC, last_obs_index=1500)
    # Serialize through dict (simulates JSON round-trip)
    serialized = dict(params)
    vol2 = forecast_24h_vol(serialized, rets, spec=DEFAULT_SPEC, last_obs_index=1500)
    assert math.isclose(vol1, vol2, abs_tol=1e-9)
