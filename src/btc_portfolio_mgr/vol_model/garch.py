"""GJR-GARCH fit (returns params dict) and 24h vol forecast (from params + returns)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from arch import arch_model

from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR, GarchSpec

MIN_RETURNS_FOR_FIT = 500


def _build_arch_model(scaled_returns: np.ndarray, spec: GarchSpec) -> Any:
    return arch_model(
        scaled_returns,
        mean=spec.mean,  # type: ignore[arg-type]
        vol=spec.vol,  # type: ignore[arg-type]
        p=spec.p,
        o=spec.o,
        q=spec.q,
        dist=spec.dist,  # type: ignore[arg-type]
        rescale=False,
    )


def fit_gjr_garch(
    log_returns: pl.Series, spec: GarchSpec = DEFAULT_SPEC
) -> dict[str, float]:
    """Fit GJR-GARCH on log returns scaled × SCALE_FACTOR. Returns params dict."""
    if log_returns.len() < MIN_RETURNS_FOR_FIT:
        raise ValueError(
            f"need >= {MIN_RETURNS_FOR_FIT} returns to fit GJR-GARCH, got {log_returns.len()}"
        )
    scaled = log_returns.to_numpy() * SCALE_FACTOR
    am = _build_arch_model(scaled, spec)
    res = am.fit(disp="off")
    # Preserve arch's parameter ordering for later am.fix() reconstruction
    return {str(k): float(v) for k, v in res.params.items()}


def forecast_integrated_vols_batch(
    params: dict[str, float],
    log_returns: pl.Series,
    horizon_hours: int,
    start_index: int,
    spec: GarchSpec = DEFAULT_SPEC,
    scale_factor: float = SCALE_FACTOR,
) -> np.ndarray:
    """Forecast integrated vol at every observation from `start_index` onward.

    Returns a 1-D numpy array of integrated vol values (one entry per index
    from `start_index` to the end of the series, inclusive). Much faster than
    calling `forecast_24h_vol()` in a loop because the arch model is built and
    fixed once and arch computes all anchor-point forecasts internally.

    Use this for in-sample evaluation / backtesting (Phase 6) where the model
    must walk over many anchor points. For single-point inference at the
    current bar, use `forecast_24h_vol`.
    """
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive, got {horizon_hours}")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    scaled = log_returns.to_numpy() * scale_factor
    am = _build_arch_model(scaled, spec)
    param_array = np.array(list(params.values()), dtype=np.float64)
    fixed_res = am.fix(param_array)
    fc = fixed_res.forecast(
        horizon=horizon_hours, start=start_index, reindex=False
    )
    variance_matrix = fc.variance.values  # shape: (n - start_index, horizon_hours)
    integrated_var_scaled = variance_matrix.sum(axis=1)
    integrated_var = integrated_var_scaled / (scale_factor**2)
    return np.sqrt(integrated_var)


def forecast_24h_vol(
    params: dict[str, float],
    log_returns: pl.Series,
    spec: GarchSpec = DEFAULT_SPEC,
    scale_factor: float = SCALE_FACTOR,
    last_obs_index: int | None = None,
    horizon_hours: int = 24,
) -> float:
    """Forecast integrated vol over `horizon_hours` from fixed params + returns.

    Despite the historical name, this function supports any forecast horizon
    via the `horizon_hours` parameter. It reconstructs the arch model on the
    provided returns, fixes the params (no refit), forecasts that many steps
    ahead, sums the per-hour variances, and takes the sqrt. The scale factor
    is unwound at the end.
    """
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive, got {horizon_hours}")
    scaled = log_returns.to_numpy() * scale_factor
    am = _build_arch_model(scaled, spec)
    # arch's fix() expects parameters in the same order as res.params produced.
    param_array = np.array(list(params.values()), dtype=np.float64)
    fixed_res = am.fix(param_array)
    if last_obs_index is None:
        fc = fixed_res.forecast(horizon=horizon_hours, reindex=False)
    else:
        fc = fixed_res.forecast(
            horizon=horizon_hours, start=last_obs_index, reindex=False
        )
    variance_row = fc.variance.values[-1]
    integrated_variance_scaled = float(np.sum(variance_row))
    integrated_variance = integrated_variance_scaled / (scale_factor**2)
    return math.sqrt(integrated_variance)
