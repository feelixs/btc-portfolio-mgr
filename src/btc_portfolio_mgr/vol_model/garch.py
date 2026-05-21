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


def forecast_24h_vol(
    params: dict[str, float],
    log_returns: pl.Series,
    spec: GarchSpec = DEFAULT_SPEC,
    scale_factor: float = SCALE_FACTOR,
    last_obs_index: int | None = None,
) -> float:
    """Forecast integrated 24h vol from fixed params + returns history.

    Reconstructs the arch model on the provided returns, fixes the params
    (no refit), and forecasts horizon=24. Sums per-hour variances and takes
    sqrt; the scale factor is unwound at the end.
    """
    scaled = log_returns.to_numpy() * scale_factor
    am = _build_arch_model(scaled, spec)
    # arch's fix() expects parameters in the same order as res.params produced.
    param_array = np.array(list(params.values()), dtype=np.float64)
    fixed_res = am.fix(param_array)
    if last_obs_index is None:
        fc = fixed_res.forecast(horizon=24, reindex=False)
    else:
        fc = fixed_res.forecast(horizon=24, start=last_obs_index, reindex=False)
    variance_row = fc.variance.values[-1]
    integrated_variance_scaled = float(np.sum(variance_row))
    integrated_variance = integrated_variance_scaled / (scale_factor**2)
    return math.sqrt(integrated_variance)
