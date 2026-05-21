# Phase 4: GJR-GARCH Vol Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a GJR-GARCH(1,1) model with Student's t innovations on hourly BTC log returns, forecast the integrated 24-hour realized vol, and persist a JSON-only VolArtifact for Phase 5 sizing to consume alongside the return model's μ̂.

**Architecture:** Use the `arch` library to fit GJR-GARCH(p=1, o=1, q=1) with Student's t distribution on all available hourly log returns (scaled ×100 for numerical stability). Persist only the fitted parameters as JSON (no pickle — security-driven design choice). At inference time, reconstruct the model via `arch_model(...).fix(params)` using the current returns history and call `.forecast(horizon=24)`, then sum the per-hour variances and take sqrt to get σ̂_24h.

**Tech Stack:** `arch>=7.0` (new dependency — Kevin Sheppard's library, the industry standard), numpy, polars. No pickle. No new dependency beyond `arch`.

---

## Phase status (4 of 7)

| Phase | Status |
|---|---|
| 1. Data layer | ✅ shipped |
| 2. Features | ✅ shipped |
| 3. Return model (LightGBM) | ✅ shipped |
| **4. Vol model (GJR-GARCH)** | **this plan** |
| 5. Sizing (Kelly + vol-target) | next |
| 6. Backtest (walk-forward) | after Phase 5 |
| 7. Execution | after Phase 6 |

---

## Key design decisions

- **Model**: GJR-GARCH(p=1, o=1, q=1) — asymmetric variant capturing the leverage effect.
- **Mean**: Constant.
- **Distribution**: Student's t.
- **Estimation window**: all available hourly log returns.
- **Scale**: multiply returns ×100 before fitting (percentage points). Divide back when computing vol forecasts.
- **Forecast target**: integrated 24h vol = `sqrt(sum(expected variance over h=1..24))`. Same time scale as the Phase 3 return model.
- **Evaluation**: in-sample evaluation for v1 (proper walk-forward refit is Phase 6). Documented caveat.
- **Metrics**: QLIKE, MSE on log(vol), Pearson correlation of σ̂ with realized 24h vol, MAE on realized vol.
- **Persistence**: JSON-only sidecar (params dict + model spec + provenance). At inference time, the artifact + current returns history reconstruct the forecaster via `am.fix(params)`. **No pickle** — security-conscious design choice; arch results aren't audit-friendly when pickled.

---

## File structure

```
btc-portfolio-mgr/
├── src/btc_portfolio_mgr/
│   ├── data/             # (Phase 1)
│   ├── features/         # (Phase 2)
│   ├── model/            # (Phase 3 — return model)
│   └── vol_model/        # NEW
│       ├── __init__.py
│       ├── spec.py           # GarchSpec dataclass + DEFAULT_SPEC + SCALE_FACTOR
│       ├── returns.py        # extract_log_returns: 1h log returns from prices parquet
│       ├── target.py         # compute_realized_24h_vol: backward-looking 24h realized vol
│       ├── garch.py          # fit_gjr_garch (returns params dict), forecast_24h_vol (from params + returns)
│       ├── metrics.py        # QLIKE, MSE-log, correlation, MAE
│       └── inference.py      # VolArtifact, save_vol_artifact, load_vol_artifact, predict_24h_vol
├── scripts/
│   ├── train_vol.py          # NEW: CLI that fits + evaluates + saves artifact
│   └── ...
└── tests/
    └── vol_model/            # NEW
        ├── __init__.py
        ├── test_returns.py
        ├── test_target.py
        ├── test_garch.py
        ├── test_metrics.py
        ├── test_inference.py
        └── test_train_vol.py
```

**Why a separate `vol_model/` package:** distinct artifact type, distinct storage primitives (JSON-only vs LightGBM native), distinct evaluation metrics. Don't try to generalize over Phase 3's `model/` — the abstraction isn't load-bearing for v1.

---

## Task 1: Returns extraction + realized vol target + GarchSpec

**Files:**
- Modify: `pyproject.toml` — add `"arch>=7.0"` dependency
- Create: `src/btc_portfolio_mgr/vol_model/__init__.py`
- Create: `src/btc_portfolio_mgr/vol_model/spec.py`
- Create: `src/btc_portfolio_mgr/vol_model/returns.py`
- Create: `src/btc_portfolio_mgr/vol_model/target.py`
- Create: `tests/vol_model/__init__.py`
- Create: `tests/vol_model/test_returns.py`
- Create: `tests/vol_model/test_target.py`

`GarchSpec` is the model specification frozen dataclass — mean, vol family, p/o/q orders, distribution. `DEFAULT_SPEC` is the locked Phase 4 spec.

`extract_log_returns` reads a prices DataFrame, reindexes to hourly grid, computes 1h log returns, drops null rows, returns a DataFrame with `timestamp` and `log_return` columns.

`compute_realized_24h_vol` produces backward-looking realized vol via rolling-sum-of-squares + sqrt.

- [ ] **Step 1.1: Add arch dependency**

Modify `pyproject.toml`. Add `"arch>=7.0"` to `[project] dependencies`. Then run:
```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
uv pip install -e ".[dev]"
```
Expected: `arch` installs cleanly.

- [ ] **Step 1.2: Write the GarchSpec module**

`src/btc_portfolio_mgr/vol_model/__init__.py`: empty.

`src/btc_portfolio_mgr/vol_model/spec.py`:
```python
"""Locked-in GJR-GARCH model specification + scale factor."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GarchSpec:
    mean: str = "Constant"
    vol: str = "GARCH"
    p: int = 1
    o: int = 1  # asymmetry (GJR)
    q: int = 1
    dist: str = "t"  # Student's t innovations

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GarchSpec":
        return cls(
            mean=str(d["mean"]),
            vol=str(d["vol"]),
            p=int(d["p"]),
            o=int(d["o"]),
            q=int(d["q"]),
            dist=str(d["dist"]),
        )


DEFAULT_SPEC = GarchSpec()
SCALE_FACTOR = 100.0  # log returns × 100 → percentage for arch's optimizer
```

- [ ] **Step 1.3: Write the failing test for returns**

`tests/vol_model/__init__.py`: empty.

`tests/vol_model/test_returns.py`:
```python
from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.vol_model.returns import extract_log_returns
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_extract_log_returns_basic() -> None:
    prices = hourly([100.0, 110.0, 121.0, 133.1])
    df = extract_log_returns(prices)
    assert df.height == 3
    assert df.columns == ["timestamp", "log_return"]
    values = df["log_return"].to_list()
    assert math.isclose(values[0], math.log(110 / 100))
    assert math.isclose(values[1], math.log(121 / 110))
    assert math.isclose(values[2], math.log(133.1 / 121))


def test_extract_log_returns_drops_nulls_from_gap() -> None:
    prices = with_gap([100.0, 101.0, 102.0], gap_hours=2, prices_after=[110.0, 111.0])
    df = extract_log_returns(prices)
    # Reindexed: [100, 101, 102, null, null, 110, 111]
    # 1h returns: [None, r1, r2, None, None, None, r3]; after dropna -> 3 rows.
    assert df.height == 3
    values = df["log_return"].to_list()
    assert math.isclose(values[0], math.log(101 / 100))
    assert math.isclose(values[1], math.log(102 / 101))
    assert math.isclose(values[2], math.log(111 / 110))


def test_extract_log_returns_empty_input_returns_empty() -> None:
    empty = pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "price": pl.Float64(), "volume": pl.Float64()}
    )
    df = extract_log_returns(empty)
    assert df.height == 0
    assert df.columns == ["timestamp", "log_return"]
```

- [ ] **Step 1.4: Run test to verify it fails**

Run: `cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr && .venv/bin/pytest tests/vol_model/test_returns.py -v`
Expected: ImportError.

- [ ] **Step 1.5: Write the returns module**

`src/btc_portfolio_mgr/vol_model/returns.py`:
```python
"""1h log return extraction from hourly prices for GARCH fitting."""
from __future__ import annotations

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.returns import compute_log_return


def extract_log_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Reindex prices to hourly grid, compute 1h log returns, drop nulls.

    Returns a DataFrame with columns (timestamp: Datetime us UTC, log_return: Float64).
    """
    schema = {
        "timestamp": pl.Datetime("us", "UTC"),
        "log_return": pl.Float64(),
    }
    if prices.height == 0:
        return pl.DataFrame(schema=schema)
    reindexed = reindex_to_hourly(prices)
    returns = compute_log_return(reindexed, lookback_hours=1)
    return (
        reindexed.select("timestamp")
        .with_columns(log_return=returns)
        .drop_nulls()
    )
```

- [ ] **Step 1.6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/vol_model/test_returns.py -v`
Expected: 3 passed.

- [ ] **Step 1.7: Write the failing test for realized vol target**

`tests/vol_model/test_target.py`:
```python
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
```

- [ ] **Step 1.8: Run test to verify it fails**

Run: `.venv/bin/pytest tests/vol_model/test_target.py -v`
Expected: ImportError.

- [ ] **Step 1.9: Write the target module**

`src/btc_portfolio_mgr/vol_model/target.py`:
```python
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
```

- [ ] **Step 1.10: Run test to verify it passes**

Run: `.venv/bin/pytest tests/vol_model/test_target.py -v`
Expected: 4 passed.

- [ ] **Step 1.11: Pyright check**

Run:
```bash
.venv/bin/pyright src/btc_portfolio_mgr/vol_model/ tests/vol_model/ 2>&1 | tail -10
```
Expected: 0 errors.

- [ ] **Step 1.12: Commit**

```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
git add pyproject.toml src/btc_portfolio_mgr/vol_model/__init__.py src/btc_portfolio_mgr/vol_model/spec.py src/btc_portfolio_mgr/vol_model/returns.py src/btc_portfolio_mgr/vol_model/target.py tests/vol_model/__init__.py tests/vol_model/test_returns.py tests/vol_model/test_target.py
git commit -m "feat(vol_model): GarchSpec + log return extraction + realized vol target"
```

---

## Task 2: GJR-GARCH fit + 24h forecast (params-only API)

**Files:**
- Create: `src/btc_portfolio_mgr/vol_model/garch.py`
- Create: `tests/vol_model/test_garch.py`

`fit_gjr_garch(log_returns, spec)` fits GJR-GARCH with the locked spec, returns a `dict[str, float]` of params (omega, alpha[1], gamma[1], beta[1], mu, nu — preserving arch's ordering). Internally scales returns × SCALE_FACTOR before fitting.

`forecast_24h_vol(params, log_returns, spec, scale_factor, last_obs_index)` reconstructs the arch model with the provided returns history and fixed params (via `am.fix(...)`), then forecasts integrated 24h vol. No pickle — just params + spec + the returns.

- [ ] **Step 2.1: Write the failing tests**

`tests/vol_model/test_garch.py`:
```python
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
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/vol_model/test_garch.py -v`
Expected: ImportError.

- [ ] **Step 2.3: Write the garch module**

`src/btc_portfolio_mgr/vol_model/garch.py`:
```python
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
        mean=spec.mean,
        vol=spec.vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        dist=spec.dist,
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
    integrated_variance = integrated_variance_scaled / (scale_factor ** 2)
    return math.sqrt(integrated_variance)
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/vol_model/test_garch.py -v`
Expected: 6 passed. (Each fit takes ~1-3s on 3000-point synthetic series; full file should run in <30s.)

- [ ] **Step 2.5: Pyright check**

Run:
```bash
.venv/bin/pyright src/btc_portfolio_mgr/vol_model/garch.py tests/vol_model/test_garch.py 2>&1 | tail -10
```
Expected: 0 errors. The `arch` library has incomplete type stubs; if pyright complains about ARCHModelResult attributes or arch_model returning Any, the `Any` typing on `_build_arch_model` should absorb most issues. If specific attribute errors remain, add `# type: ignore[attr-defined]` minimally — do NOT add defensive runtime checks.

- [ ] **Step 2.6: Commit**

```bash
git add src/btc_portfolio_mgr/vol_model/garch.py tests/vol_model/test_garch.py
git commit -m "feat(vol_model): GJR-GARCH(1,1,1) fit (params-only) + 24h vol forecast via fix()"
```

---

## Task 3: Vol metrics

**Files:**
- Create: `src/btc_portfolio_mgr/vol_model/metrics.py`
- Create: `tests/vol_model/test_metrics.py`

Four metric functions: QLIKE, MSE on log(vol), Pearson correlation, MAE. All accept `(realized_vol, predicted_vol)` numpy arrays and return float. Edge cases (non-positive vols, empty arrays, constant inputs) return NaN.

QLIKE = `mean(log(σ̂²) + σ²_realized / σ̂²)`. Penalizes underestimation more than overestimation.

- [ ] **Step 3.1: Write the failing tests**

`tests/vol_model/test_metrics.py`:
```python
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
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/vol_model/test_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3.3: Write the metrics module**

`src/btc_portfolio_mgr/vol_model/metrics.py`:
```python
"""Volatility-forecast evaluation metrics.

All accept (realized_vol, predicted_vol) 1-D numpy arrays. Edge cases
(non-positive vols, empty, constant) return NaN.
"""
from __future__ import annotations

import numpy as np


def _valid(realized: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return (
        (realized > 0)
        & (predicted > 0)
        & np.isfinite(realized)
        & np.isfinite(predicted)
    )


def compute_qlike(realized: np.ndarray, predicted: np.ndarray) -> float:
    """QLIKE = mean(log(σ̂²) + σ²_realized / σ̂²). Lower is better.

    Penalizes underestimation more than overestimation.
    """
    mask = _valid(realized, predicted)
    if not mask.any():
        return float("nan")
    r = realized[mask]
    p = predicted[mask]
    return float(np.mean(np.log(p ** 2) + (r ** 2) / (p ** 2)))


def compute_mse_log(realized: np.ndarray, predicted: np.ndarray) -> float:
    """MSE of log(vol). Symmetric in log-space."""
    mask = _valid(realized, predicted)
    if not mask.any():
        return float("nan")
    return float(np.mean((np.log(realized[mask]) - np.log(predicted[mask])) ** 2))


def compute_vol_correlation(realized: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation between realized and predicted vol."""
    mask = _valid(realized, predicted)
    if mask.sum() < 2:
        return float("nan")
    r = realized[mask]
    p = predicted[mask]
    if r.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(r, p)[0, 1])


def compute_mae_vol(realized: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error in vol units."""
    mask = _valid(realized, predicted)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(realized[mask] - predicted[mask])))
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/vol_model/test_metrics.py -v`
Expected: 10 passed.

- [ ] **Step 3.5: Pyright check**

Run:
```bash
.venv/bin/pyright src/btc_portfolio_mgr/vol_model/metrics.py tests/vol_model/test_metrics.py 2>&1 | tail -10
```
Expected: 0 errors.

- [ ] **Step 3.6: Commit**

```bash
git add src/btc_portfolio_mgr/vol_model/metrics.py tests/vol_model/test_metrics.py
git commit -m "feat(vol_model): QLIKE + MSE-log + correlation + MAE metrics"
```

---

## Task 4: VolArtifact + JSON-only persistence

**Files:**
- Create: `src/btc_portfolio_mgr/vol_model/inference.py`
- Create: `tests/vol_model/test_inference.py`

`VolArtifact` is a frozen dataclass holding params + spec + provenance — all JSON-serializable. `save_vol_artifact(artifact, path)` writes a single JSON file. `load_vol_artifact(path)` restores it. `predict_24h_vol(artifact, log_returns, last_obs_index)` reconstructs the forecaster from the saved params + the provided returns and returns σ̂_24h.

The Phase 5 sizing module will: (1) load the artifact once at startup, (2) fetch fresh returns each cycle, (3) call `predict_24h_vol(artifact, fresh_returns)`. No pickle anywhere.

- [ ] **Step 4.1: Write the failing tests**

`tests/vol_model/test_inference.py`:
```python
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from btc_portfolio_mgr.vol_model.garch import fit_gjr_garch
from btc_portfolio_mgr.vol_model.inference import (
    VolArtifact,
    load_vol_artifact,
    predict_24h_vol,
    save_vol_artifact,
)
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR


def _toy_artifact_and_returns() -> tuple[VolArtifact, pl.Series]:
    rng = np.random.default_rng(7)
    rets = pl.Series(rng.normal(0, 0.01, 1500), dtype=pl.Float64)
    params = fit_gjr_garch(rets)
    artifact = VolArtifact(
        params=params,
        spec=DEFAULT_SPEC,
        scale_factor=SCALE_FACTOR,
        trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        git_sha="deadbee",
        eval_metrics={"qlike": -8.5, "mse_log": 0.04, "correlation": 0.6, "mae": 0.005},
        n_training_returns=1500,
    )
    return artifact, rets


def test_save_and_load_artifact_roundtrip(tmp_path: Path) -> None:
    artifact, _ = _toy_artifact_and_returns()
    path = tmp_path / "btc_vol.json"
    save_vol_artifact(artifact, path)
    assert path.exists()
    loaded = load_vol_artifact(path)
    # All fields round-trip
    assert loaded.params == artifact.params
    assert loaded.spec == artifact.spec
    assert loaded.scale_factor == artifact.scale_factor
    assert loaded.trained_at == artifact.trained_at
    assert loaded.git_sha == artifact.git_sha
    assert loaded.eval_metrics == artifact.eval_metrics
    assert loaded.n_training_returns == artifact.n_training_returns


def test_save_writes_human_readable_json(tmp_path: Path) -> None:
    artifact, _ = _toy_artifact_and_returns()
    path = tmp_path / "btc_vol.json"
    save_vol_artifact(artifact, path)
    data = json.loads(path.read_text())
    assert "params" in data
    assert "spec" in data
    assert "scale_factor" in data
    assert "trained_at" in data
    assert "git_sha" in data
    assert "eval_metrics" in data
    assert "n_training_returns" in data
    assert data["spec"]["vol"] == "GARCH"
    assert data["spec"]["o"] == 1


def test_predict_24h_vol_returns_positive_float() -> None:
    artifact, rets = _toy_artifact_and_returns()
    vol = predict_24h_vol(artifact, rets)
    assert isinstance(vol, float)
    assert vol > 0
    assert math.isfinite(vol)


def test_predict_24h_vol_at_specific_index() -> None:
    artifact, rets = _toy_artifact_and_returns()
    vol = predict_24h_vol(artifact, rets, last_obs_index=1000)
    assert vol > 0
    assert math.isfinite(vol)


def test_predict_after_roundtrip_matches(tmp_path: Path) -> None:
    artifact, rets = _toy_artifact_and_returns()
    path = tmp_path / "btc_vol.json"
    save_vol_artifact(artifact, path)
    loaded = load_vol_artifact(path)
    vol_original = predict_24h_vol(artifact, rets)
    vol_loaded = predict_24h_vol(loaded, rets)
    assert math.isclose(vol_original, vol_loaded, abs_tol=1e-9)
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/vol_model/test_inference.py -v`
Expected: ImportError.

- [ ] **Step 4.3: Write the inference module**

`src/btc_portfolio_mgr/vol_model/inference.py`:
```python
"""VolArtifact JSON persistence and 24h vol inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from btc_portfolio_mgr.vol_model.garch import forecast_24h_vol
from btc_portfolio_mgr.vol_model.spec import GarchSpec


@dataclass(frozen=True)
class VolArtifact:
    params: dict[str, float]
    spec: GarchSpec
    scale_factor: float
    trained_at: datetime
    git_sha: str
    eval_metrics: dict[str, float]
    n_training_returns: int


def save_vol_artifact(artifact: VolArtifact, path: Path) -> None:
    """Write VolArtifact as JSON. No pickle — all fields are JSON-serializable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "params": artifact.params,
        "spec": artifact.spec.to_dict(),
        "scale_factor": artifact.scale_factor,
        "trained_at": artifact.trained_at.isoformat(),
        "git_sha": artifact.git_sha,
        "eval_metrics": artifact.eval_metrics,
        "n_training_returns": artifact.n_training_returns,
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_vol_artifact(path: Path) -> VolArtifact:
    """Restore VolArtifact from JSON."""
    with path.open() as f:
        data = json.load(f)
    return VolArtifact(
        params={str(k): float(v) for k, v in data["params"].items()},
        spec=GarchSpec.from_dict(data["spec"]),
        scale_factor=float(data["scale_factor"]),
        trained_at=datetime.fromisoformat(data["trained_at"]),
        git_sha=str(data["git_sha"]),
        eval_metrics={str(k): float(v) for k, v in data["eval_metrics"].items()},
        n_training_returns=int(data["n_training_returns"]),
    )


def predict_24h_vol(
    artifact: VolArtifact,
    log_returns: pl.Series,
    last_obs_index: int | None = None,
) -> float:
    """Forecast integrated 24h vol from the artifact's saved params + provided returns."""
    return forecast_24h_vol(
        params=artifact.params,
        log_returns=log_returns,
        spec=artifact.spec,
        scale_factor=artifact.scale_factor,
        last_obs_index=last_obs_index,
    )
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/vol_model/test_inference.py -v`
Expected: 5 passed.

- [ ] **Step 4.5: Pyright check**

Run:
```bash
.venv/bin/pyright src/btc_portfolio_mgr/vol_model/inference.py tests/vol_model/test_inference.py 2>&1 | tail -10
```
Expected: 0 errors.

- [ ] **Step 4.6: Commit**

```bash
git add src/btc_portfolio_mgr/vol_model/inference.py tests/vol_model/test_inference.py
git commit -m "feat(vol_model): VolArtifact (JSON-only) + predict_24h_vol()"
```

---

## Task 5: Train script (CLI orchestrator)

**Files:**
- Create: `scripts/train_vol.py`
- Create: `tests/vol_model/test_train_vol.py`

The CLI reads `data/btc_hourly.parquet`, extracts log returns, fits GJR-GARCH, runs in-sample evaluation, computes the 4 metrics, saves the VolArtifact as JSON. Mirrors the Phase 3 train_model.py orchestration pattern.

For evaluation: walk timestamps after a warm-up of 500 observations (every `EVAL_STRIDE=24` observations to keep runtime bounded). At each t, forecast σ̂_24h via `forecast_24h_vol(...last_obs_index=t)` and compare to the realized 24h vol over [t+1, t+24]. Aggregate to the 4 metrics. In-sample evaluation — Phase 6 backtest does proper walk-forward refit.

- [ ] **Step 5.1: Write the failing test**

`tests/vol_model/test_train_vol.py`:
```python
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from btc_portfolio_mgr.data.storage import write_parquet
from tests.fixtures.synthetic_prices import hourly


def test_train_vol_produces_artifact(tmp_path: Path) -> None:
    from scripts import train_vol as tv

    rng = np.random.default_rng(7)
    n = 2000
    returns = rng.normal(0, 0.01, n - 1)
    prices_list = [100.0]
    for r in returns:
        prices_list.append(prices_list[-1] * float(math.exp(r)))
    prices = hourly(prices_list)
    assert prices.height == n

    prices_path = tmp_path / "btc_hourly.parquet"
    artifact_path = tmp_path / "btc_vol.json"
    write_parquet(prices, prices_path)

    result = tv.run(prices_path=prices_path, artifact_path=artifact_path)

    assert artifact_path.exists()
    # Result dict has the 4 metrics + n_training_returns
    assert "qlike" in result
    assert "mse_log" in result
    assert "correlation" in result
    assert "mae" in result
    assert "n_training_returns" in result
    # JSON sidecar has the right structure
    data = json.loads(artifact_path.read_text())
    assert "params" in data
    assert "spec" in data
    assert data["spec"]["vol"] == "GARCH"
    assert data["spec"]["o"] == 1
    assert data["spec"]["dist"] == "t"
    assert "trained_at" in data
    assert "git_sha" in data
    assert "eval_metrics" in data
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/vol_model/test_train_vol.py -v`
Expected: ModuleNotFoundError for `scripts.train_vol`.

- [ ] **Step 5.3: Write the train script**

`scripts/train_vol.py`:
```python
"""Train the BTC GJR-GARCH 24h vol model: fit, in-sample eval, save JSON artifact."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.model.git_info import current_git_sha
from btc_portfolio_mgr.vol_model.garch import fit_gjr_garch, forecast_24h_vol
from btc_portfolio_mgr.vol_model.inference import VolArtifact, save_vol_artifact
from btc_portfolio_mgr.vol_model.metrics import (
    compute_mae_vol,
    compute_mse_log,
    compute_qlike,
    compute_vol_correlation,
)
from btc_portfolio_mgr.vol_model.returns import extract_log_returns
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR
from btc_portfolio_mgr.vol_model.target import compute_realized_24h_vol

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_ARTIFACT = REPO_ROOT / "models" / "btc_vol.json"

WARMUP_OBS = 500
EVAL_STRIDE = 24


def _evaluate_in_sample(
    params: dict[str, float],
    log_returns,
    realized_24h_arr: np.ndarray,
) -> dict[str, float]:
    n = log_returns.len()
    preds: list[float] = []
    realized: list[float] = []
    for t in range(WARMUP_OBS, n - 24, EVAL_STRIDE):
        forecast_vol = forecast_24h_vol(
            params=params,
            log_returns=log_returns,
            spec=DEFAULT_SPEC,
            scale_factor=SCALE_FACTOR,
            last_obs_index=t,
        )
        realized_vol = float(realized_24h_arr[t + 24])
        if not np.isfinite(realized_vol):
            continue
        preds.append(forecast_vol)
        realized.append(realized_vol)
    pred_arr = np.array(preds)
    real_arr = np.array(realized)
    return {
        "qlike": compute_qlike(real_arr, pred_arr),
        "mse_log": compute_mse_log(real_arr, pred_arr),
        "correlation": compute_vol_correlation(real_arr, pred_arr),
        "mae": compute_mae_vol(real_arr, pred_arr),
    }


def run(
    prices_path: Path = DEFAULT_PRICES,
    artifact_path: Path = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    prices = read_parquet(prices_path)
    rets_df = extract_log_returns(prices)
    n = rets_df.height
    print(f"returns: {n} rows after gap drop")
    log_returns = rets_df["log_return"]
    params = fit_gjr_garch(log_returns, spec=DEFAULT_SPEC)
    print(
        "fitted GJR-GARCH(1,1,1)-t: "
        f"omega={params['omega']:.6f} "
        f"alpha={params['alpha[1]']:.4f} "
        f"gamma={params['gamma[1]']:.4f} "
        f"beta={params['beta[1]']:.4f} "
        f"nu={params.get('nu', float('nan')):.2f}"
    )
    realized_24h = compute_realized_24h_vol(log_returns, window_hours=24)
    metrics = _evaluate_in_sample(params, log_returns, realized_24h.to_numpy())
    print("in-sample 24h vol forecast metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    artifact = VolArtifact(
        params=params,
        spec=DEFAULT_SPEC,
        scale_factor=SCALE_FACTOR,
        trained_at=datetime.now(tz=timezone.utc),
        git_sha=current_git_sha(REPO_ROOT),
        eval_metrics=metrics,
        n_training_returns=n,
    )
    save_vol_artifact(artifact, artifact_path)
    print(f"saved vol artifact to {artifact_path}")
    return {**metrics, "n_training_returns": n}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/vol_model/test_train_vol.py -v`
Expected: 1 passed (takes ~10-30s — fits GARCH on 2000-point series + runs in-sample eval with EVAL_STRIDE=24 means ~60 forecasts).

- [ ] **Step 5.5: Run full suite**

Run: `.venv/bin/pytest -v 2>&1 | tail -20`
Expected: 76 (Phase 3) + 3 (returns) + 4 (target) + 6 (garch) + 10 (vol metrics) + 5 (vol inference) + 1 (train script) = **105 passed**.

- [ ] **Step 5.6: Pyright check**

Run:
```bash
.venv/bin/pyright src/btc_portfolio_mgr/vol_model/ scripts/train_vol.py tests/vol_model/ 2>&1 | tail -10
```
Expected: 0 errors (the pre-existing lightgbm + new arch stub warnings are acceptable).

- [ ] **Step 5.7: Commit**

```bash
git add scripts/train_vol.py tests/vol_model/test_train_vol.py
git commit -m "feat(vol_model): train_vol.py CLI (fit + in-sample eval + JSON artifact save)"
```

---

## Task 6: Manual smoke test against real data

Skip in auto mode.

- [ ] **Step 6.1: Confirm data is present**

```bash
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/data/btc_hourly.parquet
```

- [ ] **Step 6.2: Run the trainer**

```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
.venv/bin/python scripts/train_vol.py
```

Expected: prints n=~70k returns, the GARCH parameters (omega tiny, alpha ~0.05-0.15, gamma ~0.05-0.15, beta ~0.80-0.92, nu ~5-10), and 4 in-sample metrics. Completes in <2 minutes.

- [ ] **Step 6.3: Sanity-check metrics**

Realistic ranges for a v1 24h-vol model on BTC:
- **QLIKE**: -7 to -5 (more negative = better; domain-specific).
- **MSE on log(vol)**: 0.05–0.30.
- **Correlation**: 0.40–0.70 (above 0.5 is decent for hourly BTC).
- **MAE on vol**: ~0.005–0.012 (in raw vol units).

If correlation < 0.2 or QLIKE wildly positive: check returns aren't contaminated with NaN/zero and that scale_factor is correctly applied.

- [ ] **Step 6.4: Inspect artifact**

```bash
cat /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/models/btc_vol.json
```

Expected JSON with: params (omega/alpha[1]/gamma[1]/beta[1]/mu/nu), spec (vol=GARCH, o=1, dist=t), scale_factor=100, trained_at, git_sha, eval_metrics, n_training_returns. **Human-readable, auditable — that's the point of the JSON-only design.**

---

## Done criteria (Phase 4)

- [ ] `pytest -v` passes 105 tests
- [ ] `pyright` reports 0 errors on vol_model + scripts/train_vol.py + tests
- [ ] `scripts/train_vol.py` runs on real Phase 1 data and produces a JSON vol artifact (manual)
- [ ] Phases 1, 2, 3 still green (no regressions)
- [ ] **No pickle anywhere** in vol_model/ or scripts/train_vol.py (audit: `grep -r "pickle" src/btc_portfolio_mgr/vol_model/ scripts/train_vol.py` should return nothing)

## What's deliberately not in Phase 4

- **Walk-forward refit for true OOS evaluation** — v1 uses in-sample params; Phase 6 backtest does proper OOS via rolling refit.
- **Multivariate vol model** (ETH co-vol, BTC dominance) — BTC-only.
- **Realized-volatility models (HAR-RV)** as alternative spec — defer; GJR-GARCH is locked v1.
- **Confidence intervals on σ̂** — point forecast only.
- **Online updating** — for Phase 5 inference, the artifact + fresh returns reconstruct the forecaster each call. Streaming filter updates deferred to Phase 7 if cost becomes a concern.
- **Retraining cadence** — single train invocation. Production retrain logic is Phase 7 ops.

## Design note on no-pickle

`arch` model results aren't natively JSON-serializable, so the conventional persistence path is pickle. That has two real downsides:
1. **Security**: pickle can execute arbitrary code on load. For a portfolio system that may eventually load artifacts from CI artifacts, S3, or shared filesystems, this is a real concern.
2. **Auditability**: pickled binary blobs aren't diff-friendly or human-inspectable. JSON params lets you `cat` the file and see exactly what the model is.

`arch_model(...).fix(params)` is the supported API for reconstructing a forecaster from a params array — no refit, just analytical state filtering forward through the provided returns. We pay the cost of having to pass the returns history at inference time, which Phase 5 already does anyway.
