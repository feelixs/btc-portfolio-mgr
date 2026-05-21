# Phase 5: Fractional Kelly Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the sizing layer that combines μ̂ from the Phase 3 return model and σ̂ from the Phase 4 vol model into a target portfolio weight in [-1, +1], using fractional Kelly with k=0.20 and a 2% rebalance threshold.

**Architecture:** A pure-function math core (Kelly weight + clipping + rebalance threshold) plus a thin integration pipeline that loads both artifacts, invokes their `predict` APIs, and returns a target weight. No new dependencies. Stateless — Phase 7 execution layer is the one that tracks the actual position.

**Tech Stack:** numpy, polars, the Phase 3+4 artifact types. No new deps.

---

## Phase status (5 of 7)

| Phase | Status |
|---|---|
| 1. Data layer | ✅ shipped |
| 2. Features | ✅ shipped |
| 3. Return model (LightGBM) | ✅ shipped |
| 4. Vol model (GJR-GARCH) | ✅ shipped |
| **5. Sizing (Kelly + threshold)** | **this plan** |
| 6. Backtest (walk-forward) | next |
| 7. Execution | after Phase 6 |

---

## Key design decisions (locked in via discussion)

- **Position bounds**: [-1, +1] — long-short, no leverage.
- **Sizing formula**: `w = clip(k · μ̂ / σ̂², lower, upper)` — fractional Kelly with safety multiplier.
- **Kelly fraction k**: 0.20 (industry-standard conservative default; recovers ~80% of optimal growth at much lower drawdown variance).
- **Rebalance threshold**: 0.02 absolute — if |new_weight - current_weight| ≤ 0.02, keep current. Reduces churn from noise.
- **API shape**: scalar inputs (μ̂, σ̂ as floats). Phase 6 backtest will vectorize externally.
- **Stateless**: the sizer doesn't track current position. The caller supplies it as `current_weight`.

The Kelly formula `w = k·μ̂/σ̂²` is exact when μ̂ and σ̂ are on the same horizon. Both are 24h-ahead by construction — μ̂ from Phase 3 is `log(p_{t+24}/p_t)`, σ̂ from Phase 4 is `sqrt(sum(per-hour variances over h=1..24))`. So `σ̂²` is the integrated 24h variance, matching the units of `μ̂²`.

---

## File structure

```
btc-portfolio-mgr/
├── src/btc_portfolio_mgr/
│   ├── data/             # (Phase 1)
│   ├── features/         # (Phase 2)
│   ├── model/            # (Phase 3 — return model)
│   ├── vol_model/        # (Phase 4 — vol model)
│   └── sizing/           # NEW
│       ├── __init__.py
│       ├── params.py         # SizingParams dataclass + DEFAULT_PARAMS
│       ├── sizer.py          # compute_kelly_weight, apply_rebalance_threshold, target_weight
│       └── pipeline.py       # compute_target_weight (loads artifacts + composes)
├── scripts/
│   ├── run_sizing.py         # NEW: CLI that prints target weight from latest data
│   └── ...
└── tests/
    └── sizing/               # NEW
        ├── __init__.py
        ├── test_sizer.py
        ├── test_pipeline.py
        └── test_run_sizing.py
```

**Why a single math module (`sizer.py`):** the three math functions (Kelly weight, threshold, composer) are ~30 lines combined. Splitting them further would add ceremony without clarity. They share inputs/outputs (all scalar float).

**Why `pipeline.py` separately:** it's the integration boundary — imports both `model.inference.predict` (Phase 3) and `vol_model.inference.predict_24h_vol` (Phase 4). Keeping it separate lets the math module stay free of artifact dependencies for fast tests.

---

## Task 1: SizingParams + sizer math

**Files:**
- Create: `src/btc_portfolio_mgr/sizing/__init__.py`
- Create: `src/btc_portfolio_mgr/sizing/params.py`
- Create: `src/btc_portfolio_mgr/sizing/sizer.py`
- Create: `tests/sizing/__init__.py`
- Create: `tests/sizing/test_sizer.py`

`SizingParams` frozen dataclass holds: `kelly_fraction=0.20`, `lower_bound=-1.0`, `upper_bound=+1.0`, `rebalance_threshold=0.02`. `DEFAULT_PARAMS` is the canonical instance.

`sizer.py` exports three pure functions:
- `compute_kelly_weight(mu, sigma, kelly_fraction, lower_bound, upper_bound) -> float`
- `apply_rebalance_threshold(new_weight, current_weight, threshold) -> float`
- `target_weight(mu, sigma, current_weight, params) -> float` — composes the above

- [ ] **Step 1.1: Write the params module**

`src/btc_portfolio_mgr/sizing/__init__.py`: empty.

`src/btc_portfolio_mgr/sizing/params.py`:
```python
"""Sizing parameters for the fractional Kelly + threshold sizer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizingParams:
    kelly_fraction: float = 0.20
    lower_bound: float = -1.0
    upper_bound: float = 1.0
    rebalance_threshold: float = 0.02


DEFAULT_PARAMS = SizingParams()
```

- [ ] **Step 1.2: Write the failing tests**

`tests/sizing/__init__.py`: empty.

`tests/sizing/test_sizer.py`:
```python
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
    # mu = 0.005 (0.5% expected 24h return), sigma = 0.02 (2% vol), k = 0.20
    # raw = 0.20 * 0.005 / 0.02^2 = 0.20 * 0.005 / 0.0004 = 2.5 -> clipped to 1.0
    w = compute_kelly_weight(
        mu=0.005, sigma=0.02, kelly_fraction=0.20, lower_bound=-1.0, upper_bound=1.0
    )
    assert w == 1.0


def test_kelly_weight_negative_mu_gives_negative_weight() -> None:
    # mu = -0.001, sigma = 0.02, k = 0.20
    # raw = 0.20 * -0.001 / 0.0004 = -0.5 (within bounds)
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
    # Large positive mu + small sigma -> huge raw, clipped to upper bound.
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
    # Halving k should halve the raw weight (when not clipped).
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
    # Change of 0.01 < threshold 0.02 -> stay at current.
    w = apply_rebalance_threshold(new_weight=0.51, current_weight=0.50, threshold=0.02)
    assert w == 0.50


def test_threshold_large_change_uses_new() -> None:
    # Change of 0.05 > threshold 0.02 -> use new.
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
    # With DEFAULT_PARAMS, mu=0.001, sigma=0.02 -> raw = 0.20*0.001/0.0004 = 0.5
    # current = 0.0 -> diff 0.5 > 0.02 -> use new weight 0.5.
    w = target_weight(mu=0.001, sigma=0.02, current_weight=0.0, params=DEFAULT_PARAMS)
    assert math.isclose(w, 0.5)


def test_target_weight_respects_threshold() -> None:
    # Small change should keep current.
    w = target_weight(mu=0.001, sigma=0.02, current_weight=0.49, params=DEFAULT_PARAMS)
    # raw new = 0.5, current = 0.49, diff = 0.01 < 0.02 -> stay.
    assert w == 0.49


def test_target_weight_respects_bounds() -> None:
    # Huge mu -> raw clipped to upper bound 1.0
    w = target_weight(mu=1.0, sigma=0.01, current_weight=0.0, params=DEFAULT_PARAMS)
    assert w == 1.0


def test_target_weight_custom_params() -> None:
    # With k=1.0 (full Kelly) instead of 0.20, the same inputs give a 5x raw weight.
    full_kelly = SizingParams(kelly_fraction=1.0)
    # mu=0.001, sigma=0.02 -> raw = 1.0 * 0.001 / 0.0004 = 2.5 -> clipped at 1.0
    w = target_weight(mu=0.001, sigma=0.02, current_weight=0.0, params=full_kelly)
    assert w == 1.0
```

- [ ] **Step 1.3: Run test to verify it fails**

Run: `cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr && .venv/bin/pytest tests/sizing/test_sizer.py -v`
Expected: ImportError.

- [ ] **Step 1.4: Write the sizer module**

`src/btc_portfolio_mgr/sizing/sizer.py`:
```python
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
    """Return new_weight if |new - current| > threshold; otherwise current_weight.

    A threshold of 0 always rebalances. Negative thresholds are not allowed.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    if abs(new_weight - current_weight) > threshold:
        return new_weight
    return current_weight


def target_weight(
    mu: float, sigma: float, current_weight: float, params: SizingParams
) -> float:
    """Compose Kelly weight + bound clipping + rebalance threshold.

    The full sizing pipeline as a single scalar function. μ and σ are the
    24h-ahead return and vol forecasts (same horizon).
    """
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
```

- [ ] **Step 1.5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/sizing/test_sizer.py -v`
Expected: 18 passed.

- [ ] **Step 1.6: Pyright check**

Run:
```bash
.venv/bin/pyright src/btc_portfolio_mgr/sizing/ tests/sizing/ 2>&1 | tail -10
```
Expected: 0 errors.

- [ ] **Step 1.7: Commit**

```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
git add src/btc_portfolio_mgr/sizing/__init__.py src/btc_portfolio_mgr/sizing/params.py src/btc_portfolio_mgr/sizing/sizer.py tests/sizing/__init__.py tests/sizing/test_sizer.py
git commit -m "feat(sizing): SizingParams + fractional Kelly + rebalance threshold"
```

---

## Task 2: Pipeline (load artifacts + integrate)

**Files:**
- Create: `src/btc_portfolio_mgr/sizing/pipeline.py`
- Create: `tests/sizing/test_pipeline.py`

The pipeline function loads both artifacts (return + vol), produces predictions, and computes the target weight. Two entry points:

- `compute_target_weight(return_artifact, vol_artifact, features_row, log_returns, current_weight, params)` — takes already-loaded artifacts (Phase 6 backtest will use this in a hot loop, so we don't re-load each call).
- `compute_target_weight_from_paths(return_model_path, return_metadata_path, vol_artifact_path, features_row, log_returns, current_weight, params)` — convenience wrapper that loads from disk first. Phase 7 ops will use this.

The `features_row` is a single-row polars DataFrame (matches Phase 3's `predict` API). The pipeline extracts the scalar prediction from the returned Series. `log_returns` is the Series of 1h log returns up to and including the forecast point.

- [ ] **Step 2.1: Write the failing tests**

`tests/sizing/test_pipeline.py`:
```python
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.inference import ModelArtifact
from btc_portfolio_mgr.model.train import train_lightgbm
from btc_portfolio_mgr.sizing.params import DEFAULT_PARAMS, SizingParams
from btc_portfolio_mgr.sizing.pipeline import (
    compute_target_weight,
    compute_target_weight_from_paths,
)
from btc_portfolio_mgr.vol_model.garch import fit_gjr_garch
from btc_portfolio_mgr.vol_model.inference import (
    VolArtifact,
    save_vol_artifact,
)
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR


def _toy_return_artifact() -> ModelArtifact:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (500, len(FEATURE_COLUMNS)))
    y = 0.3 * X[:, 0] + rng.normal(0, 0.1, 500)
    booster = train_lightgbm(X, y, num_boost_round=50)
    return ModelArtifact(
        booster=booster,
        feature_columns=list(FEATURE_COLUMNS),
        target_horizon_hours=24,
        trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        git_sha="aaaaaaa",
        cv_metrics={"mean_ic": 0.05, "mean_hit_rate": 0.52, "mean_rmse": 0.015, "mean_r_squared": 0.0},
    )


def _toy_vol_artifact_and_returns() -> tuple[VolArtifact, pl.Series]:
    rng = np.random.default_rng(7)
    rets = pl.Series(rng.normal(0, 0.01, 1500), dtype=pl.Float64)
    params = fit_gjr_garch(rets)
    return (
        VolArtifact(
            params=params,
            spec=DEFAULT_SPEC,
            scale_factor=SCALE_FACTOR,
            trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
            git_sha="bbbbbbb",
            eval_metrics={"qlike": -8.0, "mse_log": 0.05, "correlation": 0.5, "mae": 0.006},
            n_training_returns=1500,
        ),
        rets,
    )


def _toy_features_row() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    row = {col: [rng.normal(0, 1)] for col in FEATURE_COLUMNS}
    return pl.DataFrame(row)


def test_compute_target_weight_returns_float_in_bounds() -> None:
    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    features = _toy_features_row()
    w = compute_target_weight(
        return_artifact=return_artifact,
        vol_artifact=vol_artifact,
        features_row=features,
        log_returns=rets,
        current_weight=0.0,
        params=DEFAULT_PARAMS,
    )
    assert isinstance(w, float)
    assert -1.0 <= w <= 1.0


def test_compute_target_weight_respects_threshold() -> None:
    """A small change between target and current should keep current."""
    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    features = _toy_features_row()
    w_initial = compute_target_weight(
        return_artifact=return_artifact,
        vol_artifact=vol_artifact,
        features_row=features,
        log_returns=rets,
        current_weight=0.0,
        params=DEFAULT_PARAMS,
    )
    # Call again with current = w_initial; small re-computation noise should not change it.
    w_repeat = compute_target_weight(
        return_artifact=return_artifact,
        vol_artifact=vol_artifact,
        features_row=features,
        log_returns=rets,
        current_weight=w_initial,
        params=DEFAULT_PARAMS,
    )
    # The threshold should pin the second call to w_initial exactly.
    assert w_repeat == w_initial


def test_compute_target_weight_features_row_must_be_single_row() -> None:
    import pytest

    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    rng = np.random.default_rng(0)
    multi_row = pl.DataFrame({col: rng.normal(0, 1, 3).tolist() for col in FEATURE_COLUMNS})
    with pytest.raises(ValueError):
        compute_target_weight(
            return_artifact=return_artifact,
            vol_artifact=vol_artifact,
            features_row=multi_row,
            log_returns=rets,
            current_weight=0.0,
            params=DEFAULT_PARAMS,
        )


def test_compute_target_weight_from_paths(tmp_path: Path) -> None:
    from btc_portfolio_mgr.model.inference import save_artifact

    return_artifact = _toy_return_artifact()
    vol_artifact, rets = _toy_vol_artifact_and_returns()
    features = _toy_features_row()

    return_model_path = tmp_path / "ret.txt"
    return_meta_path = tmp_path / "ret.metadata.json"
    vol_path = tmp_path / "vol.json"
    save_artifact(return_artifact, return_model_path, return_meta_path)
    save_vol_artifact(vol_artifact, vol_path)

    w = compute_target_weight_from_paths(
        return_model_path=return_model_path,
        return_metadata_path=return_meta_path,
        vol_artifact_path=vol_path,
        features_row=features,
        log_returns=rets,
        current_weight=0.0,
        params=DEFAULT_PARAMS,
    )
    assert isinstance(w, float)
    assert -1.0 <= w <= 1.0
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/sizing/test_pipeline.py -v`
Expected: ImportError for `btc_portfolio_mgr.sizing.pipeline`.

- [ ] **Step 2.3: Write the pipeline module**

`src/btc_portfolio_mgr/sizing/pipeline.py`:
```python
"""End-to-end target-weight computation from artifacts + current state."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from btc_portfolio_mgr.model.inference import (
    ModelArtifact,
    load_artifact,
    predict,
)
from btc_portfolio_mgr.sizing.params import SizingParams
from btc_portfolio_mgr.sizing.sizer import target_weight
from btc_portfolio_mgr.vol_model.inference import (
    VolArtifact,
    load_vol_artifact,
    predict_24h_vol,
)


def compute_target_weight(
    return_artifact: ModelArtifact,
    vol_artifact: VolArtifact,
    features_row: pl.DataFrame,
    log_returns: pl.Series,
    current_weight: float,
    params: SizingParams,
) -> float:
    """Compute target weight from already-loaded artifacts.

    `features_row` MUST be a 1-row polars DataFrame matching the return
    artifact's `feature_columns`. `log_returns` is the 1h log-return history
    up to (and including) the forecast point — used by the vol artifact.
    `current_weight` is the actual current position (caller-tracked).
    """
    if features_row.height != 1:
        raise ValueError(
            f"features_row must have exactly 1 row, got {features_row.height}"
        )
    mu_series = predict(return_artifact, features_row)
    mu = float(mu_series.to_numpy()[0])
    sigma = predict_24h_vol(vol_artifact, log_returns)
    return target_weight(
        mu=mu,
        sigma=sigma,
        current_weight=current_weight,
        params=params,
    )


def compute_target_weight_from_paths(
    return_model_path: Path,
    return_metadata_path: Path,
    vol_artifact_path: Path,
    features_row: pl.DataFrame,
    log_returns: pl.Series,
    current_weight: float,
    params: SizingParams,
) -> float:
    """Convenience wrapper: load both artifacts from disk, then compute."""
    return_artifact = load_artifact(return_model_path, return_metadata_path)
    vol_artifact = load_vol_artifact(vol_artifact_path)
    return compute_target_weight(
        return_artifact=return_artifact,
        vol_artifact=vol_artifact,
        features_row=features_row,
        log_returns=log_returns,
        current_weight=current_weight,
        params=params,
    )
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/sizing/test_pipeline.py -v`
Expected: 4 passed. (Each test fits a LightGBM + GARCH on small synthetic data; runs in ~5-15s total.)

- [ ] **Step 2.5: Pyright check**

Run:
```bash
.venv/bin/pyright src/btc_portfolio_mgr/sizing/pipeline.py tests/sizing/test_pipeline.py 2>&1 | tail -10
```
Expected: 0 errors.

- [ ] **Step 2.6: Commit**

```bash
git add src/btc_portfolio_mgr/sizing/pipeline.py tests/sizing/test_pipeline.py
git commit -m "feat(sizing): pipeline composes return + vol artifacts into target weight"
```

---

## Task 3: CLI script (run_sizing.py)

**Files:**
- Create: `scripts/run_sizing.py`
- Create: `tests/sizing/test_run_sizing.py`

The CLI reads the latest features row from `data/btc_features.parquet`, the latest log returns from `data/btc_hourly.parquet`, loads both artifacts from `models/`, and prints the target weight assuming `current_weight=0` (a clean-slate baseline). Phase 7 ops will integrate with actual position tracking; this script is for inspection/debugging.

- [ ] **Step 3.1: Write the failing test**

`tests/sizing/test_run_sizing.py`:
```python
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from btc_portfolio_mgr.data.storage import write_parquet
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.model.inference import ModelArtifact, save_artifact
from btc_portfolio_mgr.model.train import train_lightgbm
from btc_portfolio_mgr.vol_model.garch import fit_gjr_garch
from btc_portfolio_mgr.vol_model.inference import VolArtifact, save_vol_artifact
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR
from tests.fixtures.synthetic_prices import hourly


def test_run_sizing_prints_target_weight(tmp_path: Path) -> None:
    from scripts import run_sizing as rs

    # Synthetic prices long enough for features (2160h lookback) + GARCH.
    n = 2500
    rng = np.random.default_rng(13)
    returns = rng.normal(0, 0.01, n - 1)
    prices_list = [100.0]
    for r in returns:
        prices_list.append(prices_list[-1] * float(math.exp(r)))
    prices = hourly(prices_list)

    prices_path = tmp_path / "btc_hourly.parquet"
    features_path = tmp_path / "btc_features.parquet"
    return_model_path = tmp_path / "btc_24h.txt"
    return_meta_path = tmp_path / "btc_24h.metadata.json"
    vol_path = tmp_path / "btc_vol.json"

    write_parquet(prices, prices_path)
    features = compose_features(prices)
    features.write_parquet(features_path)

    # Train a toy return model on the features that are non-null
    valid_features = features.drop_nulls()
    rng2 = np.random.default_rng(0)
    X = valid_features.select(FEATURE_COLUMNS).to_numpy()
    y = rng2.normal(0, 0.01, X.shape[0])
    booster = train_lightgbm(X, y, num_boost_round=50)
    return_artifact = ModelArtifact(
        booster=booster,
        feature_columns=list(FEATURE_COLUMNS),
        target_horizon_hours=24,
        trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        git_sha="aaaaaaa",
        cv_metrics={"mean_ic": 0.0, "mean_hit_rate": 0.5, "mean_rmse": 0.01, "mean_r_squared": 0.0},
    )
    save_artifact(return_artifact, return_model_path, return_meta_path)

    # Train a toy vol model
    from btc_portfolio_mgr.vol_model.returns import extract_log_returns

    log_returns = extract_log_returns(prices)["log_return"]
    vol_params = fit_gjr_garch(log_returns)
    vol_artifact = VolArtifact(
        params=vol_params,
        spec=DEFAULT_SPEC,
        scale_factor=SCALE_FACTOR,
        trained_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        git_sha="bbbbbbb",
        eval_metrics={"qlike": -8.0, "mse_log": 0.05, "correlation": 0.5, "mae": 0.006},
        n_training_returns=int(log_returns.len()),
    )
    save_vol_artifact(vol_artifact, vol_path)

    result = rs.run(
        prices_path=prices_path,
        features_path=features_path,
        return_model_path=return_model_path,
        return_metadata_path=return_meta_path,
        vol_artifact_path=vol_path,
        current_weight=0.0,
    )

    assert "target_weight" in result
    assert isinstance(result["target_weight"], float)
    assert -1.0 <= result["target_weight"] <= 1.0
    assert "mu" in result
    assert "sigma" in result
    assert result["sigma"] > 0
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/sizing/test_run_sizing.py -v`
Expected: ModuleNotFoundError for `scripts.run_sizing`.

- [ ] **Step 3.3: Write the CLI script**

`scripts/run_sizing.py`:
```python
"""Compute the current target portfolio weight from latest data + saved artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.model.inference import load_artifact, predict
from btc_portfolio_mgr.sizing.params import DEFAULT_PARAMS
from btc_portfolio_mgr.sizing.sizer import target_weight
from btc_portfolio_mgr.vol_model.inference import load_vol_artifact, predict_24h_vol
from btc_portfolio_mgr.vol_model.returns import extract_log_returns

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "btc_features.parquet"
DEFAULT_RETURN_MODEL = REPO_ROOT / "models" / "btc_24h.txt"
DEFAULT_RETURN_METADATA = REPO_ROOT / "models" / "btc_24h.metadata.json"
DEFAULT_VOL_ARTIFACT = REPO_ROOT / "models" / "btc_vol.json"


def run(
    prices_path: Path = DEFAULT_PRICES,
    features_path: Path = DEFAULT_FEATURES,
    return_model_path: Path = DEFAULT_RETURN_MODEL,
    return_metadata_path: Path = DEFAULT_RETURN_METADATA,
    vol_artifact_path: Path = DEFAULT_VOL_ARTIFACT,
    current_weight: float = 0.0,
) -> dict[str, Any]:
    """Load latest data + artifacts, compute and print target weight."""
    prices = read_parquet(prices_path)
    features = pl.read_parquet(features_path)
    # Use the most recent feature row whose values are all non-null.
    valid_features = features.drop_nulls()
    if valid_features.height == 0:
        raise RuntimeError("no non-null feature rows available")
    latest_features_row = valid_features.tail(1)
    log_returns = extract_log_returns(prices)["log_return"]

    return_artifact = load_artifact(return_model_path, return_metadata_path)
    vol_artifact = load_vol_artifact(vol_artifact_path)

    mu_series = predict(return_artifact, latest_features_row)
    mu = float(mu_series.to_numpy()[0])
    sigma = predict_24h_vol(vol_artifact, log_returns)
    w = target_weight(
        mu=mu, sigma=sigma, current_weight=current_weight, params=DEFAULT_PARAMS
    )

    print(f"mu (24h log return forecast):    {mu:+.6f}")
    print(f"sigma (24h vol forecast):        {sigma:.6f}")
    print(f"current_weight:                  {current_weight:+.4f}")
    print(f"target_weight (after threshold): {w:+.4f}")
    return {
        "mu": mu,
        "sigma": sigma,
        "current_weight": current_weight,
        "target_weight": w,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/sizing/test_run_sizing.py -v`
Expected: 1 passed (takes ~10-30s due to fitting a real LightGBM + GARCH).

- [ ] **Step 3.5: Run full suite**

Run: `.venv/bin/pytest -v 2>&1 | tail -20`
Expected: 105 (Phase 4) + 18 (sizer) + 4 (pipeline) + 1 (run_sizing) = **128 passed**.

- [ ] **Step 3.6: Pyright check**

Run:
```bash
.venv/bin/pyright src/btc_portfolio_mgr/sizing/ scripts/run_sizing.py tests/sizing/ 2>&1 | tail -10
```
Expected: 0 errors.

- [ ] **Step 3.7: Commit**

```bash
git add scripts/run_sizing.py tests/sizing/test_run_sizing.py
git commit -m "feat(sizing): run_sizing.py CLI (loads artifacts, prints target weight)"
```

---

## Task 4: Manual smoke test against real data

Skip in auto mode.

- [ ] **Step 4.1: Confirm artifacts exist**

```bash
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/models/btc_24h.txt
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/models/btc_24h.metadata.json
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/models/btc_vol.json
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/data/btc_hourly.parquet
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/data/btc_features.parquet
```
Expected: all 5 exist.

- [ ] **Step 4.2: Run the sizer**

```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
.venv/bin/python scripts/run_sizing.py
```

Expected output: 4 lines — mu, sigma, current_weight, target_weight. Completes in <3 seconds.

- [ ] **Step 4.3: Sanity-check the output**

Sanity ranges for v1 with real data:
- `mu`: typically -0.005 to +0.005 (the LightGBM forecast is a 24h log return; this is small in magnitude — that's expected, not a bug)
- `sigma`: typically 0.01 to 0.05 (24h vol, 1-5%)
- `target_weight`: typically -1.0 to +1.0 with most values in [-0.5, +0.5]. If you see ±1.0 saturating, the model is very confident — investigate the Phase 3/4 metrics.
- If `target_weight == 0.0` always: check that mu isn't exactly 0 (model didn't learn) and that the Kelly raw isn't being clipped by an inverted bound.

---

## Done criteria (Phase 5)

- [ ] `pytest -v` passes 128 tests
- [ ] `pyright` reports 0 errors on sizing/ + scripts/run_sizing.py + tests/sizing/
- [ ] `scripts/run_sizing.py` runs on real Phase 1-4 artifacts and prints a sensible target weight (manual)
- [ ] Phases 1-4 still green (no regressions)

## What's deliberately not in Phase 5

- **Transaction cost modeling** — Phase 6 backtest will need this; not part of the sizing computation itself.
- **Position-state persistence** — the sizer is stateless; Phase 7 ops will persist `current_weight` across runs.
- **Multi-asset sizing** — Phase 5 produces a scalar weight for BTC. Multi-asset is a redesign (mean-variance portfolio optimization).
- **Hysteresis / band sizing** — the threshold is symmetric. Asymmetric bands (e.g., easier to deleverage than re-leverage) could be added later.
- **Drawdown-based vol scaling** — adaptive vol-targeting that responds to recent drawdowns. Future polish.
- **Full Kelly diagnostics** — we don't expose the raw Kelly fraction or the unclipped weight. Phase 6 backtest may want to log these for inspection; the `target_weight` function can be augmented later.
