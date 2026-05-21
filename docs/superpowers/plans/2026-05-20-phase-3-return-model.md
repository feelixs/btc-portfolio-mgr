# Phase 3: LightGBM Return Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a LightGBM regressor that predicts the 24-hour-ahead log return of BTC from the Phase 2 feature matrix, validated with López de Prado purged k-fold CV, and persist the model + metadata for hourly inference in later phases.

**Architecture:** A pure-function model layer organized around a strict data flow: target construction (`target.py`) → dataset assembly (drop nulls, schema-checked) → purged-CV indices (`cv.py`) → fit with Huber loss (`train.py`) → metric aggregation (`metrics.py`) → save model artifact + JSON metadata. Inference is a separate module that loads the artifact and predicts on a feature row.

**Tech Stack:** lightgbm (new dep), numpy, polars. No deep learning, no Optuna, no scikit-learn (overkill for what we need).

---

## Phase status (3 of 7)

| Phase | Status |
|---|---|
| 1. Data layer | ✅ shipped |
| 2. Features | ✅ shipped |
| **3. Return model (LightGBM)** | **this plan** |
| 4. Vol model (GJR-GARCH) | next |
| 5. Sizing (Kelly + vol-target) | after Phase 4 |
| 6. Backtest (purged CV walk-forward) | after Phase 5 |
| 7. Execution | after Phase 6 |

---

## Key design decisions (locked in via discussion)

- **Target**: `log(p_{t+24} / p_t)` — 24-hour-ahead log return.
- **Objective**: Huber loss (LightGBM `objective=huber`, `alpha=0.02`). Robust to BTC fat tails. Alpha=0.02 ≈ ~1x typical 24h residual magnitude.
- **Hyperparameters**: Fixed defaults. `num_leaves=31`, `learning_rate=0.05`, `num_boost_round=500`, `min_data_in_leaf=200`, `lambda_l1=0.1`, `lambda_l2=0.1`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`, `seed=42`.
- **CV**: López de Prado purged k-fold. 5 folds, embargo=24 hours (1× label horizon).
- **Metrics**: IC (Pearson), hit rate (sign accuracy), RMSE, OOF R². All reported per fold and aggregated.
- **No early stopping in v1**: fixed `num_boost_round=500`. Defer early stopping to a polish iteration.
- **Artifact**: LightGBM native text format + sidecar JSON metadata (feature columns, horizon, training date, OOF metrics).

---

## File structure

```
btc-portfolio-mgr/
├── src/btc_portfolio_mgr/
│   ├── data/          # (Phase 1)
│   ├── features/      # (Phase 2)
│   └── model/         # NEW
│       ├── __init__.py
│       ├── target.py        # forward-return target + dataset assembly
│       ├── cv.py            # purged k-fold indices
│       ├── metrics.py       # IC, hit_rate, rmse, r_squared
│       ├── train.py         # LightGBM fit wrapper
│       └── inference.py     # ModelArtifact + predict
├── scripts/
│   ├── train_model.py       # NEW: CLI runs CV + trains final + saves artifact
│   └── ...
├── models/                  # NEW directory; gitignored
│   └── .gitkeep
└── tests/
    └── model/               # NEW
        ├── __init__.py
        ├── test_target.py
        ├── test_cv.py
        ├── test_metrics.py
        ├── test_train.py
        └── test_inference.py
```

**Schema contracts:**

- `DATASET_SCHEMA` = `{timestamp: Datetime("us","UTC"), <15 feature cols>: Float64, target: Float64}` — defined in `target.py`.
- `ModelArtifact` (frozen dataclass) = `(booster, feature_columns, target_horizon_hours, trained_at)`.

---

## Task 1: Target construction + dataset assembly

**Files:**
- Create: `src/btc_portfolio_mgr/model/__init__.py`
- Create: `src/btc_portfolio_mgr/model/target.py`
- Create: `tests/model/__init__.py`
- Create: `tests/model/test_target.py`

The target is `log(p_{t+H} / p_t)` for horizon H. We use polars' `shift(-H)` on a reindexed price series so the alignment is exact. `build_dataset` joins the feature matrix with the target column and drops any row with a null in features or target — those rows can't train the model.

`DATASET_SCHEMA` is the contract Phase 4+ depends on.

- [ ] **Step 1.1: Add lightgbm dependency**

Modify `pyproject.toml`. Add `"lightgbm>=4.5"` to the `[project] dependencies` list. After modification, run `uv pip install -e ".[dev]"` from the repo root to install. Expected: lightgbm installs cleanly.

- [ ] **Step 1.2: Write the failing test**

`tests/model/__init__.py`: empty.

`tests/model/test_target.py`:
```python
from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.target import (
    DATASET_SCHEMA,
    build_dataset,
    compute_forward_log_return,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_forward_return_constant_price_is_zero() -> None:
    prices = reindex_to_hourly(hourly([100.0] * 10))
    target = compute_forward_log_return(prices, horizon_hours=2)
    values = target.to_list()
    # First 8 rows have a future price -> 0.0; last 2 don't -> null.
    assert all(v == 0.0 for v in values[:8])
    assert values[8] is None
    assert values[9] is None


def test_forward_return_known_value() -> None:
    prices = reindex_to_hourly(hourly([100.0, 110.0, 121.0, 133.1]))
    target = compute_forward_log_return(prices, horizon_hours=2)
    # target[0] = log(121/100), target[1] = log(133.1/110), target[2..3] = null
    values = target.to_list()
    assert math.isclose(values[0], math.log(121 / 100))
    assert math.isclose(values[1], math.log(133.1 / 110))
    assert values[2] is None
    assert values[3] is None


def test_forward_return_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0, 102.0], gap_hours=2, prices_after=[110.0, 111.0])
    )
    # Reindexed: [100, 101, 102, null, null, 110, 111]
    # horizon=1: target[i] = log(price[i+1]/price[i])
    # i=0: log(101/100) ✓
    # i=1: log(102/101) ✓
    # i=2: log(null/102) = null
    # i=3: null source price -> null
    # i=4: log(110/null) = null
    # i=5: log(111/110) ✓
    # i=6: no next price -> null
    target = compute_forward_log_return(prices, horizon_hours=1)
    values = target.to_list()
    assert math.isclose(values[0], math.log(101 / 100))
    assert math.isclose(values[1], math.log(102 / 101))
    assert values[2] is None
    assert values[3] is None
    assert values[4] is None
    assert math.isclose(values[5], math.log(111 / 110))
    assert values[6] is None


def test_build_dataset_drops_null_rows_and_returns_schema() -> None:
    # Need at least 2160 + 24 = 2184 hours so some rows have both full features and a valid target.
    prices = hourly([100.0 + i * 0.01 for i in range(2300)])
    features = compose_features(prices)
    dataset = build_dataset(features=features, prices=prices, horizon_hours=24)
    # Schema must match contract.
    assert dataset.schema == DATASET_SCHEMA
    assert dataset.columns == ["timestamp"] + FEATURE_COLUMNS + ["target"]
    # No nulls remain after drop.
    for col in dataset.columns:
        assert dataset[col].null_count() == 0, f"{col} has nulls"
    # Sanity: row count is between 0 and input height, and timestamps are sorted ascending.
    assert 0 < dataset.height < 2300
    assert dataset["timestamp"].is_sorted()


def test_build_dataset_empty_input_returns_empty() -> None:
    empty_prices = pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "price": pl.Float64(), "volume": pl.Float64()}
    )
    empty_features = compose_features(empty_prices)
    dataset = build_dataset(features=empty_features, prices=empty_prices, horizon_hours=24)
    assert dataset.height == 0
    assert dataset.schema == DATASET_SCHEMA
```

- [ ] **Step 1.3: Run test to verify it fails**

Run: `cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr && .venv/bin/pytest tests/model/test_target.py -v`
Expected: ImportError for `btc_portfolio_mgr.model.target`.

- [ ] **Step 1.4: Write the target module**

`src/btc_portfolio_mgr/model/__init__.py`:
```python
```

`src/btc_portfolio_mgr/model/target.py`:
```python
"""Forward-return target construction and dataset assembly."""
from __future__ import annotations

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA

DATASET_SCHEMA: dict[str, pl.DataType] = {
    **FEATURE_SCHEMA,
    "target": pl.Float64(),
}


def compute_forward_log_return(
    prices_reindexed: pl.DataFrame, horizon_hours: int
) -> pl.Series:
    """log(price_{t+horizon} / price_t).

    `prices_reindexed` MUST be on a complete hourly grid (use
    `features.gaps.reindex_to_hourly`). Returns null where the future
    price is missing.
    """
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive, got {horizon_hours}")
    expr = pl.col("price").shift(-horizon_hours).log() - pl.col("price").log()
    return prices_reindexed.select(expr.alias("target")).get_column("target")


def build_dataset(
    features: pl.DataFrame, prices: pl.DataFrame, horizon_hours: int
) -> pl.DataFrame:
    """Join features + forward-return target by timestamp, drop null rows.

    Returns a DataFrame matching DATASET_SCHEMA (timestamp + 15 features + target).
    Rows where any feature or the target is null are dropped — those rows
    can't train the model.
    """
    prices_reindexed = reindex_to_hourly(prices)
    if prices_reindexed.height == 0:
        return pl.DataFrame(schema=DATASET_SCHEMA)
    target = compute_forward_log_return(prices_reindexed, horizon_hours)
    target_df = prices_reindexed.select("timestamp").with_columns(target=target)
    joined = features.join(target_df, on="timestamp", how="inner")
    cleaned = joined.drop_nulls()
    # Enforce column order matches DATASET_SCHEMA
    return cleaned.select(list(DATASET_SCHEMA.keys()))
```

- [ ] **Step 1.5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/model/test_target.py -v`
Expected: 5 passed.

- [ ] **Step 1.6: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/model/target.py tests/model/test_target.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 1.7: Commit**

```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
git add pyproject.toml src/btc_portfolio_mgr/model/__init__.py src/btc_portfolio_mgr/model/target.py tests/model/__init__.py tests/model/test_target.py
git commit -m "feat(model): forward-return target and dataset assembly"
```

---

## Task 2: Purged k-fold CV indices

**Files:**
- Create: `src/btc_portfolio_mgr/model/cv.py`
- Create: `tests/model/test_cv.py`

López de Prado purged k-fold: for each test fold, remove from the training set any sample whose label window overlaps the test region (purge), plus any sample immediately after the test region within an embargo buffer (embargo).

Given an ordered dataset of `n` rows (already sorted by timestamp), `label_horizon_hours`=24 (the target spans 24h forward), and `embargo_hours`=24:

- Test fold k covers rows `[k * fold_size, (k+1) * fold_size)`
- Purge: training samples whose target window overlaps the test set. A sample at position i has target window `[i, i+24]`. To prevent overlap with `[test_start, test_end)`, exclude training samples with positions in `[test_start - 24, test_start)`. (Samples ≥ test_start are either in the test set or after the embargo region.)
- Embargo: training samples in `[test_end, test_end + 24)` are excluded because their feature lookback windows may extend backward into the test region.

The function yields `(train_idx, test_idx)` numpy arrays.

- [ ] **Step 2.1: Write the failing test**

`tests/model/test_cv.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from btc_portfolio_mgr.model.cv import purged_kfold_indices


def test_basic_fold_partition() -> None:
    # n=100, 5 folds, no purge/embargo (horizon=embargo=0).
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=0, embargo_hours=0)
    )
    assert len(folds) == 5
    # Each test fold has 20 rows
    for train_idx, test_idx in folds:
        assert test_idx.shape[0] == 20
        # train + test together cover all indices (no purge/embargo)
        assert sorted(set(train_idx.tolist()) | set(test_idx.tolist())) == list(range(100))
        # train and test are disjoint
        assert len(set(train_idx) & set(test_idx)) == 0


def test_purge_removes_samples_with_overlapping_labels() -> None:
    # n=100, 5 folds, horizon=10, embargo=0.
    # Fold 1: test_idx=[20..40). Purged from train: [10..20).
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=10, embargo_hours=0)
    )
    train_idx, test_idx = folds[1]
    assert test_idx.tolist() == list(range(20, 40))
    train_set = set(train_idx.tolist())
    # Purged region [10..20) must NOT be in train
    for i in range(10, 20):
        assert i not in train_set, f"index {i} should be purged"
    # Index 9 should still be in train
    assert 9 in train_set
    # Indices after test [40..100) should be in train (no embargo)
    for i in range(40, 100):
        assert i in train_set


def test_embargo_removes_samples_after_test() -> None:
    # n=100, 5 folds, horizon=0, embargo=10.
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=0, embargo_hours=10)
    )
    train_idx, test_idx = folds[1]
    train_set = set(train_idx.tolist())
    # Embargo [40..50) must NOT be in train
    for i in range(40, 50):
        assert i not in train_set, f"index {i} should be embargoed"
    # Index 50 should be in train
    assert 50 in train_set


def test_purge_and_embargo_combined() -> None:
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=5, embargo_hours=5)
    )
    train_idx, test_idx = folds[2]  # middle fold: test=[40..60)
    train_set = set(train_idx.tolist())
    # Purged: [35..40)
    for i in range(35, 40):
        assert i not in train_set
    # Embargoed: [60..65)
    for i in range(60, 65):
        assert i not in train_set
    # Before purge zone: in train
    assert 34 in train_set
    # After embargo zone: in train
    assert 65 in train_set


def test_first_fold_handles_negative_purge_start() -> None:
    # Fold 0: test_idx=[0..20). Purge would start at -10, clamp to 0.
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=10, embargo_hours=0)
    )
    train_idx, test_idx = folds[0]
    assert test_idx.tolist() == list(range(0, 20))
    # No negative indices in train
    assert (train_idx >= 0).all()
    assert (train_idx < 100).all()
    # Embargo end is just past test_end=20, no embargo since embargo=0
    train_set = set(train_idx.tolist())
    for i in range(20, 100):
        assert i in train_set


def test_last_fold_includes_remainder() -> None:
    # n=103, 5 folds: first 4 have size 20, last has 23.
    folds = list(
        purged_kfold_indices(n=103, n_folds=5, label_horizon_hours=0, embargo_hours=0)
    )
    sizes = [test_idx.shape[0] for _, test_idx in folds]
    assert sizes == [20, 20, 20, 20, 23]


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        list(purged_kfold_indices(n=100, n_folds=0, label_horizon_hours=0, embargo_hours=0))
    with pytest.raises(ValueError):
        list(purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=-1, embargo_hours=0))
    with pytest.raises(ValueError):
        list(purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=0, embargo_hours=-1))
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/model/test_cv.py -v`
Expected: ImportError.

- [ ] **Step 2.3: Write the CV module**

`src/btc_portfolio_mgr/model/cv.py`:
```python
"""López de Prado purged k-fold cross-validation indices.

For each test fold:
- Purge training samples whose target window overlaps the test region.
- Embargo training samples immediately after the test region whose
  feature lookback windows may reach back into the test region.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def purged_kfold_indices(
    n: int,
    n_folds: int,
    label_horizon_hours: int,
    embargo_hours: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) tuples for purged k-fold CV.

    Assumes the underlying dataset is already sorted by timestamp.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be positive, got {n_folds}")
    if label_horizon_hours < 0:
        raise ValueError(
            f"label_horizon_hours must be non-negative, got {label_horizon_hours}"
        )
    if embargo_hours < 0:
        raise ValueError(f"embargo_hours must be non-negative, got {embargo_hours}")
    fold_size = n // n_folds
    for k in range(n_folds):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_folds - 1 else n
        purge_start = max(0, test_start - label_horizon_hours)
        embargo_end = min(n, test_end + embargo_hours)
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate(
            [
                np.arange(0, purge_start),
                np.arange(embargo_end, n),
            ]
        )
        yield train_idx, test_idx
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/model/test_cv.py -v`
Expected: 7 passed.

- [ ] **Step 2.5: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/model/cv.py tests/model/test_cv.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 2.6: Commit**

```bash
git add src/btc_portfolio_mgr/model/cv.py tests/model/test_cv.py
git commit -m "feat(model): purged k-fold CV indices (Lopez de Prado)"
```

---

## Task 3: Metrics module

**Files:**
- Create: `src/btc_portfolio_mgr/model/metrics.py`
- Create: `tests/model/test_metrics.py`

Four metric functions: IC (Pearson correlation), hit rate (sign accuracy), RMSE, R². All accept `(y_true, y_pred)` as 1-D numpy arrays and return a float. Each handles edge cases (constant predictions, empty arrays) by returning NaN rather than raising.

- [ ] **Step 3.1: Write the failing tests**

`tests/model/test_metrics.py`:
```python
from __future__ import annotations

import math

import numpy as np
import pytest

from btc_portfolio_mgr.model.metrics import (
    compute_hit_rate,
    compute_ic,
    compute_r_squared,
    compute_rmse,
)


def test_ic_perfect_correlation_is_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # scaled by 2
    assert math.isclose(compute_ic(y_true, y_pred), 1.0, abs_tol=1e-9)


def test_ic_anti_correlation_is_negative_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert math.isclose(compute_ic(y_true, y_pred), -1.0, abs_tol=1e-9)


def test_ic_constant_pred_is_nan() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    assert math.isnan(compute_ic(y_true, y_pred))


def test_hit_rate_all_same_sign() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.04])
    y_pred = np.array([0.005, -0.01, 0.02, -0.03])  # all same sign
    assert compute_hit_rate(y_true, y_pred) == 1.0


def test_hit_rate_half_correct() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.04])
    y_pred = np.array([0.005, 0.01, 0.02, 0.03])  # 2 of 4 same sign
    assert compute_hit_rate(y_true, y_pred) == 0.5


def test_hit_rate_zeros_excluded() -> None:
    y_true = np.array([0.0, 0.01, -0.01])
    y_pred = np.array([0.5, 0.005, -0.005])  # row 0 excluded (y_true is 0)
    assert compute_hit_rate(y_true, y_pred) == 1.0


def test_rmse_perfect_prediction_is_zero() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert compute_rmse(y_true, y_pred) == 0.0


def test_rmse_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    # squared errors: 1, 0, 1 -> mean 2/3 -> sqrt = sqrt(2/3)
    assert math.isclose(compute_rmse(y_true, y_pred), math.sqrt(2 / 3))


def test_r_squared_perfect_is_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_r_squared(y_true, y_pred) == 1.0


def test_r_squared_mean_baseline_is_zero() -> None:
    # If pred = mean(y_true), R^2 = 0
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert math.isclose(compute_r_squared(y_true, y_pred), 0.0, abs_tol=1e-9)


def test_r_squared_can_be_negative() -> None:
    # Predictions worse than mean baseline -> negative R^2
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # anti-correlated
    assert compute_r_squared(y_true, y_pred) < 0


def test_r_squared_constant_true_is_nan() -> None:
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 1.0, 1.1])
    assert math.isnan(compute_r_squared(y_true, y_pred))
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/model/test_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3.3: Write the metrics module**

`src/btc_portfolio_mgr/model/metrics.py`:
```python
"""Evaluation metrics for return-prediction models.

All functions accept (y_true, y_pred) as 1-D numpy arrays and return float.
Edge cases (constant inputs, empty arrays) return NaN rather than raising.
"""
from __future__ import annotations

import numpy as np


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between predictions and realized values.

    Returns NaN if either array is constant.
    """
    if y_true.size < 2:
        return float("nan")
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions whose sign matches the realized value's sign.

    Rows where y_true == 0 are excluded from the denominator.
    Returns NaN if no non-zero rows.
    """
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination on out-of-fold predictions.

    1 - SS_res / SS_tot. Negative when worse than mean baseline.
    Returns NaN if y_true is constant.
    """
    if y_true.size < 2:
        return float("nan")
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/model/test_metrics.py -v`
Expected: 12 passed.

- [ ] **Step 3.5: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/model/metrics.py tests/model/test_metrics.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 3.6: Commit**

```bash
git add src/btc_portfolio_mgr/model/metrics.py tests/model/test_metrics.py
git commit -m "feat(model): IC, hit rate, RMSE, R^2 metrics"
```

---

## Task 4: LightGBM training wrapper

**Files:**
- Create: `src/btc_portfolio_mgr/model/train.py`
- Create: `tests/model/test_train.py`

A thin wrapper around `lgb.train` that exposes default hyperparameters and supports overrides. Two main functions:

- `train_lightgbm(X_train, y_train, params=None, num_boost_round=500) -> lgb.Booster`: fit a regressor and return the booster.
- `cross_validate(dataset, n_folds=5, label_horizon_hours=24, embargo_hours=24, params=None) -> CVResult`: run purged k-fold, return per-fold and aggregated metrics + out-of-fold predictions for inspection.

The unit tests use a tiny synthetic dataset with a known linear relationship so we can verify the model actually learns (IC near 1, hit rate near 1).

- [ ] **Step 4.1: Write the failing tests**

`tests/model/test_train.py`:
```python
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.target import DATASET_SCHEMA
from btc_portfolio_mgr.model.train import (
    DEFAULT_PARAMS,
    CVResult,
    cross_validate,
    train_lightgbm,
)


def _synthetic_dataset(n: int, seed: int = 42) -> pl.DataFrame:
    """Build a dataset where target = 0.3 * ret_1h + 0.2 * vol_24h + noise."""
    from datetime import datetime, timezone, timedelta
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=n - 1)
    cols = {"timestamp": pl.datetime_range(
        start=start, end=end, interval="1h", time_zone="UTC", eager=True,
    )}
    # Random features
    feature_values: dict[str, np.ndarray] = {}
    for col in FEATURE_COLUMNS:
        feature_values[col] = rng.normal(0, 1, n).astype(np.float64)
    # Target driven by two features
    target = (
        0.3 * feature_values["ret_1h"]
        + 0.2 * feature_values["vol_24h"]
        + rng.normal(0, 0.1, n)
    )
    cols.update(feature_values)
    cols["target"] = target
    return pl.DataFrame(cols, schema=DATASET_SCHEMA)


def test_train_lightgbm_learns_known_signal() -> None:
    df = _synthetic_dataset(n=1000)
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["target"].to_numpy()
    booster = train_lightgbm(X, y)
    preds = booster.predict(X)
    # On training data with a strong linear signal, IC should be very high.
    ic = float(np.corrcoef(y, preds)[0, 1])
    assert ic > 0.8, f"expected IC > 0.8, got {ic}"


def test_train_lightgbm_uses_default_params() -> None:
    # Verify the default hyperparameter dict exposes the right keys.
    assert DEFAULT_PARAMS["objective"] == "huber"
    assert DEFAULT_PARAMS["alpha"] == 0.02
    assert DEFAULT_PARAMS["num_leaves"] == 31
    assert DEFAULT_PARAMS["learning_rate"] == 0.05
    assert DEFAULT_PARAMS["min_data_in_leaf"] == 200
    assert DEFAULT_PARAMS["lambda_l1"] == 0.1
    assert DEFAULT_PARAMS["lambda_l2"] == 0.1
    assert DEFAULT_PARAMS["seed"] == 42


def test_train_lightgbm_param_override() -> None:
    df = _synthetic_dataset(n=500)
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["target"].to_numpy()
    # Override learning_rate; expect a booster still produced
    booster = train_lightgbm(X, y, params={"learning_rate": 0.01}, num_boost_round=100)
    preds = booster.predict(X)
    assert preds.shape == (500,)


def test_cross_validate_returns_per_fold_metrics() -> None:
    df = _synthetic_dataset(n=2000)
    result = cross_validate(
        df,
        n_folds=5,
        label_horizon_hours=24,
        embargo_hours=24,
    )
    assert isinstance(result, CVResult)
    # Per-fold arrays of size 5
    assert len(result.fold_ic) == 5
    assert len(result.fold_hit_rate) == 5
    assert len(result.fold_rmse) == 5
    assert len(result.fold_r_squared) == 5
    # Aggregated mean fields
    assert result.mean_ic == pytest.approx(np.nanmean(result.fold_ic))
    # OOF predictions cover (some fraction of) the dataset
    assert result.oof_predictions.shape == (df.height,)
    # OOF mask: True at indices that were in any test fold
    assert result.oof_mask.shape == (df.height,)
    assert result.oof_mask.sum() == df.height  # every row is in exactly one test fold
    # Sanity: IC on the known linear signal should be positive on average.
    assert result.mean_ic > 0.2, f"expected mean IC > 0.2, got {result.mean_ic}"
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/model/test_train.py -v`
Expected: ImportError.

- [ ] **Step 4.3: Write the training module**

`src/btc_portfolio_mgr/model/train.py`:
```python
"""LightGBM training wrapper and purged-CV evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.cv import purged_kfold_indices
from btc_portfolio_mgr.model.metrics import (
    compute_hit_rate,
    compute_ic,
    compute_r_squared,
    compute_rmse,
)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "huber",
    "alpha": 0.02,  # threshold in residual-space; ~1x typical 24h log-return magnitude
    "metric": "huber",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
) -> lgb.Booster:
    """Fit a LightGBM regressor with Huber loss. Returns the booster."""
    merged = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(X_train, label=y_train)
    return lgb.train(
        merged,
        train_set,
        num_boost_round=num_boost_round,
    )


@dataclass(frozen=True)
class CVResult:
    fold_ic: list[float]
    fold_hit_rate: list[float]
    fold_rmse: list[float]
    fold_r_squared: list[float]
    mean_ic: float
    mean_hit_rate: float
    mean_rmse: float
    mean_r_squared: float
    oof_predictions: np.ndarray
    oof_mask: np.ndarray


def cross_validate(
    dataset: pl.DataFrame,
    n_folds: int = 5,
    label_horizon_hours: int = 24,
    embargo_hours: int = 24,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
) -> CVResult:
    """Run purged k-fold CV. Returns per-fold metrics + OOF predictions."""
    X = dataset.select(FEATURE_COLUMNS).to_numpy()
    y = dataset["target"].to_numpy()
    n = X.shape[0]
    oof_preds = np.full(n, np.nan)
    oof_mask = np.zeros(n, dtype=bool)
    fold_ic: list[float] = []
    fold_hit_rate: list[float] = []
    fold_rmse: list[float] = []
    fold_r_squared: list[float] = []
    for train_idx, test_idx in purged_kfold_indices(
        n=n,
        n_folds=n_folds,
        label_horizon_hours=label_horizon_hours,
        embargo_hours=embargo_hours,
    ):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        booster = train_lightgbm(X_train, y_train, params=params, num_boost_round=num_boost_round)
        preds = booster.predict(X_test)
        oof_preds[test_idx] = preds
        oof_mask[test_idx] = True
        fold_ic.append(compute_ic(y_test, preds))
        fold_hit_rate.append(compute_hit_rate(y_test, preds))
        fold_rmse.append(compute_rmse(y_test, preds))
        fold_r_squared.append(compute_r_squared(y_test, preds))
    return CVResult(
        fold_ic=fold_ic,
        fold_hit_rate=fold_hit_rate,
        fold_rmse=fold_rmse,
        fold_r_squared=fold_r_squared,
        mean_ic=float(np.nanmean(fold_ic)),
        mean_hit_rate=float(np.nanmean(fold_hit_rate)),
        mean_rmse=float(np.nanmean(fold_rmse)),
        mean_r_squared=float(np.nanmean(fold_r_squared)),
        oof_predictions=oof_preds,
        oof_mask=oof_mask,
    )
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/model/test_train.py -v`
Expected: 4 passed. (Test 4 trains 5 LightGBM models on 2000 rows × 15 features; should finish in <30 seconds.)

- [ ] **Step 4.5: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/model/train.py tests/model/test_train.py 2>&1 | tail -10`
Expected: 0 errors. (LightGBM type stubs may emit unavoidable `Any` returns; cast or `# type: ignore` minimally if needed.)

- [ ] **Step 4.6: Commit**

```bash
git add src/btc_portfolio_mgr/model/train.py tests/model/test_train.py
git commit -m "feat(model): LightGBM training wrapper + purged-CV evaluator"
```

---

## Task 5: Inference module + ModelArtifact

**Files:**
- Create: `src/btc_portfolio_mgr/model/inference.py`
- Create: `tests/model/test_inference.py`

A `ModelArtifact` frozen dataclass holds the booster + metadata. `save_artifact` and `load_artifact` persist/restore from disk (LightGBM native text file + JSON sidecar). `predict` takes a feature DataFrame and returns a polars Series of predictions.

- [ ] **Step 5.1: Write the failing tests**

`tests/model/test_inference.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.inference import (
    ModelArtifact,
    load_artifact,
    predict,
    save_artifact,
)
from btc_portfolio_mgr.model.target import DATASET_SCHEMA
from btc_portfolio_mgr.model.train import train_lightgbm


def _toy_artifact() -> ModelArtifact:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (500, len(FEATURE_COLUMNS)))
    y = 0.3 * X[:, 0] + rng.normal(0, 0.1, 500)
    booster = train_lightgbm(X, y, num_boost_round=50)
    return ModelArtifact(
        booster=booster,
        feature_columns=list(FEATURE_COLUMNS),
        target_horizon_hours=24,
        trained_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
    )


def test_save_and_load_artifact_roundtrip(tmp_path: Path) -> None:
    artifact = _toy_artifact()
    model_path = tmp_path / "btc_24h.txt"
    meta_path = tmp_path / "btc_24h.metadata.json"
    save_artifact(artifact, model_path, meta_path)
    assert model_path.exists()
    assert meta_path.exists()
    loaded = load_artifact(model_path, meta_path)
    assert loaded.feature_columns == artifact.feature_columns
    assert loaded.target_horizon_hours == artifact.target_horizon_hours
    assert loaded.trained_at == artifact.trained_at
    # Predictions match
    rng = np.random.default_rng(0)
    X_test = rng.normal(0, 1, (10, len(FEATURE_COLUMNS)))
    original_preds = artifact.booster.predict(X_test)
    loaded_preds = loaded.booster.predict(X_test)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)


def test_predict_returns_series_of_right_length() -> None:
    from datetime import timedelta
    artifact = _toy_artifact()
    # Build a feature DataFrame
    rng = np.random.default_rng(0)
    rows: dict = {col: rng.normal(0, 1, 50).tolist() for col in FEATURE_COLUMNS}
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=49)
    rows["timestamp"] = pl.datetime_range(
        start=start, end=end, interval="1h", time_zone="UTC", eager=True,
    ).to_list()
    features = pl.DataFrame(rows)
    preds = predict(artifact, features)
    assert preds.dtype == pl.Float64
    assert preds.len() == 50


def test_predict_raises_when_feature_columns_missing() -> None:
    artifact = _toy_artifact()
    bad = pl.DataFrame({"timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)], "ret_1h": [0.0]})
    with pytest.raises(KeyError):
        predict(artifact, bad)
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/model/test_inference.py -v`
Expected: ImportError.

- [ ] **Step 5.3: Write the inference module**

`src/btc_portfolio_mgr/model/inference.py`:
```python
"""Model artifact persistence and inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import polars as pl


@dataclass(frozen=True)
class ModelArtifact:
    booster: lgb.Booster
    feature_columns: list[str]
    target_horizon_hours: int
    trained_at: datetime


def save_artifact(
    artifact: ModelArtifact, model_path: Path, metadata_path: Path
) -> None:
    """Save booster + JSON metadata sidecar."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.booster.save_model(str(model_path))
    metadata = {
        "feature_columns": artifact.feature_columns,
        "target_horizon_hours": artifact.target_horizon_hours,
        "trained_at": artifact.trained_at.isoformat(),
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def load_artifact(model_path: Path, metadata_path: Path) -> ModelArtifact:
    """Load booster + JSON metadata into a ModelArtifact."""
    booster = lgb.Booster(model_file=str(model_path))
    with metadata_path.open() as f:
        metadata = json.load(f)
    return ModelArtifact(
        booster=booster,
        feature_columns=metadata["feature_columns"],
        target_horizon_hours=int(metadata["target_horizon_hours"]),
        trained_at=datetime.fromisoformat(metadata["trained_at"]),
    )


def predict(artifact: ModelArtifact, features: pl.DataFrame) -> pl.Series:
    """Predict on a feature DataFrame. Columns must match training feature set."""
    missing = [c for c in artifact.feature_columns if c not in features.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    X = features.select(artifact.feature_columns).to_numpy()
    preds = artifact.booster.predict(X)
    return pl.Series("prediction", preds, dtype=pl.Float64)
```

- [ ] **Step 5.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/model/test_inference.py -v`
Expected: 3 passed.

- [ ] **Step 5.5: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/model/inference.py tests/model/test_inference.py 2>&1 | tail -10`
Expected: 0 errors. (LightGBM type stubs may require minimal casts.)

- [ ] **Step 5.6: Commit**

```bash
git add src/btc_portfolio_mgr/model/inference.py tests/model/test_inference.py
git commit -m "feat(model): ModelArtifact persistence + predict()"
```

---

## Task 6: Train script (CLI orchestrator)

**Files:**
- Create: `scripts/train_model.py`
- Create: `tests/model/test_train_script.py`
- Modify: `.gitignore` — add `models/*.txt` and `models/*.metadata.json` (model artifacts are large + non-deterministic; not committed)
- Create: `models/.gitkeep`

The CLI reads `data/btc_hourly.parquet` and `data/btc_features.parquet`, builds the dataset, runs purged CV, trains a final model on all data, saves the artifact, and prints metrics.

- [ ] **Step 6.1: Update .gitignore**

Append to `.gitignore`:
```
models/*.txt
models/*.metadata.json
```

Then run:
```bash
mkdir -p /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/models
touch /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/models/.gitkeep
```

- [ ] **Step 6.2: Write the failing test**

`tests/model/test_train_script.py`:
```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from btc_portfolio_mgr.data.storage import SCHEMA, write_parquet
from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.features.schema import FEATURE_SCHEMA
from tests.fixtures.synthetic_prices import hourly


def test_train_script_produces_artifact(tmp_path: Path) -> None:
    from scripts import train_model as tm

    # Need enough rows to satisfy 2160h zscore_90d lookback + 24h target + room to train.
    n = 2500
    prices = hourly([100.0 + i * 0.01 + (i % 7) * 0.05 for i in range(n)])
    prices_path = tmp_path / "btc_hourly.parquet"
    features_path = tmp_path / "btc_features.parquet"
    model_path = tmp_path / "btc_24h.txt"
    metadata_path = tmp_path / "btc_24h.metadata.json"

    write_parquet(prices, prices_path)
    features = compose_features(prices)
    features.write_parquet(features_path)

    result = tm.run(
        prices_path=prices_path,
        features_path=features_path,
        model_path=model_path,
        metadata_path=metadata_path,
        n_folds=3,
        num_boost_round=50,
    )

    # The model + metadata files exist
    assert model_path.exists()
    assert metadata_path.exists()
    # Metadata contains the right keys
    metadata = json.loads(metadata_path.read_text())
    assert "feature_columns" in metadata
    assert metadata["target_horizon_hours"] == 24
    assert "trained_at" in metadata
    # The CV result is exposed
    assert "mean_ic" in result
    assert "mean_hit_rate" in result
    assert "mean_rmse" in result
    assert "mean_r_squared" in result
```

- [ ] **Step 6.3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/model/test_train_script.py -v`
Expected: ModuleNotFoundError for `scripts.train_model`.

- [ ] **Step 6.4: Write the train script**

`scripts/train_model.py`:
```python
"""Train the BTC 24h-ahead return model: purged CV, then final fit on all data."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.inference import ModelArtifact, save_artifact
from btc_portfolio_mgr.model.target import build_dataset
from btc_portfolio_mgr.model.train import cross_validate, train_lightgbm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "btc_features.parquet"
DEFAULT_MODEL = REPO_ROOT / "models" / "btc_24h.txt"
DEFAULT_METADATA = REPO_ROOT / "models" / "btc_24h.metadata.json"

TARGET_HORIZON_HOURS = 24
EMBARGO_HOURS = 24


def run(
    prices_path: Path = DEFAULT_PRICES,
    features_path: Path = DEFAULT_FEATURES,
    model_path: Path = DEFAULT_MODEL,
    metadata_path: Path = DEFAULT_METADATA,
    n_folds: int = 5,
    num_boost_round: int = 500,
) -> dict[str, Any]:
    """Build dataset → purged CV → final fit on all data → save artifact."""
    prices = read_parquet(prices_path)
    features = pl.read_parquet(features_path)
    dataset = build_dataset(features, prices, horizon_hours=TARGET_HORIZON_HOURS)
    print(f"dataset: {dataset.height} rows after null-drop")

    cv = cross_validate(
        dataset,
        n_folds=n_folds,
        label_horizon_hours=TARGET_HORIZON_HOURS,
        embargo_hours=EMBARGO_HOURS,
        num_boost_round=num_boost_round,
    )
    print(f"CV (n_folds={n_folds}):")
    print(f"  IC:        per-fold {[f'{x:.4f}' for x in cv.fold_ic]} mean={cv.mean_ic:.4f}")
    print(f"  hit_rate:  per-fold {[f'{x:.4f}' for x in cv.fold_hit_rate]} mean={cv.mean_hit_rate:.4f}")
    print(f"  RMSE:      per-fold {[f'{x:.4f}' for x in cv.fold_rmse]} mean={cv.mean_rmse:.4f}")
    print(f"  R^2:       per-fold {[f'{x:.4f}' for x in cv.fold_r_squared]} mean={cv.mean_r_squared:.4f}")

    # Final model: train on ALL data.
    X = dataset.select(FEATURE_COLUMNS).to_numpy()
    y = dataset["target"].to_numpy()
    booster = train_lightgbm(X, y, num_boost_round=num_boost_round)

    artifact = ModelArtifact(
        booster=booster,
        feature_columns=list(FEATURE_COLUMNS),
        target_horizon_hours=TARGET_HORIZON_HOURS,
        trained_at=datetime.now(tz=timezone.utc),
    )
    save_artifact(artifact, model_path, metadata_path)
    print(f"saved model to {model_path}")
    print(f"saved metadata to {metadata_path}")

    return {
        "mean_ic": cv.mean_ic,
        "mean_hit_rate": cv.mean_hit_rate,
        "mean_rmse": cv.mean_rmse,
        "mean_r_squared": cv.mean_r_squared,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/model/test_train_script.py -v`
Expected: 1 passed (test takes ~10-30s to train 3 folds + final).

- [ ] **Step 6.6: Run full suite**

Run: `.venv/bin/pytest -v`
Expected: prior 42 tests + 5 (target) + 7 (cv) + 12 (metrics) + 4 (train) + 3 (inference) + 1 (train script) = **74 passed**.

- [ ] **Step 6.7: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/model/ scripts/train_model.py tests/model/ 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 6.8: Commit**

```bash
git add .gitignore models/.gitkeep scripts/train_model.py tests/model/test_train_script.py
git commit -m "feat(model): train_model.py CLI (purged CV + final fit + artifact save)"
```

---

## Task 7: Manual smoke test against real data

This task requires Phase 1 backfill + Phase 2 build_features to have run. Skip in auto mode.

- [ ] **Step 7.1: Confirm data is present**

```bash
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/data/btc_hourly.parquet
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/data/btc_features.parquet
```
Expected: both exist.

- [ ] **Step 7.2: Run the trainer**

```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
.venv/bin/python scripts/train_model.py
```
Expected output: prints dataset row count (~65–70k), per-fold metrics for 5 folds, mean metrics, saves model + metadata to `models/`.

- [ ] **Step 7.3: Sanity-check the metrics**

Realistic targets for a v1 24h-ahead BTC model on this feature set:
- Mean IC: 0.03–0.10 (positive is good; >0.05 is solid)
- Mean hit rate: 0.51–0.55 (>0.52 is real edge)
- Mean RMSE: ~0.015–0.020 (in log-return units; for 24h moves of ~1.5%)
- Mean R²: -0.05 to +0.05 (negative is normal for noisy financial regression; positive means you beat the mean baseline)

If IC is ≤ 0.01 or strongly negative, something is wrong: likely a leakage bug, a feature schema mismatch, or the model isn't training (check num_boost_round). If IC is suspiciously high (>0.3), even more suspicious — investigate label leakage first.

- [ ] **Step 7.4: Inspect the artifact**

```bash
cat /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/models/btc_24h.metadata.json
```
Expected: JSON with feature_columns (15 of them), target_horizon_hours=24, trained_at ISO timestamp.

---

## Done criteria (Phase 3)

- [ ] `pytest -v` passes 74 tests
- [ ] `pyright src/btc_portfolio_mgr/model/ scripts/train_model.py` reports 0 errors
- [ ] `scripts/train_model.py` runs on real Phase 1+2 data and produces a model artifact (manual)
- [ ] No regressions in Phases 1 or 2

## What's deliberately not in Phase 3

- **Hyperparameter tuning** — defer to v2 if v1 metrics warrant. Optuna integration is one phase of work on its own.
- **Early stopping** — fixed `num_boost_round=500`. Add early stopping (and OOF-best-iter aggregation) when we have a reason to.
- **Feature importance analysis** — useful for understanding but not load-bearing for production. Trivial to add later (`booster.feature_importance()`).
- **Walk-forward retraining** — the canonical "production" pattern (rolling retrain every N hours/days). That's Phase 6 (backtest) and Phase 7 (live ops), not Phase 3.
- **Vol model** — Phase 4. The return model produces μ̂; vol gives σ̂; sizing (Phase 5) combines them.
- **Multi-target prediction** (predict 1h AND 24h) — defer to a Phase 3.5 if ever needed. For v1, one horizon is enough.
- **Sample weighting** — equal weights for v1. Later: time-decay weights (recent more important), or vol-scaled weights.
