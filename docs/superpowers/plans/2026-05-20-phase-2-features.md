# Phase 2: Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the feature engineering pipeline that turns hourly BTC price bars into a 15-column feature matrix used by the Phase 3 LightGBM return model.

**Architecture:** Each feature family is a separate pure module (returns, volatility, regime, gaps) that operates on a polars DataFrame of hourly prices. A pipeline module composes them into the final feature matrix. A build script materializes features to parquet for fast model training; the same compute function is reusable at inference time. Gaps in the historical data produce **null** feature rows (no fabricated values).

**Tech Stack:** polars (rolling windows, built-in skew), numpy (custom rolling kurtosis), pytest. No new dependencies beyond Phase 1.

---

## Phase status (2 of 7)

| Phase | Status |
|---|---|
| 1. Data layer | ✅ shipped (commits a22757b…c23e23d) |
| **2. Features** | **this plan** |
| 3. Return model (LightGBM) | next |
| 4. Vol model (GJR-GARCH) | after Phase 3 |
| 5. Sizing (Kelly + vol-target) | after Phases 3, 4 |
| 6. Backtest (purged k-fold + walk-forward) | after Phase 5 |
| 7. Execution | after Phase 6 |

---

## Feature inventory (15 features)

| # | Column | Family | Lookback | Definition |
|---|---|---|---|---|
| 1 | `ret_1h` | returns | 1h | `log(p_t / p_{t-1})` |
| 2 | `ret_4h` | returns | 4h | `log(p_t / p_{t-4})` |
| 3 | `ret_24h` | returns | 24h | `log(p_t / p_{t-24})` |
| 4 | `ret_7d` | returns | 168h | `log(p_t / p_{t-168})` |
| 5 | `ret_30d` | returns | 720h | `log(p_t / p_{t-720})` |
| 6 | `vol_24h` | volatility | 24h | `sqrt(sum(ret_1h_i^2, i in window))` |
| 7 | `vol_7d` | volatility | 168h | same, larger window |
| 8 | `vol_30d` | volatility | 720h | same |
| 9 | `zscore_30d` | regime | 720h | `(p_t - rolling_mean_720(p)) / rolling_std_720(p)` |
| 10 | `zscore_90d` | regime | 2160h | same, 2160h |
| 11 | `skew_24h` | volatility | 24h | sample skew of 1h returns in window |
| 12 | `skew_7d` | volatility | 168h | same |
| 13 | `kurt_24h` | volatility | 24h | sample excess kurtosis of 1h returns |
| 14 | `kurt_7d` | volatility | 168h | same |
| 15 | `ker_24h` | regime | 24h | Kaufman Efficiency Ratio over 24h |

Rule applied everywhere: if any input in the rolling window is null (because of a data gap), the output is null at that timestamp. No forward-fill, no interpolation.

---

## File structure

```
btc-portfolio-mgr/
├── src/btc_portfolio_mgr/
│   ├── data/                      # (Phase 1, unchanged)
│   └── features/                  # NEW
│       ├── __init__.py
│       ├── schema.py              # FEATURE_COLUMNS list, FEATURE_SCHEMA dict
│       ├── gaps.py                # reindex_to_hourly, find_gaps
│       ├── returns.py             # compute_log_return
│       ├── volatility.py          # realized vol/skew/kurt
│       ├── regime.py              # MA z-score, Kaufman efficiency
│       └── pipeline.py            # compose_features
├── scripts/
│   └── build_features.py          # NEW: read prices parquet, materialize features parquet
└── tests/
    ├── features/                  # NEW
    │   ├── __init__.py
    │   ├── test_gaps.py
    │   ├── test_returns.py
    │   ├── test_volatility.py
    │   ├── test_regime.py
    │   ├── test_pipeline.py
    │   └── test_build_features.py
    └── fixtures/
        └── synthetic_prices.py    # NEW: pytest helper to build deterministic price series
```

**Why this split:** Each feature family (returns, volatility, regime) is independently testable with its own synthetic data. The `pipeline` module is a thin composer — it concatenates Series from the family modules into the final DataFrame. Gaps live in their own module because reindexing logic needs to be testable in isolation and is shared by all consumers.

---

## Schema contract

Every consumer of the feature pipeline (Phase 3 model trainer, Phase 6 backtest) depends on this schema. It is defined once in `schema.py` and enforced at materialization time.

```python
FEATURE_COLUMNS: list[str] = [
    "ret_1h", "ret_4h", "ret_24h", "ret_7d", "ret_30d",
    "vol_24h", "vol_7d", "vol_30d",
    "zscore_30d", "zscore_90d",
    "skew_24h", "skew_7d",
    "kurt_24h", "kurt_7d",
    "ker_24h",
]

FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    **{col: pl.Float64() for col in FEATURE_COLUMNS},
}
```

---

## Task 1: Synthetic price fixture + gap utilities

**Files:**
- Create: `tests/fixtures/__init__.py` (if not present)
- Create: `tests/fixtures/synthetic_prices.py`
- Create: `src/btc_portfolio_mgr/features/__init__.py`
- Create: `src/btc_portfolio_mgr/features/gaps.py`
- Create: `tests/features/__init__.py`
- Create: `tests/features/test_gaps.py`

### Why first

Every other test in this phase needs deterministic small price series. Centralizing them in `synthetic_prices.py` avoids duplication. The gap utilities (`reindex_to_hourly`, `find_gaps`) are dependencies for every other feature module because gap-aware computation requires a complete-hourly index.

- [ ] **Step 1.1: Write the fixture helper**

`tests/fixtures/__init__.py` already exists from Phase 1 — leave alone. Create the helper module separately:

`tests/fixtures/synthetic_prices.py`:
```python
"""Deterministic price-series builders for feature-engineering tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from btc_portfolio_mgr.data.storage import SCHEMA


def hourly(prices: list[float], start: datetime | None = None) -> pl.DataFrame:
    """Build a contiguous hourly price DataFrame from a price list."""
    start = start or datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i, p in enumerate(prices):
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "price": float(p),
                "volume": 0.0,
            }
        )
    return pl.DataFrame(rows, schema=SCHEMA)


def with_gap(
    prices_before: list[float],
    gap_hours: int,
    prices_after: list[float],
    start: datetime | None = None,
) -> pl.DataFrame:
    """Two contiguous hourly blocks separated by `gap_hours` missing hours."""
    start = start or datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    before = hourly(prices_before, start)
    after_start = start + timedelta(hours=len(prices_before) + gap_hours)
    after = hourly(prices_after, after_start)
    return pl.concat([before, after])
```

- [ ] **Step 1.2: Write the failing test for gaps**

`tests/features/__init__.py`: empty file

`tests/features/test_gaps.py`:
```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from btc_portfolio_mgr.features.gaps import find_gaps, reindex_to_hourly
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_reindex_contiguous_is_unchanged() -> None:
    prices = hourly([100.0, 101.0, 102.0])
    reindexed = reindex_to_hourly(prices)
    assert reindexed.height == 3
    assert reindexed["price"].to_list() == [100.0, 101.0, 102.0]
    assert reindexed["price"].null_count() == 0


def test_reindex_fills_gap_with_nulls() -> None:
    prices = with_gap([100.0, 101.0], gap_hours=3, prices_after=[110.0, 111.0])
    reindexed = reindex_to_hourly(prices)
    # total span: 2 + 3 + 2 = 7 hours
    assert reindexed.height == 7
    assert reindexed["price"].to_list() == [100.0, 101.0, None, None, None, 110.0, 111.0]


def test_find_gaps_returns_empty_for_contiguous() -> None:
    prices = hourly([100.0, 101.0, 102.0])
    gaps = find_gaps(prices)
    assert gaps.height == 0


def test_find_gaps_reports_each_missing_block() -> None:
    prices = with_gap([100.0, 101.0], gap_hours=3, prices_after=[110.0])
    gaps = find_gaps(prices)
    assert gaps.height == 1
    row = gaps.row(0, named=True)
    assert row["gap_start"] == datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    assert row["gap_end"] == datetime(2024, 1, 1, 4, 0, tzinfo=timezone.utc)
    assert row["missing_hours"] == 3
```

- [ ] **Step 1.3: Run test to verify it fails**

Run: `cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr && .venv/bin/pytest tests/features/test_gaps.py -v`
Expected: ImportError for `btc_portfolio_mgr.features.gaps`.

- [ ] **Step 1.4: Write the gaps module**

`src/btc_portfolio_mgr/features/__init__.py`:
```python
```

`src/btc_portfolio_mgr/features/gaps.py`:
```python
"""Gap detection and complete-hourly reindexing for the feature pipeline."""
from __future__ import annotations

import polars as pl


def reindex_to_hourly(prices: pl.DataFrame) -> pl.DataFrame:
    """Reindex to a complete hourly grid from min to max timestamp.

    Missing hours appear as rows with null price/volume so downstream
    rolling operations naturally produce nulls when their window
    spans a gap.
    """
    if prices.height == 0:
        return prices
    start = prices["timestamp"].min()
    end = prices["timestamp"].max()
    grid = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start, end, interval="1h", time_zone="UTC", eager=True
            )
        }
    )
    return grid.join(prices, on="timestamp", how="left")


def find_gaps(prices: pl.DataFrame) -> pl.DataFrame:
    """Return a frame of (gap_start, gap_end, missing_hours) for each gap.

    A gap is a run of missing hourly timestamps between two present rows.
    gap_start is the first missing hour; gap_end is the last missing hour.
    """
    reindexed = reindex_to_hourly(prices)
    if reindexed.height == 0:
        return pl.DataFrame(
            schema={
                "gap_start": pl.Datetime("us", "UTC"),
                "gap_end": pl.Datetime("us", "UTC"),
                "missing_hours": pl.Int64,
            }
        )
    is_missing = reindexed["price"].is_null()
    if not is_missing.any():
        return pl.DataFrame(
            schema={
                "gap_start": pl.Datetime("us", "UTC"),
                "gap_end": pl.Datetime("us", "UTC"),
                "missing_hours": pl.Int64,
            }
        )
    # Identify runs of nulls
    timestamps = reindexed["timestamp"].to_list()
    missing_flags = is_missing.to_list()
    gaps: list[dict] = []
    i = 0
    n = len(missing_flags)
    while i < n:
        if missing_flags[i]:
            j = i
            while j < n and missing_flags[j]:
                j += 1
            gaps.append(
                {
                    "gap_start": timestamps[i],
                    "gap_end": timestamps[j - 1],
                    "missing_hours": j - i,
                }
            )
            i = j
        else:
            i += 1
    return pl.DataFrame(
        gaps,
        schema={
            "gap_start": pl.Datetime("us", "UTC"),
            "gap_end": pl.Datetime("us", "UTC"),
            "missing_hours": pl.Int64,
        },
    )
```

- [ ] **Step 1.5: Run test to verify it passes**

Run: `cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr && .venv/bin/pytest tests/features/test_gaps.py -v`
Expected: 4 passed.

- [ ] **Step 1.6: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/features/gaps.py tests/features/test_gaps.py tests/fixtures/synthetic_prices.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 1.7: Commit**

```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
git add tests/fixtures/synthetic_prices.py src/btc_portfolio_mgr/features/__init__.py src/btc_portfolio_mgr/features/gaps.py tests/features/__init__.py tests/features/test_gaps.py
git commit -m "feat(features): gap detection and hourly reindexing"
```

---

## Task 2: Returns module (multi-lookback log returns)

**Files:**
- Create: `src/btc_portfolio_mgr/features/returns.py`
- Create: `tests/features/test_returns.py`

The returns module exposes `compute_log_return(prices, lookback_hours) -> pl.Series`. The input is assumed reindexed (use `reindex_to_hourly` first) so `shift(L)` aligns correctly across gaps. Where either `price_t` or `price_{t-L}` is null, the output is null.

- [ ] **Step 2.1: Write the failing tests**

`tests/features/test_returns.py`:
```python
from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.returns import compute_log_return
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_log_return_constant_price_is_zero() -> None:
    prices = reindex_to_hourly(hourly([100.0] * 5))
    rets = compute_log_return(prices, lookback_hours=1)
    # First row has no prior price -> null; rest are 0.0
    values = rets.to_list()
    assert values[0] is None
    assert all(v == 0.0 for v in values[1:])


def test_log_return_known_values() -> None:
    prices = reindex_to_hourly(hourly([100.0, 110.0, 121.0]))
    rets = compute_log_return(prices, lookback_hours=1)
    values = rets.to_list()
    assert values[0] is None
    assert math.isclose(values[1], math.log(1.1))
    assert math.isclose(values[2], math.log(1.1))


def test_log_return_longer_lookback() -> None:
    # 4h return on a series that doubled over 4 hours
    prices = reindex_to_hourly(hourly([100.0, 105.0, 110.0, 115.0, 200.0]))
    rets = compute_log_return(prices, lookback_hours=4)
    values = rets.to_list()
    assert all(v is None for v in values[:4])
    assert math.isclose(values[4], math.log(2.0))


def test_log_return_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0], gap_hours=2, prices_after=[110.0, 111.0])
    )
    # Layout: [100, 101, null, null, 110, 111]
    # 1h returns: [None, log(1.01), None, None, None, log(111/110)]
    rets = compute_log_return(prices, lookback_hours=1)
    values = rets.to_list()
    assert values[0] is None
    assert math.isclose(values[1], math.log(101 / 100))
    assert values[2] is None
    assert values[3] is None
    assert values[4] is None  # prev row is null
    assert math.isclose(values[5], math.log(111 / 110))
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/features/test_returns.py -v`
Expected: ImportError for `btc_portfolio_mgr.features.returns`.

- [ ] **Step 2.3: Write the returns module**

`src/btc_portfolio_mgr/features/returns.py`:
```python
"""Multi-lookback log-return features."""
from __future__ import annotations

import polars as pl


def compute_log_return(prices: pl.DataFrame, lookback_hours: int) -> pl.Series:
    """log(price_t / price_{t-lookback_hours}).

    Assumes `prices` is reindexed to a complete hourly grid (use
    `gaps.reindex_to_hourly` first). Returns null where either endpoint
    of the lookback window is missing.
    """
    if lookback_hours <= 0:
        raise ValueError(f"lookback_hours must be positive, got {lookback_hours}")
    expr = pl.col("price").log() - pl.col("price").shift(lookback_hours).log()
    return prices.select(expr.alias("ret")).get_column("ret")
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/features/test_returns.py -v`
Expected: 4 passed.

- [ ] **Step 2.5: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/features/returns.py tests/features/test_returns.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 2.6: Commit**

```bash
git add src/btc_portfolio_mgr/features/returns.py tests/features/test_returns.py
git commit -m "feat(features): multi-lookback log returns"
```

---

## Task 3: Volatility module (realized vol, skew, kurtosis)

**Files:**
- Create: `src/btc_portfolio_mgr/features/volatility.py`
- Create: `tests/features/test_volatility.py`

Three functions:
- `compute_realized_vol(prices, window_hours)` — `sqrt(sum(ret_1h^2))` over the window.
- `compute_realized_skew(prices, window_hours)` — sample skew of 1h returns over the window. Use polars `rolling_skew`.
- `compute_realized_kurt(prices, window_hours)` — sample **excess** kurtosis over the window. Polars has no built-in rolling kurtosis; compute via numpy loop. For window W, requires W full returns (so first W rows are null).

All three operate on already-reindexed prices and produce null where the rolling window contains any null return.

- [ ] **Step 3.1: Write the failing tests**

`tests/features/test_volatility.py`:
```python
from __future__ import annotations

import math

import numpy as np
import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.volatility import (
    compute_realized_kurt,
    compute_realized_skew,
    compute_realized_vol,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_realized_vol_constant_price_is_zero() -> None:
    # 1h returns are all 0, so realized vol = 0 once window is full.
    prices = reindex_to_hourly(hourly([100.0] * 10))
    vol = compute_realized_vol(prices, window_hours=3)
    values = vol.to_list()
    # First (window) returns are null/incomplete -> vol null until index = window
    # ret series: [None, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # rolling sum of ret^2 over 3 hours with min_samples=3:
    # index 0: None; 1: None; 2: None (only 1 ret in window); 3: 0; ...
    assert values[0] is None
    assert values[1] is None
    assert values[2] is None
    assert values[3] == 0.0


def test_realized_vol_known_value() -> None:
    # Two 1h log returns each = log(2); vol over window=2 = sqrt(2 * log(2)^2)
    prices = reindex_to_hourly(hourly([100.0, 200.0, 400.0]))
    vol = compute_realized_vol(prices, window_hours=2)
    values = vol.to_list()
    # rets: [None, log(2), log(2)]
    # vol over window=2 needs 2 returns; first valid at index 2
    assert values[0] is None
    assert values[1] is None
    expected = math.sqrt(2 * (math.log(2) ** 2))
    assert math.isclose(values[2], expected)


def test_realized_vol_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0, 102.0], gap_hours=2, prices_after=[110.0, 111.0, 112.0])
    )
    # 1h returns: [None, r1, r2, None, None, None, r3, r4]
    # vol with window=2 -> null until index>=2 with two consecutive non-null rets.
    # index 2: rets at idx 1,2 -> both real -> non-null
    # index 3,4,5,6: any null in window -> null
    # index 7: rets at idx 6,7 -> both real -> non-null
    vol = compute_realized_vol(prices, window_hours=2)
    values = vol.to_list()
    assert values[2] is not None
    assert values[3] is None
    assert values[4] is None
    assert values[5] is None
    assert values[6] is None
    assert values[7] is not None


def test_realized_skew_symmetric_is_near_zero() -> None:
    # Construct returns symmetric around 0: prices 100, 110, 100, 110, 100
    # 1h log returns: [None, log(1.1), log(10/11), log(1.1), log(10/11)]
    prices = reindex_to_hourly(hourly([100.0, 110.0, 100.0, 110.0, 100.0]))
    skew = compute_realized_skew(prices, window_hours=4)
    values = skew.to_list()
    # First valid skew at index >= 4 (need 4 returns).
    assert values[4] is not None
    assert abs(values[4]) < 0.5  # roughly symmetric


def test_realized_kurt_constant_returns_is_undefined() -> None:
    # Constant returns => std=0 => kurtosis undefined; we return null.
    prices = reindex_to_hourly(hourly([100.0] * 10))
    kurt = compute_realized_kurt(prices, window_hours=4)
    values = kurt.to_list()
    assert values[5] is None  # std of all-zero returns is 0 => null


def test_realized_kurt_known_distribution() -> None:
    # Use a wider window so the moment estimate is stable.
    # Returns drawn from a known distribution: standard normal-ish.
    rng = np.random.default_rng(42)
    # Build a price series so that 1h returns approximate N(0, 0.01)
    n = 200
    returns = rng.normal(0, 0.01, n)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * math.exp(r))
    df = reindex_to_hourly(hourly(prices))
    kurt = compute_realized_kurt(df, window_hours=100)
    last_kurt = kurt.to_list()[-1]
    assert last_kurt is not None
    # Excess kurtosis of N(0,1) is 0; finite-sample noise is large -> tolerate ±1.5.
    assert abs(last_kurt) < 1.5
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/features/test_volatility.py -v`
Expected: ImportError for `btc_portfolio_mgr.features.volatility`.

- [ ] **Step 3.3: Write the volatility module**

`src/btc_portfolio_mgr/features/volatility.py`:
```python
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
    return pl.Series(out, dtype=pl.Float64)
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/features/test_volatility.py -v`
Expected: 6 passed.

- [ ] **Step 3.5: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/features/volatility.py tests/features/test_volatility.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 3.6: Commit**

```bash
git add src/btc_portfolio_mgr/features/volatility.py tests/features/test_volatility.py
git commit -m "feat(features): realized vol, skew, excess kurtosis"
```

---

## Task 4: Regime module (MA z-score + Kaufman Efficiency)

**Files:**
- Create: `src/btc_portfolio_mgr/features/regime.py`
- Create: `tests/features/test_regime.py`

Two functions:
- `compute_ma_zscore(prices, window_hours)` — `(price - rolling_mean) / rolling_std`. Operates on the raw price series, not on returns. Null when rolling stats are null or std == 0.
- `compute_kaufman_efficiency(prices, window_hours)` — `|price_t - price_{t-window}| / sum(|price_i - price_{i-1}|)` over the window. Ranges 0 (pure noise) to 1 (pure trend). We use this instead of ADX because we have close-only data, not OHLC.

- [ ] **Step 4.1: Write the failing tests**

`tests/features/test_regime.py`:
```python
from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.regime import (
    compute_kaufman_efficiency,
    compute_ma_zscore,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_zscore_constant_price_is_null() -> None:
    # std == 0 -> zscore undefined -> null
    prices = reindex_to_hourly(hourly([100.0] * 10))
    z = compute_ma_zscore(prices, window_hours=4)
    assert z.to_list()[5] is None


def test_zscore_one_sigma_above_mean() -> None:
    # Build a window where the latest price is exactly mean + 1 std.
    # Simple symmetric series: 100, 102, 100, 102, 100, 102, 100, 102 then 104.
    # The 4-hour window ending at the last index = [100, 102, 100, 104].
    # We expect a sensible nonzero z.
    prices = reindex_to_hourly(hourly([100, 102, 100, 102, 100, 102, 100, 102, 104]))
    z = compute_ma_zscore(prices, window_hours=4)
    # Last value should be positive (above mean) and finite.
    last = z.to_list()[-1]
    assert last is not None
    assert last > 0


def test_zscore_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0, 102.0, 103.0], gap_hours=2, prices_after=[110.0, 111.0])
    )
    # Reindexed: [100, 101, 102, 103, null, null, 110, 111]
    # zscore with window=4: rolling needs 4 non-null prices in window.
    # idx 3: [100,101,102,103] -> non-null
    # idx 4..6: window contains nulls -> null
    # idx 7: [null,null,110,111] -> still has nulls -> null
    z = compute_ma_zscore(prices, window_hours=4)
    values = z.to_list()
    assert values[3] is not None
    assert values[4] is None
    assert values[5] is None
    assert values[6] is None
    assert values[7] is None


def test_kaufman_efficiency_monotonic_is_one() -> None:
    # Monotonic increase: |p_t - p_{t-n}| == sum(|p_i - p_{i-1}|) -> KER = 1
    prices = reindex_to_hourly(hourly([100, 101, 102, 103, 104, 105]))
    ker = compute_kaufman_efficiency(prices, window_hours=4)
    values = ker.to_list()
    # First valid at index = window_hours (needs window+1 prices for 4 diffs)
    assert math.isclose(values[4], 1.0)
    assert math.isclose(values[5], 1.0)


def test_kaufman_efficiency_oscillation_is_low() -> None:
    # Pure oscillation: net change = 0 -> KER = 0
    prices = reindex_to_hourly(hourly([100, 101, 100, 101, 100, 101]))
    ker = compute_kaufman_efficiency(prices, window_hours=4)
    values = ker.to_list()
    assert math.isclose(values[4], 0.0)
    assert math.isclose(values[5], 0.0)


def test_kaufman_efficiency_null_across_gap() -> None:
    prices = reindex_to_hourly(
        with_gap([100.0, 101.0, 102.0], gap_hours=2, prices_after=[110.0, 111.0])
    )
    ker = compute_kaufman_efficiency(prices, window_hours=4)
    # Window=4 needs 4 abs differences -> 5 non-null prices in a row.
    # Within the [100,101,102,null,null,110,111] sequence (reindexed length 7),
    # no 5-prices-contiguous segment exists, so all KER values are null.
    assert all(v is None for v in ker.to_list())
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/features/test_regime.py -v`
Expected: ImportError for `btc_portfolio_mgr.features.regime`.

- [ ] **Step 4.3: Write the regime module**

`src/btc_portfolio_mgr/features/regime.py`:
```python
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
    # Compute via numpy so we can map inf/NaN -> null cleanly.
    raw = ((p - mean) / std).to_numpy()
    raw = np.where(np.isfinite(raw), raw, np.nan)
    return pl.Series(raw, dtype=pl.Float64).fill_nan(None)


def compute_kaufman_efficiency(
    prices: pl.DataFrame, window_hours: int
) -> pl.Series:
    """Kaufman Efficiency Ratio over the window.

    KER = |price_t - price_{t-window}| / sum(|price_i - price_{i-1}|)
    over i in (t-window+1 .. t). Range [0, 1]. Null when any input in the
    window is null, or when the denominator is 0 (constant prices).

    Implementation: numpy loop. O(n * window) but materialized once.
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
    return pl.Series(out, dtype=pl.Float64)
```

A subtlety: when `path == 0` (constant prices) AND `net == 0`, KER is mathematically 0/0. We return 0.0 (no trend) rather than null because the *meaning* is "no movement at all" which the model should interpret as zero efficiency. This is a deliberate exception to the null-on-degenerate rule. The test `test_zscore_constant_price_is_null` uses null for z-score (different feature) because z-score on a flat series genuinely is undefined.

- [ ] **Step 4.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/features/test_regime.py -v`
Expected: 6 passed.

- [ ] **Step 4.5: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/features/regime.py tests/features/test_regime.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 4.6: Commit**

```bash
git add src/btc_portfolio_mgr/features/regime.py tests/features/test_regime.py
git commit -m "feat(features): MA z-score and Kaufman Efficiency Ratio"
```

---

## Task 5: Schema + pipeline composition

**Files:**
- Create: `src/btc_portfolio_mgr/features/schema.py`
- Create: `src/btc_portfolio_mgr/features/pipeline.py`
- Create: `tests/features/test_pipeline.py`

The pipeline calls every feature module with the right lookbacks and assembles a DataFrame matching FEATURE_SCHEMA. The schema is the contract Phase 3+ consumes.

- [ ] **Step 5.1: Write the schema module**

`src/btc_portfolio_mgr/features/schema.py`:
```python
"""Feature matrix schema — the contract consumed by Phase 3+ trainers."""
from __future__ import annotations

import polars as pl

FEATURE_COLUMNS: list[str] = [
    "ret_1h", "ret_4h", "ret_24h", "ret_7d", "ret_30d",
    "vol_24h", "vol_7d", "vol_30d",
    "zscore_30d", "zscore_90d",
    "skew_24h", "skew_7d",
    "kurt_24h", "kurt_7d",
    "ker_24h",
]

FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    **{col: pl.Float64() for col in FEATURE_COLUMNS},
}


class FeatureSchemaMismatchError(ValueError):
    """Raised when a DataFrame's schema does not match FEATURE_SCHEMA."""


def assert_feature_schema(df: pl.DataFrame) -> None:
    if df.schema != FEATURE_SCHEMA:
        raise FeatureSchemaMismatchError(
            f"expected {FEATURE_SCHEMA}, got {df.schema}"
        )
```

- [ ] **Step 5.2: Write the failing tests**

`tests/features/test_pipeline.py`:
```python
from __future__ import annotations

import math

import polars as pl
import pytest

from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.features.schema import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    FeatureSchemaMismatchError,
    assert_feature_schema,
)
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_compose_features_returns_canonical_schema() -> None:
    # Need at least 90d = 2160h of data for the longest lookback (zscore_90d).
    prices = hourly([100.0 + i * 0.01 for i in range(2200)])
    features = compose_features(prices)
    assert features.schema == FEATURE_SCHEMA
    # The columns appear in deterministic order
    assert features.columns == ["timestamp"] + FEATURE_COLUMNS


def test_compose_features_emits_one_row_per_input_hour() -> None:
    prices = hourly([100.0 + i * 0.01 for i in range(2200)])
    features = compose_features(prices)
    assert features.height == 2200


def test_compose_features_early_rows_are_null_for_long_lookbacks() -> None:
    prices = hourly([100.0 + i * 0.01 for i in range(2200)])
    features = compose_features(prices)
    # ret_30d needs 720 prior hours; first 720 rows of ret_30d are null.
    ret_30d = features["ret_30d"].to_list()
    assert all(v is None for v in ret_30d[:720])
    assert ret_30d[720] is not None
    # zscore_90d needs 2160 prior hours.
    z90 = features["zscore_90d"].to_list()
    assert all(v is None for v in z90[:2159])


def test_compose_features_propagates_gap_nulls() -> None:
    # Tight series: 50 hours, then 5-hour gap, then 50 hours.
    prices = with_gap(
        prices_before=[100.0 + i * 0.01 for i in range(50)],
        gap_hours=5,
        prices_after=[110.0 + i * 0.01 for i in range(50)],
    )
    features = compose_features(prices)
    # Reindexed length: 50 + 5 + 50 = 105
    assert features.height == 105
    # Rows inside the gap have null prices -> all features null at those rows.
    # The 5 gap rows are indices 50..54.
    for idx in range(50, 55):
        for col in FEATURE_COLUMNS:
            assert features[col].to_list()[idx] is None, f"col {col} idx {idx}"


def test_assert_feature_schema_rejects_bad_input() -> None:
    bad = pl.DataFrame({"timestamp": [1], "ret_1h": [2.0]})
    with pytest.raises(FeatureSchemaMismatchError):
        assert_feature_schema(bad)
```

- [ ] **Step 5.3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/features/test_pipeline.py -v`
Expected: ImportError for `btc_portfolio_mgr.features.pipeline`.

- [ ] **Step 5.4: Write the pipeline module**

`src/btc_portfolio_mgr/features/pipeline.py`:
```python
"""Composes all feature families into the canonical feature matrix."""
from __future__ import annotations

import polars as pl

from btc_portfolio_mgr.features.gaps import reindex_to_hourly
from btc_portfolio_mgr.features.regime import (
    compute_kaufman_efficiency,
    compute_ma_zscore,
)
from btc_portfolio_mgr.features.returns import compute_log_return
from btc_portfolio_mgr.features.schema import FEATURE_SCHEMA, assert_feature_schema
from btc_portfolio_mgr.features.volatility import (
    compute_realized_kurt,
    compute_realized_skew,
    compute_realized_vol,
)

# Lookbacks expressed in hours.
_RETURN_LOOKBACKS: dict[str, int] = {
    "ret_1h": 1,
    "ret_4h": 4,
    "ret_24h": 24,
    "ret_7d": 168,
    "ret_30d": 720,
}
_VOL_WINDOWS: dict[str, int] = {
    "vol_24h": 24,
    "vol_7d": 168,
    "vol_30d": 720,
}
_ZSCORE_WINDOWS: dict[str, int] = {
    "zscore_30d": 720,
    "zscore_90d": 2160,
}
_SKEW_WINDOWS: dict[str, int] = {
    "skew_24h": 24,
    "skew_7d": 168,
}
_KURT_WINDOWS: dict[str, int] = {
    "kurt_24h": 24,
    "kurt_7d": 168,
}
_KER_WINDOWS: dict[str, int] = {
    "ker_24h": 24,
}


def compose_features(prices: pl.DataFrame) -> pl.DataFrame:
    """Build the full feature matrix from a hourly price DataFrame.

    Input: prices with canonical storage SCHEMA. Output: DataFrame with
    FEATURE_SCHEMA. One row per hour in the reindexed grid. Null entries
    where the relevant rolling window includes a data gap.
    """
    reindexed = reindex_to_hourly(prices)
    columns: dict[str, pl.Series] = {"timestamp": reindexed["timestamp"]}
    for name, lb in _RETURN_LOOKBACKS.items():
        columns[name] = compute_log_return(reindexed, lb).alias(name)
    for name, w in _VOL_WINDOWS.items():
        columns[name] = compute_realized_vol(reindexed, w).alias(name)
    for name, w in _ZSCORE_WINDOWS.items():
        columns[name] = compute_ma_zscore(reindexed, w).alias(name)
    for name, w in _SKEW_WINDOWS.items():
        columns[name] = compute_realized_skew(reindexed, w).alias(name)
    for name, w in _KURT_WINDOWS.items():
        columns[name] = compute_realized_kurt(reindexed, w).alias(name)
    for name, w in _KER_WINDOWS.items():
        columns[name] = compute_kaufman_efficiency(reindexed, w).alias(name)
    out = pl.DataFrame(columns, schema=FEATURE_SCHEMA)
    assert_feature_schema(out)
    return out
```

- [ ] **Step 5.5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/features/test_pipeline.py -v`
Expected: 5 passed.

- [ ] **Step 5.6: Pyright check**

Run: `.venv/bin/pyright src/btc_portfolio_mgr/features/ tests/features/test_pipeline.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 5.7: Commit**

```bash
git add src/btc_portfolio_mgr/features/schema.py src/btc_portfolio_mgr/features/pipeline.py tests/features/test_pipeline.py
git commit -m "feat(features): schema contract and pipeline composer"
```

---

## Task 6: Build script (materialize features to parquet)

**Files:**
- Create: `scripts/build_features.py`
- Create: `tests/features/test_build_features.py`

Reads `data/btc_hourly.parquet`, runs `compose_features`, writes `data/btc_features.parquet`. Mirrors the Phase 1 backfill pattern (absolute path anchoring, no fallbacks on errors).

- [ ] **Step 6.1: Write the failing test**

`tests/features/test_build_features.py`:
```python
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from btc_portfolio_mgr.data.storage import write_parquet
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA
from tests.fixtures.synthetic_prices import hourly


def test_build_writes_features_parquet(tmp_path: Path) -> None:
    from scripts import build_features as bf

    prices_path = tmp_path / "btc_hourly.parquet"
    features_path = tmp_path / "btc_features.parquet"

    # 2200 hourly bars so even zscore_90d (2160h lookback) has non-null tail.
    write_parquet(hourly([100.0 + i * 0.01 for i in range(2200)]), prices_path)

    bf.run(prices_path=prices_path, features_path=features_path)

    loaded = pl.read_parquet(features_path)
    assert loaded.schema == FEATURE_SCHEMA
    assert loaded.height == 2200
    assert loaded.columns == ["timestamp"] + FEATURE_COLUMNS
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/features/test_build_features.py -v`
Expected: ModuleNotFoundError for `scripts.build_features`.

- [ ] **Step 6.3: Write the build script**

`scripts/build_features.py`:
```python
"""Materialize the feature matrix from hourly prices to parquet."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.features.pipeline import compose_features
from btc_portfolio_mgr.features.schema import assert_feature_schema

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "btc_features.parquet"


def run(
    prices_path: Path = DEFAULT_PRICES,
    features_path: Path = DEFAULT_FEATURES,
) -> None:
    prices = read_parquet(prices_path)
    features = compose_features(prices)
    assert_feature_schema(features)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(features_path)
    print(f"wrote {features.height} rows to {features_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/features/test_build_features.py -v`
Expected: 1 passed.

- [ ] **Step 6.5: Run full suite**

Run: `.venv/bin/pytest -v`
Expected: 15 (Phase 1) + 4 (gaps) + 4 (returns) + 6 (volatility) + 6 (regime) + 5 (pipeline) + 1 (build) = **41 passed**.

- [ ] **Step 6.6: Pyright check**

Run: `.venv/bin/pyright scripts/build_features.py tests/features/test_build_features.py 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 6.7: Commit**

```bash
git add scripts/build_features.py tests/features/test_build_features.py
git commit -m "feat(features): build_features.py materializes feature matrix"
```

---

## Task 7: Manual smoke test against real data

This is the only step that requires the Phase 1 historical backfill to have been run. Skip in continuous-execution / auto mode.

- [ ] **Step 7.1: Confirm historical data is present**

```bash
ls -lh /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr/data/btc_hourly.parquet
```
Expected: file exists, ~1–3 MB.

- [ ] **Step 7.2: Run the build**

```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
.venv/bin/python scripts/build_features.py
```
Expected: prints "wrote N rows to .../btc_features.parquet" where N matches the hourly bar count (~70k for 2018-now).

- [ ] **Step 7.3: Spot-check**

```bash
.venv/bin/python -c "
import polars as pl
df = pl.read_parquet('data/btc_features.parquet')
print(df.tail(5))
print()
print('null counts per column:')
print(df.null_count())
"
```
Sanity checks:
- The last row has non-null values for every feature (assuming no gap in the most recent 90 days).
- `null_count()` shows ~720 for short-lookback features, ~2160 for `zscore_90d`. If null counts are much higher, there are data gaps worth investigating with `find_gaps`.

---

## Done criteria (Phase 2)

- [ ] `pytest -v` passes 41 tests
- [ ] `pyright src/btc_portfolio_mgr/features/ scripts/build_features.py` reports 0 errors
- [ ] `scripts/build_features.py` produces `data/btc_features.parquet` with FEATURE_SCHEMA (manual smoke)
- [ ] `HourlyReader` still works (no regressions in Phase 1)

## What's deliberately not in Phase 2

- **Fast / microstructure features** (last-60-min vol from minute samples, L1 imbalance) — these need a minute-resolution recorder that doesn't exist yet. Deferred to Phase 7.
- **Cross-asset features** (BTC/ETH ratio, BTC dominance) — only matter when Phase 5 considers multi-asset sizing. Not relevant for a BTC-only model.
- **Volume-based features** (OBV, VWAP-deviation) — CoinGecko volume quality unverified; we don't depend on it. Add when needed.
- **Funding-rate features** — separate Binance endpoint; not part of price data layer.
- **Feature selection / importance analysis** — that's a Phase 3 model concern.
- **Online updating** of features for live inference — for now, materialize and re-run. Phase 7 can add streaming feature updates if needed.
