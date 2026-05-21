# Phase 1: Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data layer that supplies historical hourly BTC prices (CoinGecko 2018-now, daily 2013-2018) for training and live Binance WebSocket ticks for serving, with a unified reader interface for downstream phases.

**Architecture:** Two sources reconciled through a common storage format (parquet via polars). CoinGecko Pro REST API for backfill, Binance bookTicker WebSocket for live, both writing to the same on-disk schema so backtests and live inference use identical feature inputs.

**Tech Stack:** Python 3.11+, uv (env/deps), polars (dataframes), httpx (REST), websockets (Binance WS), pytest + respx (testing), parquet/pyarrow (storage).

---

## Phasing roadmap (this is Phase 1 of 7)

| Phase | Subsystem | Depends on |
|---|---|---|
| **1. Data layer** (this plan) | CoinGecko historical + Binance WS live + unified reader | none |
| 2. Feature engineering | Multi-lookback returns, realized vol, regime z-scores | Phase 1 |
| 3. Return model | LightGBM regressor for 1h-ahead log return | Phase 2 |
| 4. Vol model | GJR-GARCH(1,1) for 1h-ahead vol | Phase 2 |
| 5. Sizing | Fractional Kelly + vol-targeting + caps | Phases 3, 4 |
| 6. Backtest | Walk-forward + López de Prado purged k-fold | Phases 3, 4, 5 |
| 7. Execution | Binance trade API client (via Helsinki egress) | Phase 5 |

Each phase produces independently-testable software. Land Phase 1 before starting Phase 2.

---

## File structure (Phase 1)

```
btc-portfolio-mgr/
├── pyproject.toml                       # uv-managed deps + project metadata
├── .gitignore
├── .python-version                      # 3.11
├── README.md
├── src/btc_portfolio_mgr/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── coingecko.py                 # CoinGecko Pro client + range fetcher
│       ├── binance_ws.py                # Binance bookTicker WS + hourly resampler
│       ├── storage.py                   # parquet read/write with schema
│       └── reader.py                    # unified historical+live reader
├── scripts/
│   └── backfill_historical.py           # one-shot CoinGecko backfill to parquet
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # shared fixtures
│   ├── fixtures/
│   │   ├── coingecko_range_response.json
│   │   └── binance_bookticker_stream.jsonl
│   └── data/
│       ├── __init__.py
│       ├── test_coingecko.py
│       ├── test_binance_ws.py
│       ├── test_storage.py
│       └── test_reader.py
└── data/                                # gitignored; backfilled parquet lives here
    └── .gitkeep
```

**Schema invariant** (used everywhere): polars DataFrame with columns
- `timestamp: Datetime("us", "UTC")` — bar close time
- `price: Float64` — close mid price in USD
- `volume: Float64` — interval volume in base units (nullable for live-resampled bars where volume isn't tracked)

All readers/writers MUST produce this schema. Failing this invariant is a fatal error — no fallback (per project convention: imports/contracts that fail should fail loudly).

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.python-version`
- Create: `README.md`
- Create: `src/btc_portfolio_mgr/__init__.py`
- Create: `src/btc_portfolio_mgr/data/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `data/.gitkeep`

- [ ] **Step 1.1: Write `pyproject.toml`**

```toml
[project]
name = "btc-portfolio-mgr"
version = "0.1.0"
description = "Systematic BTC portfolio management with hourly rebalancing."
requires-python = ">=3.11"
dependencies = [
    "polars>=1.0",
    "httpx>=0.27",
    "websockets>=12.0",
    "pyarrow>=16.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "ruff>=0.5",
    "pyright>=1.1.350",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/btc_portfolio_mgr"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 1.2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
data/*.parquet
data/*.jsonl
data/*.jsonl.gz
.env
*.egg-info/
dist/
build/
.DS_Store
```

- [ ] **Step 1.3: Write `.python-version`**

```
3.11
```

- [ ] **Step 1.4: Write `README.md`**

```markdown
# btc-portfolio-mgr

Systematic BTC portfolio management with hourly rebalancing.
Hourly return forecast (LightGBM) + GJR-GARCH vol model → fractional-Kelly vol-targeted sizing.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env  # fill in COINGECKO_API_KEY
```

## Backfill historical

```bash
python scripts/backfill_historical.py
```

## Run tests

```bash
pytest -v
```
```

- [ ] **Step 1.5: Write the empty package files**

`src/btc_portfolio_mgr/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/btc_portfolio_mgr/data/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

`data/.gitkeep`:
```
```

- [ ] **Step 1.6: Write `tests/conftest.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def coingecko_range_response() -> dict:
    with (FIXTURES_DIR / "coingecko_range_response.json").open() as f:
        return json.load(f)


@pytest.fixture
def binance_bookticker_lines() -> list[str]:
    with (FIXTURES_DIR / "binance_bookticker_stream.jsonl").open() as f:
        return [line.rstrip("\n") for line in f if line.strip()]
```

- [ ] **Step 1.7: Initialize venv and install**

Run:
```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
uv venv
uv pip install -e ".[dev]"
```
Expected: venv created, all deps installed, no errors.

- [ ] **Step 1.8: Run pytest (should find no tests yet)**

Run: `cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr && .venv/bin/pytest -v`
Expected: `no tests ran` (exit code 5 is OK at this stage).

- [ ] **Step 1.9: Commit**

```bash
git add pyproject.toml .gitignore .python-version README.md src/ tests/ data/.gitkeep
git commit -m "chore: project scaffolding (uv + polars + pytest)"
```

---

## Task 2: Storage layer (parquet read/write with schema)

**Files:**
- Create: `src/btc_portfolio_mgr/data/storage.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_storage.py`

The schema invariant is enforced here. Every read returns the canonical schema; every write rejects mismatched input. This is the contract everything else builds on.

- [ ] **Step 2.1: Write the failing test**

`tests/data/test_storage.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from btc_portfolio_mgr.data.storage import (
    SCHEMA,
    SchemaMismatchError,
    read_parquet,
    write_parquet,
)


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "price": [42_000.0, 42_100.5],
            "volume": [1.23, 4.56],
        },
        schema=SCHEMA,
    )


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    df = _sample_df()
    out = tmp_path / "btc.parquet"
    write_parquet(df, out)
    loaded = read_parquet(out)
    assert loaded.schema == SCHEMA
    assert loaded.equals(df)


def test_write_rejects_wrong_schema(tmp_path: Path) -> None:
    bad = pl.DataFrame({"ts": [1], "px": [2.0]})
    with pytest.raises(SchemaMismatchError):
        write_parquet(bad, tmp_path / "bad.parquet")


def test_read_rejects_wrong_schema(tmp_path: Path) -> None:
    bad = pl.DataFrame({"ts": [1], "px": [2.0]})
    out = tmp_path / "bad.parquet"
    bad.write_parquet(out)
    with pytest.raises(SchemaMismatchError):
        read_parquet(out)
```

`tests/data/__init__.py`:
```python
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_storage.py -v`
Expected: ImportError / ModuleNotFoundError for `btc_portfolio_mgr.data.storage`.

- [ ] **Step 2.3: Write the storage module**

`src/btc_portfolio_mgr/data/storage.py`:
```python
from __future__ import annotations

from pathlib import Path

import polars as pl

SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "price": pl.Float64,
    "volume": pl.Float64,
}


class SchemaMismatchError(ValueError):
    """Raised when a DataFrame's schema does not match the canonical SCHEMA."""


def _assert_schema(df: pl.DataFrame) -> None:
    if df.schema != SCHEMA:
        raise SchemaMismatchError(
            f"expected schema {SCHEMA}, got {df.schema}"
        )


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    _assert_schema(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_parquet(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    _assert_schema(df)
    return df
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/data/test_storage.py -v`
Expected: 3 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/btc_portfolio_mgr/data/storage.py tests/data/__init__.py tests/data/test_storage.py
git commit -m "feat(data): parquet storage with canonical schema enforcement"
```

---

## Task 3: CoinGecko client — range fetcher with chunking

**Files:**
- Create: `tests/fixtures/coingecko_range_response.json`
- Create: `src/btc_portfolio_mgr/data/coingecko.py`
- Create: `tests/data/test_coingecko.py`

CoinGecko Pro `/coins/{id}/market_chart/range` returns hourly data only if the requested range is ≤90 days. We chunk longer ranges into 90-day windows. The fixture is a minimal 2-point response; chunking logic is unit-tested separately from any HTTP call.

- [ ] **Step 3.1: Write the fixture**

`tests/fixtures/coingecko_range_response.json`:
```json
{
  "prices": [
    [1704067200000, 42000.0],
    [1704070800000, 42100.5]
  ],
  "market_caps": [
    [1704067200000, 820000000000.0],
    [1704070800000, 821000000000.0]
  ],
  "total_volumes": [
    [1704067200000, 1500000000.0],
    [1704070800000, 1600000000.0]
  ]
}
```

- [ ] **Step 3.2: Write the failing tests**

`tests/data/test_coingecko.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
import respx
from httpx import Response

from btc_portfolio_mgr.data.coingecko import (
    CoinGeckoClient,
    chunk_date_range,
    parse_market_chart,
)
from btc_portfolio_mgr.data.storage import SCHEMA


def test_parse_market_chart_returns_canonical_schema(coingecko_range_response: dict) -> None:
    df = parse_market_chart(coingecko_range_response)
    assert df.schema == SCHEMA
    assert df.height == 2
    assert df["price"].to_list() == [42000.0, 42100.5]


def test_chunk_date_range_under_max_returns_single_chunk() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    chunks = chunk_date_range(start, end, max_days=90)
    assert chunks == [(start, end)]


def test_chunk_date_range_splits_long_ranges() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 7, 1, tzinfo=timezone.utc)  # ~182 days
    chunks = chunk_date_range(start, end, max_days=90)
    assert len(chunks) == 3
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    # No gaps, no overlaps
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[1] == nxt[0]


@respx.mock
async def test_client_fetch_range_calls_pro_api(
    coingecko_range_response: dict,
) -> None:
    route = respx.get("https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart/range").mock(
        return_value=Response(200, json=coingecko_range_response)
    )
    client = CoinGeckoClient(api_key="test-key")
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = await client.fetch_range("bitcoin", start, end)
    await client.aclose()
    assert route.called
    sent = route.calls[0].request
    assert sent.headers["x-cg-pro-api-key"] == "test-key"
    assert df.schema == SCHEMA
    assert df.height == 2


@respx.mock
async def test_client_chunks_long_range_into_multiple_calls(
    coingecko_range_response: dict,
) -> None:
    respx.get("https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart/range").mock(
        return_value=Response(200, json=coingecko_range_response)
    )
    client = CoinGeckoClient(api_key="test-key")
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 7, 1, tzinfo=timezone.utc)
    df = await client.fetch_range("bitcoin", start, end)
    await client.aclose()
    # 3 chunks × 2 rows per fixture = 6 rows total (dedupe handled separately downstream)
    assert df.height == 6
```

- [ ] **Step 3.3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/data/test_coingecko.py -v`
Expected: ImportError for `btc_portfolio_mgr.data.coingecko`.

- [ ] **Step 3.4: Write the coingecko module**

`src/btc_portfolio_mgr/data/coingecko.py`:
```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import polars as pl

from btc_portfolio_mgr.data.storage import SCHEMA

PRO_BASE = "https://pro-api.coingecko.com/api/v3"


def parse_market_chart(payload: dict) -> pl.DataFrame:
    """Convert /market_chart or /market_chart/range JSON to canonical schema."""
    prices = payload["prices"]
    volumes_by_ts = {ts: v for ts, v in payload["total_volumes"]}
    rows = []
    for ts_ms, price in prices:
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                "price": float(price),
                "volume": float(volumes_by_ts.get(ts_ms, 0.0)),
            }
        )
    return pl.DataFrame(rows, schema=SCHEMA)


def chunk_date_range(
    start: datetime, end: datetime, max_days: int
) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into contiguous (start, end) windows of at most max_days each."""
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    delta = timedelta(days=max_days)
    while cursor < end:
        next_cursor = min(cursor + delta, end)
        chunks.append((cursor, next_cursor))
        cursor = next_cursor
    return chunks


class CoinGeckoClient:
    """Async CoinGecko Pro client. Hourly data when chunk ≤ 90 days."""

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=PRO_BASE,
            headers={"x-cg-pro-api-key": api_key},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_range(
        self, coin_id: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Fetch all data points in [start, end]; chunks to respect hourly-granularity window."""
        frames: list[pl.DataFrame] = []
        for chunk_start, chunk_end in chunk_date_range(start, end, max_days=90):
            params = {
                "vs_currency": "usd",
                "from": int(chunk_start.timestamp()),
                "to": int(chunk_end.timestamp()),
            }
            r = await self._client.get(
                f"/coins/{coin_id}/market_chart/range", params=params
            )
            r.raise_for_status()
            frames.append(parse_market_chart(r.json()))
        return pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
```

- [ ] **Step 3.5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_coingecko.py -v`
Expected: 5 passed.

- [ ] **Step 3.6: Commit**

```bash
git add tests/fixtures/coingecko_range_response.json src/btc_portfolio_mgr/data/coingecko.py tests/data/test_coingecko.py
git commit -m "feat(data): CoinGecko Pro client with date-range chunking"
```

---

## Task 4: Backfill script

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/backfill_historical.py`
- Create: `.env.example`
- Create: `tests/data/test_backfill.py`

The script orchestrates: read API key from env, fetch 2018-now hourly, write to `data/btc_hourly.parquet`. Dedupe identical timestamps (chunk boundaries can produce overlap) and sort by timestamp before write.

- [ ] **Step 4.1: Write `.env.example`**

`.env.example`:
```
COINGECKO_API_KEY=your-pro-api-key-here
```

- [ ] **Step 4.2: Write the failing test**

`tests/data/test_backfill.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import polars as pl
import pytest

from btc_portfolio_mgr.data.storage import SCHEMA, read_parquet


@pytest.mark.asyncio
async def test_backfill_dedups_and_sorts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the client to return a frame with duplicate timestamps + out-of-order rows
    from scripts import backfill_historical as bh

    duplicated = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),  # duplicate
            ],
            "price": [42100.5, 42000.0, 42100.5],
            "volume": [4.56, 1.23, 4.56],
        },
        schema=SCHEMA,
    )

    fake_client = AsyncMock()
    fake_client.fetch_range.return_value = duplicated
    fake_client.aclose.return_value = None
    monkeypatch.setattr(bh, "CoinGeckoClient", lambda *a, **kw: fake_client)
    monkeypatch.setenv("COINGECKO_API_KEY", "test-key")

    out_path = tmp_path / "btc_hourly.parquet"
    await bh.run(
        out_path=out_path,
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    df = read_parquet(out_path)
    assert df.height == 2  # deduplicated
    assert df["timestamp"].is_sorted()
```

- [ ] **Step 4.3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_backfill.py -v`
Expected: ModuleNotFoundError for `scripts.backfill_historical`.

- [ ] **Step 4.4: Write the backfill script**

`scripts/__init__.py`:
```python
```

`scripts/backfill_historical.py`:
```python
"""Backfill BTC hourly history from CoinGecko Pro to data/btc_hourly.parquet."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from btc_portfolio_mgr.data.coingecko import CoinGeckoClient
from btc_portfolio_mgr.data.storage import write_parquet

DEFAULT_OUT = Path("data/btc_hourly.parquet")
DEFAULT_START = datetime(2018, 1, 1, tzinfo=timezone.utc)


async def run(
    out_path: Path = DEFAULT_OUT,
    start: datetime = DEFAULT_START,
    end: datetime | None = None,
) -> None:
    api_key = os.environ["COINGECKO_API_KEY"]
    end = end or datetime.now(tz=timezone.utc)
    client = CoinGeckoClient(api_key=api_key)
    try:
        raw = await client.fetch_range("bitcoin", start, end)
    finally:
        await client.aclose()
    cleaned = raw.unique(subset=["timestamp"], keep="first").sort("timestamp")
    write_parquet(cleaned, out_path)
    print(f"wrote {cleaned.height} rows to {out_path}")


def main() -> None:
    load_dotenv()
    asyncio.run(run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/data/test_backfill.py -v`
Expected: 1 passed.

- [ ] **Step 4.6: Commit**

```bash
git add scripts/__init__.py scripts/backfill_historical.py .env.example tests/data/test_backfill.py
git commit -m "feat(data): historical backfill script (CoinGecko -> parquet)"
```

---

## Task 5: Binance WebSocket client with hourly resampler

**Files:**
- Create: `tests/fixtures/binance_bookticker_stream.jsonl`
- Create: `src/btc_portfolio_mgr/data/binance_ws.py`
- Create: `tests/data/test_binance_ws.py`

Binance `bookTicker` is L1 quote updates. We track the mid price `(bid + ask) / 2` per tick and emit one "hourly close" row whenever the wall-clock hour rolls over. The resampling logic is pure (no WS connection in tests).

- [ ] **Step 5.1: Write the fixture**

`tests/fixtures/binance_bookticker_stream.jsonl`:
```
{"u":1,"s":"BTCUSDT","b":"42000.00","B":"1.0","a":"42001.00","A":"1.0","E":1704067200000}
{"u":2,"s":"BTCUSDT","b":"42005.00","B":"1.0","a":"42007.00","A":"1.0","E":1704068000000}
{"u":3,"s":"BTCUSDT","b":"42010.00","B":"1.0","a":"42012.00","A":"1.0","E":1704070799000}
{"u":4,"s":"BTCUSDT","b":"42020.00","B":"1.0","a":"42022.00","A":"1.0","E":1704070800000}
{"u":5,"s":"BTCUSDT","b":"42100.00","B":"1.0","a":"42102.00","A":"1.0","E":1704074000000}
```

Timestamps (UTC): u=1 at 00:00:00, u=2 at 00:13:20, u=3 at 00:59:59 (all hour 00 of 2024-01-01), u=4 at 01:00:00, u=5 at 01:53:20 (hour 01). When u=4 arrives, the hour rolls and the resampler emits hour-00's close = mid of u=3 = 42011.0. `flush()` then emits hour 01 with close = mid of u=5 = 42101.0.

- [ ] **Step 5.2: Write the failing test**

`tests/data/test_binance_ws.py`:
```python
from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

from btc_portfolio_mgr.data.binance_ws import HourlyResampler, parse_bookticker
from btc_portfolio_mgr.data.storage import SCHEMA


def test_parse_bookticker_computes_mid() -> None:
    raw = '{"u":1,"s":"BTCUSDT","b":"42000.00","B":"1.0","a":"42010.00","A":"1.0","E":1704067200000}'
    tick = parse_bookticker(raw)
    assert tick.mid == 42005.0
    assert tick.event_time == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_resampler_emits_one_row_per_completed_hour(
    binance_bookticker_lines: list[str],
) -> None:
    sampler = HourlyResampler()
    emitted: list[dict] = []
    for line in binance_bookticker_lines:
        tick = parse_bookticker(line)
        row = sampler.observe(tick)
        if row is not None:
            emitted.append(row)
    # 5 ticks spanning 2 hours: emit 1 row when hour 00 -> hour 01 rolls.
    # The currently-open hour 01 is not emitted until flush().
    assert len(emitted) == 1
    df = pl.DataFrame(emitted, schema=SCHEMA)
    assert df.schema == SCHEMA
    assert df["timestamp"].to_list() == [
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    ]
    # Close of hour 00:00 = mid of the LAST tick within hour 00 (u=3, mid=42011.0).
    # The hour-rolling tick (u=4) is the FIRST tick of the next hour, not the previous close.
    assert df["price"].to_list() == [42011.0]


def test_resampler_flush_emits_partial_hour(binance_bookticker_lines: list[str]) -> None:
    sampler = HourlyResampler()
    for line in binance_bookticker_lines:
        sampler.observe(parse_bookticker(line))
    final = sampler.flush()
    assert final is not None
    assert final["timestamp"] == datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
    assert final["price"] == 42101.0  # mid of last tick (u=5)
```

- [ ] **Step 5.3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_binance_ws.py -v`
Expected: ImportError for `btc_portfolio_mgr.data.binance_ws`.

- [ ] **Step 5.4: Write the binance_ws module**

`src/btc_portfolio_mgr/data/binance_ws.py`:
```python
"""Binance bookTicker WebSocket client + hourly close resampler."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import websockets

WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"


@dataclass(frozen=True)
class BookTickerTick:
    event_time: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def parse_bookticker(raw: str) -> BookTickerTick:
    msg = json.loads(raw)
    return BookTickerTick(
        event_time=datetime.fromtimestamp(msg["E"] / 1000, tz=timezone.utc),
        bid=float(msg["b"]),
        ask=float(msg["a"]),
    )


def _hour_floor(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


class HourlyResampler:
    """Aggregates tick stream into completed-hour close prices.

    `observe(tick)` returns a row dict (matching SCHEMA) when the hour rolls over,
    else None. `flush()` emits the currently-open hour as a partial close.
    """

    def __init__(self) -> None:
        self._current_hour: datetime | None = None
        self._last_mid: float | None = None

    def observe(self, tick: BookTickerTick) -> dict | None:
        tick_hour = _hour_floor(tick.event_time)
        emitted: dict | None = None
        if self._current_hour is None:
            self._current_hour = tick_hour
        elif tick_hour > self._current_hour:
            assert self._last_mid is not None
            emitted = {
                "timestamp": self._current_hour,
                "price": self._last_mid,
                "volume": None,
            }
            self._current_hour = tick_hour
        self._last_mid = tick.mid
        return emitted

    def flush(self) -> dict | None:
        if self._current_hour is None or self._last_mid is None:
            return None
        return {
            "timestamp": self._current_hour,
            "price": self._last_mid,
            "volume": None,
        }


async def stream_hourly_closes(
    url: str = WS_URL,
) -> AsyncIterator[dict]:
    """Connect to Binance and yield hourly-close rows as the stream progresses."""
    sampler = HourlyResampler()
    async with websockets.connect(url) as ws:
        async for raw in ws:
            row = sampler.observe(parse_bookticker(raw))
            if row is not None:
                yield row
```

- [ ] **Step 5.5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_binance_ws.py -v`
Expected: 3 passed.

- [ ] **Step 5.6: Commit**

```bash
git add tests/fixtures/binance_bookticker_stream.jsonl src/btc_portfolio_mgr/data/binance_ws.py tests/data/test_binance_ws.py
git commit -m "feat(data): Binance bookTicker WS client with hourly resampler"
```

---

## Task 6: Unified reader (historical + live)

**Files:**
- Create: `src/btc_portfolio_mgr/data/reader.py`
- Create: `tests/data/test_reader.py`

The reader is the single interface downstream phases use to get hourly bars. It reads from the parquet file for any historical range and exposes a method for tailing the live stream. Splitting historical/live retrieval keeps tests deterministic — the live tail is tested by injecting a fake async iterator.

- [ ] **Step 6.1: Write the failing test**

`tests/data/test_reader.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from btc_portfolio_mgr.data.reader import HourlyReader
from btc_portfolio_mgr.data.storage import SCHEMA, write_parquet


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 3, tzinfo=timezone.utc),
            ],
            "price": [42000.0, 42100.0, 42050.0, 42200.0],
            "volume": [1.0, 2.0, 3.0, 4.0],
        },
        schema=SCHEMA,
    )


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    out = tmp_path / "btc_hourly.parquet"
    write_parquet(_sample_df(), out)
    return out


def test_reader_range_inclusive(parquet_file: Path) -> None:
    r = HourlyReader(parquet_file)
    df = r.range(
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
    )
    assert df["timestamp"].to_list() == [
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
    ]


def test_reader_latest(parquet_file: Path) -> None:
    r = HourlyReader(parquet_file)
    row = r.latest()
    assert row["timestamp"] == datetime(2024, 1, 1, 3, tzinfo=timezone.utc)
    assert row["price"] == 42200.0


@pytest.mark.asyncio
async def test_reader_tail_yields_live_rows(parquet_file: Path) -> None:
    async def fake_stream():
        for row in [
            {
                "timestamp": datetime(2024, 1, 1, 4, tzinfo=timezone.utc),
                "price": 42300.0,
                "volume": None,
            },
            {
                "timestamp": datetime(2024, 1, 1, 5, tzinfo=timezone.utc),
                "price": 42400.0,
                "volume": None,
            },
        ]:
            yield row

    r = HourlyReader(parquet_file)
    collected = []
    async for row in r.tail(fake_stream()):
        collected.append(row)
    assert [r["price"] for r in collected] == [42300.0, 42400.0]
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_reader.py -v`
Expected: ImportError for `btc_portfolio_mgr.data.reader`.

- [ ] **Step 6.3: Write the reader module**

`src/btc_portfolio_mgr/data/reader.py`:
```python
"""Unified historical + live hourly-bar reader."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import polars as pl

from btc_portfolio_mgr.data.storage import read_parquet


class HourlyReader:
    def __init__(self, parquet_path: Path) -> None:
        self._path = parquet_path

    def _df(self) -> pl.DataFrame:
        return read_parquet(self._path)

    def range(self, start: datetime, end: datetime) -> pl.DataFrame:
        df = self._df()
        return df.filter(
            (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
        )

    def latest(self) -> dict:
        df = self._df()
        return df.sort("timestamp").tail(1).to_dicts()[0]

    async def tail(
        self, stream: AsyncIterator[dict]
    ) -> AsyncIterator[dict]:
        """Pass through rows from a live stream (e.g. HourlyResampler output)."""
        async for row in stream:
            yield row
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_reader.py -v`
Expected: 3 passed.

- [ ] **Step 6.5: Run the full test suite**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (12 total: 3 storage + 5 coingecko + 1 backfill + 3 binance_ws + 3 reader, minus any I miscounted).

- [ ] **Step 6.6: Commit**

```bash
git add src/btc_portfolio_mgr/data/reader.py tests/data/test_reader.py
git commit -m "feat(data): unified hourly reader (historical range + live tail)"
```

---

## Task 7: End-to-end smoke test (live backfill)

This is the only step that hits the real CoinGecko API. It validates that the API key works and the chunking/dedup logic produces a usable file. **Run manually after the unit tests pass — not part of CI.**

- [ ] **Step 7.1: Populate `.env`**

```bash
cp .env.example .env
# edit .env, paste real COINGECKO_API_KEY
```

- [ ] **Step 7.2: Run the backfill**

Run:
```bash
cd /Users/michaelfelix/Documents/GitHub/btc-portfolio-mgr
.venv/bin/python scripts/backfill_historical.py
```
Expected: prints `wrote N rows to data/btc_hourly.parquet` where N is somewhere in the 60k–80k range (2018-now hourly ≈ ~70k rows).

- [ ] **Step 7.3: Spot-check the data**

Run:
```bash
.venv/bin/python -c "
import polars as pl
from btc_portfolio_mgr.data.storage import read_parquet
df = read_parquet('data/btc_hourly.parquet')
print(df.height, 'rows')
print('range:', df['timestamp'].min(), '->', df['timestamp'].max())
print(df.describe())
"
```
Expected: row count makes sense, timestamps span 2018 to today, price min/max plausible (e.g. min ~$3k, max ~$110k as of 2026).

- [ ] **Step 7.4: Commit the data file ... no, do not commit data**

The `.gitignore` excludes `data/*.parquet`. Confirm `git status` shows no staged data file before continuing.

Run: `git status`
Expected: clean working tree (no parquet files staged).

---

## Done criteria (Phase 1)

- [ ] `pytest -v` passes with all unit tests green
- [ ] `scripts/backfill_historical.py` produces a parquet file with 2018-now hourly data
- [ ] `HourlyReader(path).range(start, end)` returns the correct slice
- [ ] `stream_hourly_closes()` can connect to Binance live and emit rows (manual sanity check optional but recommended before declaring Phase 1 done)
- [ ] All commits land on `main` (or a `phase-1` branch merged to `main`)

## What's deliberately not in Phase 1

- **Daily bars 2013–2017** (CoinGecko has this; we'd want it for very-slow regime features). Skipped because the hourly-prediction model has 70k+ training rows from 2018 onward — plenty. Add later only if Phase 2 features specifically need >5-year lookbacks.
- Volume data quality from CoinGecko (we record it but don't depend on it yet)
- aggTrade stream from Binance (only needed if Phase 2 features want trade volume)
- Funding rate / open interest endpoints (Phase 3+ if needed)
- Reconciliation between CoinGecko and Binance prices (separate concern; tracked for later)
- Recorder daemon / systemd unit (Phase 7-adjacent, not needed for offline training)
