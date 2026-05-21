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
