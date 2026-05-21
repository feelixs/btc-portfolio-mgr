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
