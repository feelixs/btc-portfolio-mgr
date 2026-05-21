from __future__ import annotations

from pathlib import Path

import polars as pl

SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "price": pl.Float64(),
    "volume": pl.Float64(),
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
