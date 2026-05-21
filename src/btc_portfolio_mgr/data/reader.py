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
        ).sort("timestamp")

    def latest(self) -> dict:
        df = self._df()
        return df.sort("timestamp").tail(1).to_dicts()[0]

    async def tail(
        self, stream: AsyncIterator[dict]
    ) -> AsyncIterator[dict]:
        async for row in stream:
            yield row
