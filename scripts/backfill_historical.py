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

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "btc_hourly.parquet"
# Skip Jan-Mar 2018 — CoinGecko Pro returns daily (not hourly) data for that
# window, which propagates through Phase 2's 90-day-lookback nulls and destroys
# the usable training set. 2019+ is ~99% hourly coverage.
DEFAULT_START = datetime(2019, 1, 1, tzinfo=timezone.utc)


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
