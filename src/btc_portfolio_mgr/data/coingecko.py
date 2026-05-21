from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import polars as pl

from btc_portfolio_mgr.data.storage import SCHEMA

PRO_BASE = "https://pro-api.coingecko.com/api/v3"


def parse_market_chart(payload: dict) -> pl.DataFrame:
    """Convert /market_chart or /market_chart/range JSON to canonical schema.

    Timestamps are floored to the hour. CoinGecko returns samples at "hourly
    granularity" but the actual timestamps drift by seconds-to-minutes from the
    hour mark; flooring aligns them to the canonical hourly grid that the rest
    of the pipeline (reindex_to_hourly, feature lookbacks, GARCH filter)
    assumes.
    """
    prices = payload["prices"]
    volumes_by_ts = {ts: v for ts, v in payload["total_volumes"]}
    rows = []
    for ts_ms, price in prices:
        raw_ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        floored_ts = raw_ts.replace(minute=0, second=0, microsecond=0)
        rows.append(
            {
                "timestamp": floored_ts,
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
    """Async CoinGecko Pro client. Hourly data when chunk <= 90 days."""

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
