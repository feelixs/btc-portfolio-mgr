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
    assert df.height == 6  # 3 chunks × 2 rows per fixture (dedupe is downstream)
