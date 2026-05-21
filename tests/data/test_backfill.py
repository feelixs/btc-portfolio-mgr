from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import polars as pl
import pytest

from btc_portfolio_mgr.data.storage import SCHEMA, read_parquet


@pytest.mark.asyncio
async def test_backfill_dedups_and_sorts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import backfill_historical as bh

    duplicated = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
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
    assert df.height == 2
    assert df["timestamp"].is_sorted()
