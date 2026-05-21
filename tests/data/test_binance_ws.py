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
    assert len(emitted) == 1
    df = pl.DataFrame(emitted, schema=SCHEMA)
    assert df.schema == SCHEMA
    assert df["timestamp"].to_list() == [
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    ]
    assert df["price"].to_list() == [42011.0]


def test_resampler_flush_emits_partial_hour(binance_bookticker_lines: list[str]) -> None:
    sampler = HourlyResampler()
    for line in binance_bookticker_lines:
        sampler.observe(parse_bookticker(line))
    final = sampler.flush()
    assert final is not None
    assert final["timestamp"] == datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
    assert final["price"] == 42101.0
