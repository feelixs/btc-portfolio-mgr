"""Binance bookTicker WebSocket client + hourly close resampler."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, cast

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

    observe(tick) returns a row dict (matching SCHEMA) when the hour rolls over,
    else None. flush() emits the currently-open hour as a partial close.
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
            if self._last_mid is None:
                raise RuntimeError("invariant violated: hour rolled before any tick observed")
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
    async with websockets.connect(url) as ws:  # type: ignore[attr-defined]
        async for raw in ws:
            row = sampler.observe(parse_bookticker(cast(str, raw)))
            if row is not None:
                yield row
