"""Append-only parquet log of every live cycle. Fixed schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

LOG_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "equity_before": pl.Float64,
    "equity_after": pl.Float64,
    "current_weight": pl.Float64,
    "target_weight": pl.Float64,
    "mu": pl.Float64,
    "sigma": pl.Float64,
    "action": pl.Utf8,
    "side": pl.Utf8,
    "quantity": pl.Float64,
    "delta_btc": pl.Float64,
    "notional_usdt": pl.Float64,
    "reason": pl.Utf8,
    "halted": pl.Boolean,
    "halt_reason": pl.Utf8,
}


@dataclass(frozen=True)
class LiveLogRow:
    timestamp: datetime
    equity_before: float
    equity_after: float
    current_weight: float
    target_weight: float
    mu: float
    sigma: float
    action: str
    side: str | None
    quantity: float
    delta_btc: float
    notional_usdt: float
    reason: str | None
    halted: bool
    halt_reason: str | None


def append_log_row(path: Path, row: LiveLogRow) -> None:
    new_df = pl.DataFrame([asdict(row)], schema=LOG_SCHEMA)
    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, new_df], how="vertical_relaxed")
    else:
        combined = new_df
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(path)
