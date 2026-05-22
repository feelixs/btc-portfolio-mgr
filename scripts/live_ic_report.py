"""Compute IC over the live log: for each cycle with mu, join realized 7d-forward return."""
from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import polars as pl

from btc_portfolio_mgr.live.ic_monitor import compute_live_ic

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "data" / "live_log.parquet"
PRICES_PATH = REPO_ROOT / "data" / "btc_hourly.parquet"
HORIZON_DAYS = 7


def main() -> int:
    if not LOG_PATH.exists():
        print(f"no live log at {LOG_PATH}")
        return 1
    log = pl.read_parquet(LOG_PATH)
    prices = pl.read_parquet(PRICES_PATH).sort("timestamp")  # ensure time-sorted
    max_price_ts = prices["timestamp"].max()
    rows = []
    for r in log.iter_rows(named=True):
        if r["halted"] or r["mu"] is None:
            continue
        ts = r["timestamp"]
        ts_future = ts + timedelta(days=HORIZON_DAYS)
        if ts_future > max_price_ts:
            # Future window not yet realized — skip rather than use stale last bar.
            continue
        p_now = prices.filter(pl.col("timestamp") <= ts).tail(1)
        p_fut = prices.filter(pl.col("timestamp") <= ts_future).tail(1)
        if p_now.height == 0 or p_fut.height == 0:
            continue
        realized = math.log(float(p_fut["price"][0]) / float(p_now["price"][0]))
        rows.append({"timestamp": ts, "mu": r["mu"], "realized_return": realized})
    if not rows:
        print("no eligible log rows with future-window data")
        return 0
    df = pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "mu": pl.Float64,
            "realized_return": pl.Float64,
        },
    )
    result = compute_live_ic(df)
    print(f"live IC over {result['n']} samples: {result['ic']:+.4f}")
    print("compare to CV folds 1-5 in models/btc_7d.metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
