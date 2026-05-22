from datetime import datetime, timedelta, timezone

import polars as pl

from btc_portfolio_mgr.live.ic_monitor import compute_live_ic


def test_compute_live_ic_returns_nan_when_too_few_rows():
    df = pl.DataFrame(
        {"timestamp": [], "mu": [], "realized_return": []},
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "mu": pl.Float64,
            "realized_return": pl.Float64,
        },
    )
    result = compute_live_ic(df)
    assert result["n"] == 0
    assert result["ic"] != result["ic"]  # NaN check


def test_compute_live_ic_perfectly_correlated():
    rows = []
    for i in range(20):
        rows.append(
            {
                "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
                "mu": float(i),
                "realized_return": float(i),
            }
        )
    df = pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "mu": pl.Float64,
            "realized_return": pl.Float64,
        },
    )
    result = compute_live_ic(df)
    assert result["n"] == 20
    assert abs(result["ic"] - 1.0) < 1e-9
