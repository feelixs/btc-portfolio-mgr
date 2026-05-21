from datetime import datetime, timezone

import polars as pl

from btc_portfolio_mgr.live.live_logger import LiveLogRow, append_log_row


def _row(ts):
    return LiveLogRow(
        timestamp=ts,
        equity_before=10.0,
        equity_after=10.05,
        current_weight=0.0,
        target_weight=0.5,
        mu=0.01,
        sigma=0.02,
        action="order",
        side="BUY",
        quantity=0.0001,
        delta_btc=0.0001,
        notional_usdt=5.0,
        reason=None,
        halted=False,
        halt_reason=None,
    )


def test_append_creates_file(tmp_path):
    p = tmp_path / "live_log.parquet"
    append_log_row(p, _row(datetime(2026, 5, 21, 0, 0, tzinfo=timezone.utc)))
    df = pl.read_parquet(p)
    assert df.height == 1
    assert df["target_weight"][0] == 0.5


def test_append_concats(tmp_path):
    p = tmp_path / "live_log.parquet"
    append_log_row(p, _row(datetime(2026, 5, 21, 0, 0, tzinfo=timezone.utc)))
    append_log_row(p, _row(datetime(2026, 5, 22, 0, 0, tzinfo=timezone.utc)))
    df = pl.read_parquet(p)
    assert df.height == 2
    assert df["timestamp"][1].date().isoformat() == "2026-05-22"


def test_append_preserves_nulls(tmp_path):
    p = tmp_path / "live_log.parquet"
    row = _row(datetime(2026, 5, 21, 0, 0, tzinfo=timezone.utc))
    append_log_row(p, row)
    df = pl.read_parquet(p)
    assert df["reason"][0] is None
    assert df["halt_reason"][0] is None


def test_append_survives_schema_drift(tmp_path):
    """An old log missing a column should still be readable + appendable."""
    p = tmp_path / "live_log.parquet"
    # Simulate an old log with fewer columns.
    old = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc)],
            "equity_before": [10.0],
            "equity_after": [10.0],
            "current_weight": [0.0],
            "target_weight": [0.0],
            "mu": [0.0],
            "sigma": [0.0],
            "action": ["skip"],
            "side": [None],
            "quantity": [0.0],
            "delta_btc": [0.0],
            "notional_usdt": [0.0],
            "reason": [None],
            "halted": [False],
            # halt_reason intentionally omitted
        }
    )
    old.write_parquet(p)
    append_log_row(p, _row(datetime(2026, 5, 21, 0, 0, tzinfo=timezone.utc)))
    df = pl.read_parquet(p)
    assert df.height == 2
    assert "halt_reason" in df.columns


def test_append_uses_atomic_replace(tmp_path):
    """The temp file should not linger after a successful append."""
    p = tmp_path / "live_log.parquet"
    append_log_row(p, _row(datetime(2026, 5, 21, 0, 0, tzinfo=timezone.utc)))
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists()
