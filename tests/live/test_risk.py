from datetime import datetime, timedelta, timezone
from pathlib import Path

from btc_portfolio_mgr.live.risk import (
    LiveState,
    RiskCheck,
    evaluate_risk,
    load_state,
    save_state,
    update_state,
)

NOW = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)


def test_risk_green_when_all_clear():
    state = LiveState(peak_equity=100.0, consec_losses=0, last_equity=98.0)
    check = evaluate_risk(
        state=state,
        equity=98.0,
        latest_bar_ts=NOW - timedelta(hours=1),
        now=NOW,
        halt_file=Path("/tmp/does-not-exist-halt"),
    )
    assert check.allowed is True
    assert check.reason is None


def test_risk_halts_on_drawdown():
    state = LiveState(peak_equity=100.0, consec_losses=0, last_equity=84.0)
    check = evaluate_risk(
        state=state,
        equity=84.0,  # 16% DD
        latest_bar_ts=NOW - timedelta(hours=1),
        now=NOW,
        halt_file=Path("/tmp/does-not-exist-halt"),
    )
    assert check.allowed is False
    assert "drawdown" in check.reason


def test_risk_halts_on_stale_data():
    state = LiveState(peak_equity=100.0, consec_losses=0, last_equity=100.0)
    check = evaluate_risk(
        state=state,
        equity=100.0,
        latest_bar_ts=NOW - timedelta(hours=7),
        now=NOW,
        halt_file=Path("/tmp/does-not-exist-halt"),
    )
    assert check.allowed is False
    assert "stale" in check.reason


def test_risk_halts_on_consec_losses():
    state = LiveState(peak_equity=100.0, consec_losses=5, last_equity=100.0)
    check = evaluate_risk(
        state=state,
        equity=100.0,
        latest_bar_ts=NOW - timedelta(hours=1),
        now=NOW,
        halt_file=Path("/tmp/does-not-exist-halt"),
    )
    assert check.allowed is False
    assert "consecutive" in check.reason


def test_risk_halts_on_halt_file(tmp_path):
    halt = tmp_path / "HALT"
    halt.write_text("manual stop")
    state = LiveState(peak_equity=100.0, consec_losses=0, last_equity=100.0)
    check = evaluate_risk(
        state=state,
        equity=100.0,
        latest_bar_ts=NOW - timedelta(hours=1),
        now=NOW,
        halt_file=halt,
    )
    assert check.allowed is False
    assert "halt file" in check.reason.lower()


def test_update_state_increments_losses_on_drop():
    state = LiveState(peak_equity=100.0, consec_losses=0, last_equity=100.0)
    new_state = update_state(state, equity=98.0)
    assert new_state.consec_losses == 1
    assert new_state.peak_equity == 100.0
    assert new_state.last_equity == 98.0


def test_update_state_resets_losses_on_gain():
    state = LiveState(peak_equity=100.0, consec_losses=3, last_equity=95.0)
    new_state = update_state(state, equity=96.0)
    assert new_state.consec_losses == 0
    assert new_state.last_equity == 96.0


def test_update_state_raises_peak():
    state = LiveState(peak_equity=100.0, consec_losses=0, last_equity=100.0)
    new_state = update_state(state, equity=105.0)
    assert new_state.peak_equity == 105.0


def test_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    save_state(p, LiveState(peak_equity=123.45, consec_losses=2, last_equity=120.0))
    loaded = load_state(p)
    assert loaded == LiveState(peak_equity=123.45, consec_losses=2, last_equity=120.0)


def test_load_state_returns_zero_state_when_missing(tmp_path):
    loaded = load_state(tmp_path / "missing.json")
    assert loaded == LiveState(peak_equity=0.0, consec_losses=0, last_equity=0.0)


def test_evaluate_risk_rejects_naive_datetimes():
    import pytest
    naive_now = datetime(2026, 5, 21, 12, 0)  # no tzinfo
    state = LiveState(peak_equity=100.0, consec_losses=0, last_equity=100.0)
    with pytest.raises(ValueError, match="tz-aware"):
        evaluate_risk(
            state=state,
            equity=100.0,
            latest_bar_ts=NOW - timedelta(hours=1),
            now=naive_now,
            halt_file=Path("/tmp/does-not-exist-halt"),
        )
    with pytest.raises(ValueError, match="tz-aware"):
        evaluate_risk(
            state=state,
            equity=100.0,
            latest_bar_ts=naive_now,
            now=NOW,
            halt_file=Path("/tmp/does-not-exist-halt"),
        )
