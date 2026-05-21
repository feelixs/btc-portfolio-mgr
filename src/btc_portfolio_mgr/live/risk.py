"""Kill-switch evaluation: cumulative DD, stale data, consec losses, HALT file."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

MAX_DRAWDOWN = 0.15
MAX_STALENESS = timedelta(hours=6)
MAX_CONSEC_LOSSES = 5


@dataclass(frozen=True)
class LiveState:
    peak_equity: float
    consec_losses: int
    last_equity: float


@dataclass(frozen=True)
class RiskCheck:
    allowed: bool
    reason: str | None


def evaluate_risk(
    state: LiveState,
    equity: float,
    latest_bar_ts: datetime,
    now: datetime,
    halt_file: Path,
) -> RiskCheck:
    if latest_bar_ts.tzinfo is None or now.tzinfo is None:
        raise ValueError(
            f"evaluate_risk requires tz-aware datetimes; got "
            f"latest_bar_ts={latest_bar_ts}, now={now}"
        )
    if halt_file.exists():
        return RiskCheck(allowed=False, reason=f"halt file present at {halt_file}")
    if state.peak_equity > 0:
        dd = 1.0 - equity / state.peak_equity
        if dd > MAX_DRAWDOWN:
            return RiskCheck(
                allowed=False,
                reason=f"drawdown {dd:.3f} exceeds {MAX_DRAWDOWN:.2f}",
            )
    staleness = now - latest_bar_ts
    if staleness > MAX_STALENESS:
        return RiskCheck(
            allowed=False,
            reason=f"data stale by {staleness} (limit {MAX_STALENESS})",
        )
    if state.consec_losses >= MAX_CONSEC_LOSSES:
        return RiskCheck(
            allowed=False,
            reason=f"{state.consec_losses} consecutive losing days (limit {MAX_CONSEC_LOSSES})",
        )
    return RiskCheck(allowed=True, reason=None)


def update_state(state: LiveState, equity: float) -> LiveState:
    peak = max(state.peak_equity, equity)
    if equity < state.last_equity:
        consec = state.consec_losses + 1
    elif equity > state.last_equity:
        consec = 0
    else:
        consec = state.consec_losses
    return LiveState(peak_equity=peak, consec_losses=consec, last_equity=equity)


def save_state(path: Path, state: LiveState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(state), f)


def load_state(path: Path) -> LiveState:
    if not path.exists():
        return LiveState(peak_equity=0.0, consec_losses=0, last_equity=0.0)
    with path.open() as f:
        d = json.load(f)
    return LiveState(
        peak_equity=float(d["peak_equity"]),
        consec_losses=int(d["consec_losses"]),
        last_equity=float(d["last_equity"]),
    )
