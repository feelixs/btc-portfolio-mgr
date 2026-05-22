"""One full live cycle. Cron drives this at 00:05 UTC daily."""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from btc_portfolio_mgr.live.account import (
    get_equity_usdt,
    get_position_btc,
    get_symbol_info,
    set_leverage,
)
from btc_portfolio_mgr.live.binance_client import BinanceClient
from btc_portfolio_mgr.live.inference_loop import run_inference
from btc_portfolio_mgr.live.live_logger import LiveLogRow, append_log_row
from btc_portfolio_mgr.live.notifications import (
    HaltContext,
    SummaryContext,
    post_halt,
    post_summary,
)
from btc_portfolio_mgr.live.order_manager import reconcile_to_target
from btc_portfolio_mgr.live.risk import (
    LiveState,
    evaluate_risk,
    load_state,
    save_state,
    update_state,
)
from btc_portfolio_mgr.sizing.params import SizingParams

REPO_ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = REPO_ROOT / "data" / "btc_hourly.parquet"
FEATURES_PATH = REPO_ROOT / "data" / "btc_features.parquet"
MODEL_PATH = REPO_ROOT / "models" / "btc_7d.txt"
MODEL_METADATA_PATH = REPO_ROOT / "models" / "btc_7d.metadata.json"
VOL_PATH = REPO_ROOT / "models" / "btc_vol.json"
LOG_PATH = REPO_ROOT / "data" / "live_log.parquet"
STATE_PATH = REPO_ROOT / "data" / "live_state.json"
HALT_PATH = REPO_ROOT / "HALT"

SYMBOL = "BTCUSDT"
LEVERAGE = 5


def _log_row(
    *,
    now: datetime,
    equity_before: float,
    equity_after: float,
    current_weight: float,
    target_weight: float,
    mu: float,
    sigma: float,
    action: str,
    side: str | None = None,
    quantity: float = 0.0,
    delta_btc: float = 0.0,
    notional_usdt: float = 0.0,
    reason: str | None = None,
    halted: bool = False,
    halt_reason: str | None = None,
) -> LiveLogRow:
    return LiveLogRow(
        timestamp=now,
        equity_before=equity_before,
        equity_after=equity_after,
        current_weight=current_weight,
        target_weight=target_weight,
        mu=mu,
        sigma=sigma,
        action=action,
        side=side,
        quantity=quantity,
        delta_btc=delta_btc,
        notional_usdt=notional_usdt,
        reason=reason,
        halted=halted,
        halt_reason=halt_reason,
    )


def run(dry_run: bool = False) -> int:
    now = datetime.now(tz=timezone.utc)
    print(f"[{now.isoformat()}] starting live cycle (dry_run={dry_run})")

    client = BinanceClient.from_env()
    print(f"  network: {client.network}")
    set_leverage(client, SYMBOL, LEVERAGE)
    symbol_info = get_symbol_info(client, SYMBOL)
    equity_before = get_equity_usdt(client)
    current_btc = get_position_btc(client, SYMBOL)
    mark_price = client.mark_price(SYMBOL)
    current_weight = (current_btc * mark_price) / equity_before if equity_before > 0 else 0.0
    print(
        f"  equity=${equity_before:.4f}  position={current_btc:+.6f} BTC "
        f"current_weight={current_weight:+.4f}  mark=${mark_price:,.2f}"
    )

    # Cold-start: seed peak from broker equity.
    state = load_state(STATE_PATH)
    if state.peak_equity == 0.0:
        state = LiveState(
            peak_equity=equity_before, consec_losses=0, last_equity=equity_before
        )

    inference = run_inference(
        prices_path=PRICES_PATH,
        features_path=FEATURES_PATH,
        model_path=MODEL_PATH,
        model_metadata_path=MODEL_METADATA_PATH,
        vol_path=VOL_PATH,
        sizing_params=SizingParams(),
        current_weight=current_weight,
    )
    print(
        f"  inference @ {inference.timestamp.isoformat()}: "
        f"mu={inference.mu:.6f} sigma={inference.sigma:.6f} "
        f"target_weight={inference.target_weight:+.4f}"
    )

    risk = evaluate_risk(
        state=state,
        equity=equity_before,
        latest_bar_ts=inference.timestamp,
        now=now,
        halt_file=HALT_PATH,
    )
    if not risk.allowed:
        print(f"  RISK HALT: {risk.reason}")
        append_log_row(
            LOG_PATH,
            _log_row(
                now=now,
                equity_before=equity_before,
                equity_after=equity_before,
                current_weight=current_weight,
                target_weight=inference.target_weight,
                mu=inference.mu,
                sigma=inference.sigma,
                action="halt",
                halted=True,
                halt_reason=risk.reason,
            ),
        )
        save_state(STATE_PATH, state)
        peak_equity = state.peak_equity if state.peak_equity > 0 else equity_before
        dd = (1.0 - equity_before / peak_equity) if peak_equity > 0 else 0.0
        post_halt(HaltContext(
            network=client.network,
            reason=risk.reason or "unknown",
            equity=equity_before,
            position_btc=current_btc,
            mark_price=mark_price,
        ))
        post_summary(SummaryContext(
            network=client.network,
            equity_before=equity_before,
            equity_after=equity_before,
            position_btc=current_btc,
            mark_price=mark_price,
            mu=inference.mu,
            sigma=inference.sigma,
            target_weight=inference.target_weight,
            action="halt",
            order_side=None,
            order_qty=0.0,
            order_notional=0.0,
            halted=True,
            halt_reason=risk.reason,
            drawdown=dd,
            peak_equity=peak_equity,
        ))
        return 0

    target_btc = (equity_before * inference.target_weight) / mark_price
    delta_intended = target_btc - current_btc

    if dry_run:
        print(
            f"  DRY RUN: would target {target_btc:+.6f} BTC "
            f"(delta {delta_intended:+.6f}, ${abs(delta_intended) * mark_price:.2f})"
        )
        append_log_row(
            LOG_PATH,
            _log_row(
                now=now,
                equity_before=equity_before,
                equity_after=equity_before,
                current_weight=current_weight,
                target_weight=inference.target_weight,
                mu=inference.mu,
                sigma=inference.sigma,
                action="dry_run",
                delta_btc=delta_intended,
                notional_usdt=abs(delta_intended) * mark_price,
                reason="dry_run_flag",
            ),
        )
        peak_equity = state.peak_equity if state.peak_equity > 0 else equity_before
        dd = (1.0 - equity_before / peak_equity) if peak_equity > 0 else 0.0
        post_summary(SummaryContext(
            network=client.network,
            equity_before=equity_before,
            equity_after=equity_before,
            position_btc=current_btc,
            mark_price=mark_price,
            mu=inference.mu,
            sigma=inference.sigma,
            target_weight=inference.target_weight,
            action="dry_run",
            order_side=None,
            order_qty=0.0,
            order_notional=abs(delta_intended) * mark_price,
            halted=False,
            halt_reason=None,
            drawdown=dd,
            peak_equity=peak_equity,
        ))
        return 0

    # Live: generate id BEFORE the network call, log pre-order row, then submit.
    client_order_id = f"mtg-{uuid.uuid4().hex[:24]}"
    append_log_row(
        LOG_PATH,
        _log_row(
            now=now,
            equity_before=equity_before,
            equity_after=equity_before,
            current_weight=current_weight,
            target_weight=inference.target_weight,
            mu=inference.mu,
            sigma=inference.sigma,
            action="pre_order",
            delta_btc=delta_intended,
            notional_usdt=abs(delta_intended) * mark_price,
            reason=client_order_id,
        ),
    )

    try:
        decision = reconcile_to_target(
            client=client,
            symbol_info=symbol_info,
            equity_usdt=equity_before,
            target_weight=inference.target_weight,
            current_btc=current_btc,
            price_usdt=mark_price,
            client_order_id=client_order_id,
        )
    except Exception as exc:
        append_log_row(
            LOG_PATH,
            _log_row(
                now=datetime.now(tz=timezone.utc),
                equity_before=equity_before,
                equity_after=equity_before,  # unknown; conservative
                current_weight=current_weight,
                target_weight=inference.target_weight,
                mu=inference.mu,
                sigma=inference.sigma,
                action="error",
                delta_btc=delta_intended,
                notional_usdt=abs(delta_intended) * mark_price,
                reason=f"{client_order_id}: {exc}",
            ),
        )
        raise
    print(f"  ORDER: {decision.action} side={decision.side} qty={decision.quantity:.6f}")

    # Let Binance settle realized PnL into wallet balance before refetching equity.
    time.sleep(2.0)
    # Re-fetch equity after the order to capture realized PnL + fees.
    equity_after = get_equity_usdt(client)
    new_state = update_state(state, equity=equity_after)
    save_state(STATE_PATH, new_state)

    append_log_row(
        LOG_PATH,
        _log_row(
            now=now,
            equity_before=equity_before,
            equity_after=equity_after,
            current_weight=current_weight,
            target_weight=inference.target_weight,
            mu=inference.mu,
            sigma=inference.sigma,
            action=decision.action,
            side=decision.side,
            quantity=decision.quantity,
            delta_btc=decision.delta_btc,
            notional_usdt=decision.notional_usdt,
            reason=decision.reason,
        ),
    )
    peak_equity = new_state.peak_equity if new_state.peak_equity > 0 else equity_after
    dd = (1.0 - equity_after / peak_equity) if peak_equity > 0 else 0.0
    # Reflect actual filled position post-order
    position_after = current_btc + decision.delta_btc
    post_summary(SummaryContext(
        network=client.network,
        equity_before=equity_before,
        equity_after=equity_after,
        position_btc=position_after,
        mark_price=mark_price,
        mu=inference.mu,
        sigma=inference.sigma,
        target_weight=inference.target_weight,
        action=decision.action,
        order_side=decision.side,
        order_qty=decision.quantity,
        order_notional=decision.notional_usdt,
        halted=False,
        halt_reason=None,
        drawdown=dd,
        peak_equity=peak_equity,
    ))
    print(f"[{datetime.now(tz=timezone.utc).isoformat()}] cycle complete; equity ${equity_after:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
