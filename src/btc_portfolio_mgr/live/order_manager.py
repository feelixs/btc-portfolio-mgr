"""target weight + current position + price → market order (or skip)."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any

from btc_portfolio_mgr.live.account import SymbolInfo


@dataclass(frozen=True)
class OrderDecision:
    action: str  # "order" or "skip"
    side: str | None = None
    quantity: float = 0.0
    target_btc: float = 0.0
    delta_btc: float = 0.0
    notional_usdt: float = 0.0
    reason: str | None = None
    response: dict[str, Any] | None = None


def round_to_lot(qty: float, lot_step: float) -> float:
    """Floor |qty| to the nearest multiple of lot_step, preserving sign."""
    if lot_step <= 0:
        return qty
    sign = 1.0 if qty >= 0 else -1.0
    floored = math.floor(abs(qty) / lot_step) * lot_step
    return sign * floored


def compute_required_delta(
    equity_usdt: float,
    target_weight: float,
    current_btc: float,
    price_usdt: float,
) -> float:
    """target_btc - current_btc. target_notional = equity * weight (interpretation A)."""
    if price_usdt <= 0:
        raise ValueError("price_usdt must be positive")
    target_btc = (equity_usdt * target_weight) / price_usdt
    return target_btc - current_btc


def reconcile_to_target(
    client: Any,
    symbol_info: SymbolInfo,
    equity_usdt: float,
    target_weight: float,
    current_btc: float,
    price_usdt: float,
    client_order_id: str | None = None,
) -> OrderDecision:
    """Compute required delta, round to lot, check min notional, place market order.

    `client_order_id` forwards to Binance's `newClientOrderId` for idempotent retries.
    If None, a fresh UUID4-derived id is generated.
    """
    if client_order_id is None:
        client_order_id = f"mtg-{uuid.uuid4().hex[:24]}"
    delta = compute_required_delta(
        equity_usdt=equity_usdt,
        target_weight=target_weight,
        current_btc=current_btc,
        price_usdt=price_usdt,
    )
    target_btc = current_btc + delta
    delta_rounded = round_to_lot(delta, symbol_info.lot_step)
    notional = abs(delta_rounded) * price_usdt
    if abs(delta_rounded) < symbol_info.min_qty or notional < symbol_info.min_notional:
        return OrderDecision(
            action="skip",
            target_btc=target_btc,
            delta_btc=delta_rounded,
            notional_usdt=notional,
            reason="below_min_notional_or_lot",
        )
    side = "BUY" if delta_rounded > 0 else "SELL"
    quantity = abs(delta_rounded)
    response = client.new_market_order(
        symbol=symbol_info.symbol,
        side=side,
        quantity=quantity,
        client_order_id=client_order_id,
    )
    return OrderDecision(
        action="order",
        side=side,
        quantity=quantity,
        target_btc=target_btc,
        delta_btc=delta_rounded,
        notional_usdt=notional,
        response=response,
    )
