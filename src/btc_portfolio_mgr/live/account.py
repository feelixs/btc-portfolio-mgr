"""Read-only views over the Binance perp account. Returns typed structs, not raw dicts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    lot_step: float
    min_qty: float
    min_notional: float


def get_equity_usdt(client: Any) -> float:
    """Total equity (wallet balance + unrealized PnL) in USDT."""
    acct = client.account()
    return float(acct["totalWalletBalance"]) + float(acct["totalUnrealizedProfit"])


def get_position_btc(client: Any, symbol: str) -> float:
    """Signed position size in BTC (one-way mode). Raises if account is in hedge mode."""
    rows = client.position_info(symbol=symbol)
    if not rows:
        return 0.0
    both_rows = [r for r in rows if r.get("positionSide", "BOTH") == "BOTH"]
    if not both_rows:
        raise RuntimeError(
            f"no positionSide=BOTH row for {symbol}; account is likely in hedge mode "
            f"(we require one-way mode)"
        )
    if len(both_rows) > 1:
        raise RuntimeError(f"multiple BOTH rows for {symbol}: {both_rows}")
    return float(both_rows[0]["positionAmt"])


def get_symbol_info(client: Any, symbol: str) -> SymbolInfo:
    """Look up lot step, min qty, and min notional from exchangeInfo filters."""
    info = client.exchange_info()
    for s in info["symbols"]:
        if s["symbol"] != symbol:
            continue
        lot_step = 0.0
        min_qty = 0.0
        min_notional = 0.0
        for f in s["filters"]:
            if f["filterType"] == "LOT_SIZE":
                lot_step = float(f["stepSize"])
                min_qty = float(f["minQty"])
            elif f["filterType"] == "MIN_NOTIONAL":
                min_notional = float(f["notional"])
        return SymbolInfo(
            symbol=symbol,
            lot_step=lot_step,
            min_qty=min_qty,
            min_notional=min_notional,
        )
    raise RuntimeError(f"symbol {symbol} not found in exchangeInfo")


def set_leverage(client: Any, symbol: str, leverage: int) -> None:
    """Set per-symbol leverage. Idempotent: Binance accepts re-setting the same value."""
    client.change_leverage(symbol=symbol, leverage=leverage)
