"""EMERGENCY: market-out the current BTCUSDT perp position to zero."""
from __future__ import annotations

import sys
import uuid

from btc_portfolio_mgr.live.account import get_position_btc
from btc_portfolio_mgr.live.binance_client import BinanceClient

SYMBOL = "BTCUSDT"


def main() -> int:
    client = BinanceClient.from_env()
    print(f"network: {client.network}")
    pos = get_position_btc(client, SYMBOL)
    if abs(pos) < 1e-12:
        print("position already flat")
        return 0
    side = "SELL" if pos > 0 else "BUY"
    qty = abs(pos)
    client_order_id = f"mtg-flat-{uuid.uuid4().hex[:18]}"
    print(f"flattening: {side} {qty:.6f} {SYMBOL} (id={client_order_id})")
    resp = client.new_market_order(
        symbol=SYMBOL, side=side, quantity=qty, client_order_id=client_order_id
    )
    print(f"response: {resp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
