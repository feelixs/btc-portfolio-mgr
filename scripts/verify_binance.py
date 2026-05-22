"""Read-only Binance connectivity + account snapshot. Places no orders."""
from __future__ import annotations

import sys

from btc_portfolio_mgr.live.account import (
    get_equity_usdt,
    get_position_btc,
    get_symbol_info,
)
from btc_portfolio_mgr.live.binance_client import BinanceClient

SYMBOL = "BTCUSDT"


def main() -> int:
    client = BinanceClient.from_env()
    print(f"network: {client.network}")
    equity = get_equity_usdt(client)
    print(f"equity USDT: ${equity:.4f}")
    pos = get_position_btc(client, SYMBOL)
    print(f"position {SYMBOL}: {pos:+.6f} BTC")
    info = get_symbol_info(client, SYMBOL)
    print(
        f"{SYMBOL} filters: lot_step={info.lot_step} "
        f"min_qty={info.min_qty} min_notional=${info.min_notional}"
    )
    mark = client.mark_price(SYMBOL)
    print(f"mark price: ${mark:,.2f}")
    print("verification complete (no orders placed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
