"""Thin wrapper over binance-futures-connector UMFutures. Owns network + key loading."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from binance.um_futures import UMFutures

TESTNET_URL = "https://testnet.binancefuture.com"
MAINNET_URL = "https://fapi.binance.com"


@dataclass
class BinanceClient:
    raw: UMFutures
    network: str

    @classmethod
    def from_env(cls) -> "BinanceClient":
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        network = os.environ.get("BINANCE_NETWORK", "testnet").lower()
        if not api_key or not api_secret:
            raise RuntimeError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set in env"
            )
        if network not in ("testnet", "mainnet"):
            raise RuntimeError(f"BINANCE_NETWORK must be testnet or mainnet, got {network}")
        base_url = TESTNET_URL if network == "testnet" else MAINNET_URL
        raw = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
        return cls(raw=raw, network=network)

    def account(self) -> dict[str, Any]:
        return self.raw.account()

    def position_info(self, symbol: str) -> list[dict[str, Any]]:
        return self.raw.get_position_risk(symbol=symbol)

    def exchange_info(self) -> dict[str, Any]:
        return self.raw.exchange_info()

    def change_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self.raw.change_leverage(symbol=symbol, leverage=leverage)

    def mark_price(self, symbol: str) -> float:
        resp = self.raw.mark_price(symbol=symbol)
        return float(resp["markPrice"])

    def new_market_order(
        self, symbol: str, side: str, quantity: float, client_order_id: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }
        if client_order_id is not None:
            params["newClientOrderId"] = client_order_id
        return self.raw.new_order(**params)
