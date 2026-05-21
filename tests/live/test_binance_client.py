import os
from unittest.mock import MagicMock, patch

from btc_portfolio_mgr.live.binance_client import BinanceClient


def test_client_selects_testnet_url(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_NETWORK", "testnet")
    with patch("btc_portfolio_mgr.live.binance_client.UMFutures") as mock_umf:
        BinanceClient.from_env()
        kwargs = mock_umf.call_args.kwargs
        assert kwargs["base_url"] == "https://testnet.binancefuture.com"


def test_client_selects_mainnet_url(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_NETWORK", "mainnet")
    with patch("btc_portfolio_mgr.live.binance_client.UMFutures") as mock_umf:
        BinanceClient.from_env()
        kwargs = mock_umf.call_args.kwargs
        assert kwargs["base_url"] == "https://fapi.binance.com"


def test_client_raises_on_missing_keys(monkeypatch):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="BINANCE_API_KEY"):
        BinanceClient.from_env()


def test_new_market_order_forwards_client_order_id(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_NETWORK", "testnet")
    with patch("btc_portfolio_mgr.live.binance_client.UMFutures") as mock_umf:
        client = BinanceClient.from_env()
        client.raw.new_order = MagicMock(return_value={"orderId": 1})
        client.new_market_order("BTCUSDT", "BUY", 0.001, client_order_id="abc123")
        client.raw.new_order.assert_called_once_with(
            symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.001, newClientOrderId="abc123"
        )


def test_new_market_order_omits_client_order_id_when_none(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_NETWORK", "testnet")
    with patch("btc_portfolio_mgr.live.binance_client.UMFutures") as mock_umf:
        client = BinanceClient.from_env()
        client.raw.new_order = MagicMock(return_value={"orderId": 1})
        client.new_market_order("BTCUSDT", "BUY", 0.001)
        kwargs = client.raw.new_order.call_args.kwargs
        assert "newClientOrderId" not in kwargs
