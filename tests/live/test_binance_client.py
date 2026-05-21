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
