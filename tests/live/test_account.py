from unittest.mock import MagicMock

from btc_portfolio_mgr.live.account import (
    SymbolInfo,
    get_equity_usdt,
    get_position_btc,
    get_symbol_info,
)


def test_get_equity_usdt():
    client = MagicMock()
    client.account.return_value = {
        "totalWalletBalance": "10.50",
        "totalUnrealizedProfit": "0.20",
    }
    eq = get_equity_usdt(client)
    assert abs(eq - 10.70) < 1e-9


def test_get_position_btc_long():
    client = MagicMock()
    client.position_info.return_value = [{"positionAmt": "0.0123", "positionSide": "BOTH"}]
    assert abs(get_position_btc(client, "BTCUSDT") - 0.0123) < 1e-12


def test_get_position_btc_short():
    client = MagicMock()
    client.position_info.return_value = [{"positionAmt": "-0.005", "positionSide": "BOTH"}]
    assert abs(get_position_btc(client, "BTCUSDT") - (-0.005)) < 1e-12


def test_get_position_btc_empty():
    client = MagicMock()
    client.position_info.return_value = []
    assert get_position_btc(client, "BTCUSDT") == 0.0


def test_get_position_btc_raises_on_hedge_mode():
    import pytest
    client = MagicMock()
    client.position_info.return_value = [
        {"positionAmt": "0.01", "positionSide": "LONG"},
        {"positionAmt": "-0.005", "positionSide": "SHORT"},
    ]
    with pytest.raises(RuntimeError, match="hedge mode"):
        get_position_btc(client, "BTCUSDT")


def test_get_symbol_info_parses_filters():
    client = MagicMock()
    client.exchange_info.return_value = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        ]
    }
    info = get_symbol_info(client, "BTCUSDT")
    assert info == SymbolInfo(symbol="BTCUSDT", lot_step=0.001, min_qty=0.001, min_notional=5.0)
