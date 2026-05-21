import math
from unittest.mock import MagicMock

from btc_portfolio_mgr.live.account import SymbolInfo
from btc_portfolio_mgr.live.order_manager import (
    OrderDecision,
    compute_required_delta,
    reconcile_to_target,
    round_to_lot,
)


def test_round_to_lot_floors_toward_zero():
    assert round_to_lot(0.00499, 0.001) == 0.004
    assert round_to_lot(-0.00499, 0.001) == -0.004
    assert round_to_lot(0.0, 0.001) == 0.0


def test_compute_required_delta_long_position():
    # equity=$10, weight=1.0, price=$50_000 -> target_btc = 10/50_000 = 0.0002
    delta = compute_required_delta(
        equity_usdt=10.0,
        target_weight=1.0,
        current_btc=0.0,
        price_usdt=50_000.0,
    )
    assert abs(delta - 0.0002) < 1e-9


def test_compute_required_delta_short_to_flat():
    delta = compute_required_delta(
        equity_usdt=10.0,
        target_weight=0.0,
        current_btc=-0.005,
        price_usdt=50_000.0,
    )
    assert abs(delta - 0.005) < 1e-9


def test_reconcile_skips_when_delta_below_min_notional():
    client = MagicMock()
    symbol = SymbolInfo(symbol="BTCUSDT", lot_step=0.001, min_qty=0.001, min_notional=5.0)
    decision = reconcile_to_target(
        client=client,
        symbol_info=symbol,
        equity_usdt=10.0,
        target_weight=0.05,
        current_btc=0.0,
        price_usdt=50_000.0,
    )
    assert decision.action == "skip"
    assert decision.reason == "below_min_notional_or_lot"
    client.new_market_order.assert_not_called()


def test_reconcile_places_buy_for_positive_delta():
    client = MagicMock()
    client.new_market_order.return_value = {"orderId": 1, "status": "FILLED"}
    symbol = SymbolInfo(symbol="BTCUSDT", lot_step=0.001, min_qty=0.001, min_notional=5.0)
    decision = reconcile_to_target(
        client=client,
        symbol_info=symbol,
        equity_usdt=1000.0,
        target_weight=1.0,
        current_btc=0.0,
        price_usdt=50_000.0,
    )
    assert decision.action == "order"
    assert decision.side == "BUY"
    assert decision.quantity == 0.020  # 1000/50_000 = 0.02 BTC
    call = client.new_market_order.call_args
    assert call.kwargs["symbol"] == "BTCUSDT"
    assert call.kwargs["side"] == "BUY"
    assert call.kwargs["quantity"] == 0.02
    assert call.kwargs["client_order_id"].startswith("mtg-")


def test_reconcile_places_sell_for_negative_delta():
    client = MagicMock()
    client.new_market_order.return_value = {"orderId": 2, "status": "FILLED"}
    symbol = SymbolInfo(symbol="BTCUSDT", lot_step=0.001, min_qty=0.001, min_notional=5.0)
    decision = reconcile_to_target(
        client=client,
        symbol_info=symbol,
        equity_usdt=1000.0,
        target_weight=-1.0,
        current_btc=0.0,
        price_usdt=50_000.0,
    )
    assert decision.action == "order"
    assert decision.side == "SELL"
    assert decision.quantity == 0.020


def test_reconcile_uses_caller_provided_client_order_id():
    client = MagicMock()
    client.new_market_order.return_value = {"orderId": 3}
    symbol = SymbolInfo(symbol="BTCUSDT", lot_step=0.001, min_qty=0.001, min_notional=5.0)
    reconcile_to_target(
        client=client,
        symbol_info=symbol,
        equity_usdt=1000.0,
        target_weight=1.0,
        current_btc=0.0,
        price_usdt=50_000.0,
        client_order_id="pinned-id-001",
    )
    assert client.new_market_order.call_args.kwargs["client_order_id"] == "pinned-id-001"
