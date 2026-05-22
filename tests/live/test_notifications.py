import json
from unittest.mock import MagicMock, patch

from btc_portfolio_mgr.live.notifications import (
    HaltContext,
    SummaryContext,
    post_halt,
    post_summary,
)


def test_post_summary_skips_when_url_unset(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with patch("btc_portfolio_mgr.live.notifications.urlopen") as mock:
        post_summary(SummaryContext(
            network="testnet",
            equity_before=5000.0,
            equity_after=4999.23,
            position_btc=0.0183,
            mark_price=77200.0,
            mu=0.01,
            sigma=0.09,
            target_weight=0.28,
            action="order",
            order_side="BUY",
            order_qty=0.0183,
            order_notional=1414.0,
            halted=False,
            halt_reason=None,
            drawdown=0.0002,
            peak_equity=5000.0,
        ))
        mock.assert_not_called()


def test_post_summary_sends_payload(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        ctx = MagicMock()
        ctx.__enter__ = lambda self: MagicMock(status=204)
        ctx.__exit__ = lambda self, *a: False
        return ctx

    with patch("btc_portfolio_mgr.live.notifications.urlopen", side_effect=fake_urlopen):
        post_summary(SummaryContext(
            network="testnet",
            equity_before=5000.0,
            equity_after=4999.23,
            position_btc=0.0183,
            mark_price=77200.0,
            mu=0.01115,
            sigma=0.0888,
            target_weight=0.2830,
            action="order",
            order_side="BUY",
            order_qty=0.0183,
            order_notional=1414.0,
            halted=False,
            halt_reason=None,
            drawdown=0.000154,
            peak_equity=5000.0,
        ))
    assert captured["url"] == "https://discord.com/api/webhooks/x/y"
    embed = captured["body"]["embeds"][0]
    assert "testnet" in embed["title"].lower() or "testnet" in str(embed.get("fields", [])).lower()
    fields_text = json.dumps(embed["fields"])
    assert "0.2830" in fields_text or "0.283" in fields_text
    assert "BUY" in fields_text


def test_post_halt_uses_red_color_and_urgent_marker(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode())
        ctx = MagicMock()
        ctx.__enter__ = lambda self: MagicMock(status=204)
        ctx.__exit__ = lambda self, *a: False
        return ctx

    with patch("btc_portfolio_mgr.live.notifications.urlopen", side_effect=fake_urlopen):
        post_halt(HaltContext(
            network="mainnet",
            reason="drawdown 0.180 exceeds 0.15",
            equity=820.0,
            position_btc=0.005,
            mark_price=70000.0,
        ))
    embed = captured["body"]["embeds"][0]
    # Red-ish color (any nonzero int — at minimum the implementer should NOT use green/0x00FF00)
    assert embed["color"] != 0x00FF00
    title = embed["title"]
    assert "halt" in title.lower() or "🚨" in title or "⛔" in title
    assert "drawdown" in json.dumps(embed).lower()


def test_post_halt_truncates_long_reason(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode())
        ctx = MagicMock()
        ctx.__enter__ = lambda self: MagicMock(status=204)
        ctx.__exit__ = lambda self, *a: False
        return ctx

    long_reason = "X" * 5000
    with patch("btc_portfolio_mgr.live.notifications.urlopen", side_effect=fake_urlopen):
        post_halt(HaltContext(
            network="testnet",
            reason=long_reason,
            equity=1000.0,
            position_btc=0.0,
            mark_price=70000.0,
        ))
    desc = captured["body"]["embeds"][0]["description"]
    assert len(desc) <= 1100  # bounded
    assert "XXX" in desc  # but content preserved


def test_post_summary_swallows_network_errors(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
    from urllib.error import URLError
    with patch(
        "btc_portfolio_mgr.live.notifications.urlopen", side_effect=URLError("boom")
    ):
        # Must NOT raise
        post_summary(SummaryContext(
            network="testnet",
            equity_before=5000.0,
            equity_after=4999.23,
            position_btc=0.0,
            mark_price=77200.0,
            mu=0.0,
            sigma=0.09,
            target_weight=0.0,
            action="skip",
            order_side=None,
            order_qty=0.0,
            order_notional=0.0,
            halted=False,
            halt_reason=None,
            drawdown=0.0,
            peak_equity=5000.0,
        ))
