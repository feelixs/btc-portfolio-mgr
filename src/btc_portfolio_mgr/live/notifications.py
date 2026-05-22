"""Discord webhook notifications for halts and daily cycle summaries.

Reads DISCORD_WEBHOOK_URL from env. If unset, posts are silently skipped (no error).
Network failures during the post are logged to stderr but do NOT raise — trading is
primary, notifications are observability.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

HTTP_TIMEOUT_S = 10
COLOR_GREEN = 0x2ECC71
COLOR_YELLOW = 0xF1C40F
COLOR_RED = 0xE74C3C


@dataclass(frozen=True)
class SummaryContext:
    network: str
    equity_before: float
    equity_after: float
    position_btc: float
    mark_price: float
    mu: float
    sigma: float
    target_weight: float
    action: str
    order_side: str | None
    order_qty: float
    order_notional: float
    halted: bool
    halt_reason: str | None
    drawdown: float
    peak_equity: float


@dataclass(frozen=True)
class HaltContext:
    network: str
    reason: str
    equity: float
    position_btc: float
    mark_price: float


def _post(payload: dict[str, Any]) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            status = getattr(resp, "status", None)
            if status is not None and status >= 400:
                print(f"discord webhook returned {status}", file=sys.stderr, flush=True)
    except (URLError, TimeoutError) as exc:
        print(f"discord webhook failed: {exc}", file=sys.stderr, flush=True)


def _color_for_summary(ctx: SummaryContext) -> int:
    if ctx.halted:
        return COLOR_RED
    if ctx.action in ("error",):
        return COLOR_RED
    if ctx.action in ("skip", "dry_run"):
        return COLOR_YELLOW
    return COLOR_GREEN


def _summary_fields(ctx: SummaryContext) -> list[dict[str, Any]]:
    delta_pct = (
        (ctx.equity_after / ctx.equity_before - 1.0) if ctx.equity_before > 0 else 0.0
    )
    notional_held = abs(ctx.position_btc) * ctx.mark_price
    fields: list[dict[str, Any]] = [
        {
            "name": "Equity",
            "value": (
                f"${ctx.equity_after:,.2f} ({delta_pct * 100:+.2f}% vs cycle start)\n"
                f"DD: {ctx.drawdown * 100:.2f}% from peak ${ctx.peak_equity:,.2f}"
            ),
            "inline": False,
        },
        {
            "name": "Position",
            "value": (
                f"{ctx.position_btc:+.6f} BTC (${notional_held:,.2f})\n"
                f"mark ${ctx.mark_price:,.2f}"
            ),
            "inline": False,
        },
        {
            "name": "Signal",
            "value": (
                f"μ = {ctx.mu:+.4f}\n"
                f"σ = {ctx.sigma:.4f}\n"
                f"target_weight = {ctx.target_weight:+.4f}"
            ),
            "inline": False,
        },
    ]
    if ctx.action == "order":
        fields.append({
            "name": "Action",
            "value": (
                f"order {ctx.order_side} {ctx.order_qty:.6f} BTC "
                f"(${ctx.order_notional:,.2f})"
            ),
            "inline": False,
        })
    elif ctx.action == "halt":
        fields.append({
            "name": "Action",
            "value": f"halted: {(ctx.halt_reason or '')[:1000]}",
            "inline": False,
        })
    else:
        fields.append({"name": "Action", "value": ctx.action, "inline": False})
    return fields


def post_summary(ctx: SummaryContext) -> None:
    """Post a per-cycle summary embed to Discord. No-op if no webhook configured."""
    embed = {
        "title": f"📊 BTC Portfolio — daily cycle ({ctx.network})",
        "color": _color_for_summary(ctx),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "fields": _summary_fields(ctx),
    }
    _post({"embeds": [embed]})


def post_halt(ctx: HaltContext) -> None:
    """Post an urgent halt notification. Distinct red embed."""
    notional_held = abs(ctx.position_btc) * ctx.mark_price
    embed = {
        "title": f"🚨 BTC Portfolio — RISK HALT ({ctx.network})",
        "description": f"**Reason:** {ctx.reason[:1000]}",
        "color": COLOR_RED,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "fields": [
            {
                "name": "Equity",
                "value": f"${ctx.equity:,.2f}",
                "inline": True,
            },
            {
                "name": "Position",
                "value": (
                    f"{ctx.position_btc:+.6f} BTC\n"
                    f"(${notional_held:,.2f} @ ${ctx.mark_price:,.2f})"
                ),
                "inline": True,
            },
        ],
    }
    _post({"embeds": [embed]})
