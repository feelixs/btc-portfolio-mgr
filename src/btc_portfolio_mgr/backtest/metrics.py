"""Backtest aggregate metrics: Sharpe, Sortino, max DD, Calmar, turnover, hit rate."""
from __future__ import annotations

import math

import numpy as np
import polars as pl

BARS_PER_YEAR = 365  # crypto trades every day


def compute_backtest_metrics(bars: pl.DataFrame) -> dict[str, float]:
    """Aggregate a bar-by-bar equity DataFrame into summary metrics.

    Expects columns: timestamp, weight, net_pnl, equity. Returns a dict with
    keys: total_return, annualized_return, annualized_vol, sharpe, sortino,
    max_drawdown, calmar, turnover_annualized, time_in_market, hit_rate.
    """
    if bars.height == 0:
        return {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "turnover_annualized": float("nan"),
            "time_in_market": float("nan"),
            "hit_rate": float("nan"),
        }

    equity = bars["equity"].to_numpy()
    weights = bars["weight"].to_numpy()
    n = len(equity)
    initial = float(equity[0])
    daily_returns = np.diff(equity) / equity[:-1] if n > 1 else np.array([])

    total_return = float(equity[-1] / initial - 1.0) if initial > 0 else float("nan")
    annualized_return = (
        float((equity[-1] / initial) ** (BARS_PER_YEAR / n) - 1.0)
        if initial > 0 and n > 0
        else float("nan")
    )
    annualized_vol = (
        float(daily_returns.std(ddof=1) * math.sqrt(BARS_PER_YEAR))
        if daily_returns.size > 1
        else 0.0
    )
    sharpe = (
        float(annualized_return / annualized_vol)
        if annualized_vol > 0
        else float("nan") if daily_returns.size <= 1 else 0.0
    )

    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = (
        float(downside_returns.std(ddof=1) * math.sqrt(BARS_PER_YEAR))
        if downside_returns.size > 1
        else 0.0
    )
    sortino = (
        float(annualized_return / downside_vol)
        if downside_vol > 0
        else float("nan")
    )

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_drawdown = float(drawdowns.max()) if drawdowns.size > 0 else 0.0
    calmar = (
        float(annualized_return / max_drawdown) if max_drawdown > 0 else float("nan")
    )

    # Turnover: sum of |Δweight| per bar, annualized.
    weight_changes = np.diff(weights)
    sum_abs_delta = float(np.abs(weight_changes).sum())
    turnover_annualized = sum_abs_delta * BARS_PER_YEAR / n if n > 0 else 0.0

    # Time in market: fraction of bars with |weight| > 0.
    time_in_market = float((np.abs(weights) > 0).mean())

    # Hit rate: fraction of bars with positive net_pnl.
    net_pnl = bars["net_pnl"].to_numpy()
    hit_rate = float((net_pnl > 0).mean()) if net_pnl.size > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "turnover_annualized": turnover_annualized,
        "time_in_market": time_in_market,
        "hit_rate": hit_rate,
    }
