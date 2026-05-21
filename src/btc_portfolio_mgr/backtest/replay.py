"""Replay sizing decisions on cached forecasts + price series with transaction costs."""
from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.sizing.params import SizingParams
from btc_portfolio_mgr.sizing.sizer import target_weight


def replay_sizing(
    forecasts: pl.DataFrame,
    prices: pl.DataFrame,
    params: SizingParams,
    cost_bps: float = 10.0,
    initial_capital: float = 1.0,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Replay daily sizing decisions on cached (μ̂, σ̂) forecasts.

    For each bar t in `forecasts`:
      - Compute target weight from mu_t, sigma_t, current_weight, params.
      - Realized log return = log(p_{t+1d} / p_t), looked up from the prices
        DataFrame. If no future price exists (last bar), realized_return = 0.
      - PnL accumulates as: equity_{t+1} = equity_t * (1 + weight_t * (e^r - 1)).
      - Costs charged at the rebalance bar: cost = (cost_bps/10000) * |Δweight| * equity.

    Returns (bars DataFrame, summary dict).
    """
    if forecasts.height == 0:
        empty_bars = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "mu": pl.Float64(),
                "sigma": pl.Float64(),
                "target_weight": pl.Float64(),
                "weight": pl.Float64(),
                "realized_return": pl.Float64(),
                "gross_pnl": pl.Float64(),
                "costs": pl.Float64(),
                "net_pnl": pl.Float64(),
                "equity": pl.Float64(),
            }
        )
        return empty_bars, {"total_return": 0.0, "n_bars": 0}

    # Build a fast lookup from timestamp → price.
    price_by_ts = dict(zip(prices["timestamp"].to_list(), prices["price"].to_list()))

    rows: list[dict[str, float]] = []
    current_weight = 0.0
    equity = initial_capital
    forecast_ts = forecasts["timestamp"].to_list()
    forecast_mu = forecasts["mu"].to_list()
    forecast_sigma = forecasts["sigma"].to_list()
    cost_rate = cost_bps / 10000.0

    for i, ts in enumerate(forecast_ts):
        mu = forecast_mu[i]
        sigma = forecast_sigma[i]
        tgt = target_weight(
            mu=mu, sigma=sigma, current_weight=current_weight, params=params
        )
        cost = cost_rate * abs(tgt - current_weight) * equity
        equity_after_cost = equity - cost
        # Realized log return from THIS bar (t) to the NEXT eval (t + 1 day).
        next_ts = forecast_ts[i + 1] if i + 1 < len(forecast_ts) else None
        p_now = price_by_ts.get(ts)
        p_next = price_by_ts.get(next_ts) if next_ts is not None else None
        realized_log_return = (
            math.log(p_next / p_now)
            if p_now is not None and p_next is not None and p_now > 0
            else 0.0
        )
        gross_pnl = equity_after_cost * tgt * (math.exp(realized_log_return) - 1)
        net_pnl = gross_pnl - 0.0  # cost already deducted from base
        new_equity = equity_after_cost + gross_pnl
        rows.append(
            {
                "timestamp": ts,
                "mu": mu,
                "sigma": sigma,
                "target_weight": tgt,
                "weight": tgt,  # after sizer; current_weight will be updated for next iter
                "realized_return": realized_log_return,
                "gross_pnl": gross_pnl,
                "costs": cost,
                "net_pnl": net_pnl,
                "equity": new_equity,
            }
        )
        current_weight = tgt
        equity = new_equity

    bars = pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "mu": pl.Float64(),
            "sigma": pl.Float64(),
            "target_weight": pl.Float64(),
            "weight": pl.Float64(),
            "realized_return": pl.Float64(),
            "gross_pnl": pl.Float64(),
            "costs": pl.Float64(),
            "net_pnl": pl.Float64(),
            "equity": pl.Float64(),
        },
    )
    summary = {
        "total_return": float(equity - initial_capital),
        "n_bars": int(bars.height),
        "final_equity": float(equity),
        "total_costs": float(bars["costs"].sum()),
    }
    return bars, summary
