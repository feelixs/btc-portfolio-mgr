"""Walk-forward backtest: forecasts × scenario sweep + buy-and-hold benchmark."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import polars as pl

from btc_portfolio_mgr.backtest.metrics import compute_backtest_metrics
from btc_portfolio_mgr.backtest.replay import replay_sizing
from btc_portfolio_mgr.backtest.walk import walk_forward_forecasts
from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.sizing.params import SizingParams

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "btc_features.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "backtest_results.json"

DEFAULT_COST_BPS = 10.0
KELLY_SWEEP = [0.05, 0.10, 0.20, 0.40]
THRESHOLD_SWEEP = [0.0, 0.01, 0.02, 0.05]


def _buy_and_hold(prices: pl.DataFrame, eval_ts: list[datetime]) -> dict[str, float]:
    """Buy-and-hold BTC starting at eval_ts[0]; held to eval_ts[-1]."""
    if not eval_ts:
        return {"total_return": 0.0, "n_bars": 0}
    price_by_ts = dict(zip(prices["timestamp"].to_list(), prices["price"].to_list()))
    p0 = price_by_ts.get(eval_ts[0])
    pN = price_by_ts.get(eval_ts[-1])
    if p0 is None or pN is None or p0 <= 0:
        return {"total_return": float("nan")}
    rows: list[dict[str, Any]] = []
    base = 1.0
    for ts in eval_ts:
        pt = price_by_ts.get(ts, p0)
        equity = base * (pt / p0)
        rows.append(
            {
                "timestamp": ts,
                "weight": 1.0,
                "net_pnl": 0.0,  # filled below
                "equity": equity,
            }
        )
    # Recompute net_pnl as Δequity.
    for i in range(1, len(rows)):
        rows[i]["net_pnl"] = rows[i]["equity"] - rows[i - 1]["equity"]
    bars = pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "weight": pl.Float64(),
            "net_pnl": pl.Float64(),
            "equity": pl.Float64(),
        },
    )
    return compute_backtest_metrics(bars)


def run(
    prices_path: Path = DEFAULT_PRICES,
    features_path: Path = DEFAULT_FEATURES,
    out_path: Path = DEFAULT_OUT,
    eval_days: int | None = None,
) -> dict[str, Any]:
    """Run the full backtest sweep. Returns nested dict of scenario → metrics."""
    prices = read_parquet(prices_path)
    features = pl.read_parquet(features_path)
    # Eval window: from first usable feature row + 7d (to allow refit warmup) to end.
    nonnull_features = features.drop_nulls()
    if nonnull_features.height == 0:
        raise RuntimeError("no non-null feature rows; can't backtest")
    earliest_usable = cast(datetime, nonnull_features["timestamp"].min())
    eval_start: datetime = earliest_usable + timedelta(days=7)
    eval_end: datetime = cast(datetime, prices["timestamp"].max()) - timedelta(days=8)  # leave a 7d label window
    if eval_days is not None:
        eval_end = min(eval_end, eval_start + timedelta(days=eval_days))
    print(f"eval window: {eval_start} → {eval_end} ({(eval_end - eval_start).days} days)")

    print("walking forward (refit weekly, forecast daily)…")
    forecasts = walk_forward_forecasts(
        prices=prices,
        features=features,
        eval_start=eval_start,
        eval_end=eval_end,
        refit_cadence_days=7,
    )
    print(f"forecasts: {forecasts.height} daily bars")

    results: dict[str, Any] = {}

    # Defaults
    bars_def, _ = replay_sizing(
        forecasts, prices, SizingParams(), cost_bps=DEFAULT_COST_BPS
    )
    results["defaults"] = compute_backtest_metrics(bars_def)

    # Buy-and-hold benchmark
    eval_ts_list = forecasts["timestamp"].to_list()
    results["buy_and_hold"] = _buy_and_hold(prices, eval_ts_list)

    # Kelly fraction sweep
    results["kelly_sweep"] = {}
    for k in KELLY_SWEEP:
        bars, _ = replay_sizing(
            forecasts,
            prices,
            SizingParams(kelly_fraction=k),
            cost_bps=DEFAULT_COST_BPS,
        )
        results["kelly_sweep"][f"k={k:.2f}"] = compute_backtest_metrics(bars)

    # Threshold sweep
    results["threshold_sweep"] = {}
    for thr in THRESHOLD_SWEEP:
        bars, _ = replay_sizing(
            forecasts,
            prices,
            SizingParams(rebalance_threshold=thr),
            cost_bps=DEFAULT_COST_BPS,
        )
        results["threshold_sweep"][f"thr={thr:.2f}"] = compute_backtest_metrics(bars)

    # Print summary
    print("\n--- DEFAULTS (k=0.20, threshold=0.02, 10 bps) ---")
    for k, v in results["defaults"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"  {k}: {v}")
    print("\n--- BUY-AND-HOLD ---")
    for k, v in results["buy_and_hold"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"  {k}: {v}")
    print("\n--- KELLY FRACTION SWEEP ---")
    for sweep_key, m in results["kelly_sweep"].items():
        print(
            f"  {sweep_key}: sharpe={m['sharpe']:.3f} "
            f"return={m['annualized_return']:.3f} "
            f"DD={m['max_drawdown']:.3f}"
        )
    print("\n--- THRESHOLD SWEEP ---")
    for sweep_key, m in results["threshold_sweep"].items():
        print(
            f"  {sweep_key}: sharpe={m['sharpe']:.3f} "
            f"turnover={m['turnover_annualized']:.2f}x"
        )

    # Persist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(_jsonify(results), f, indent=2, default=str)
    print(f"\nsaved results to {out_path}")
    return results


def _jsonify(obj: Any) -> Any:
    """Recursively convert NaN floats to null so the JSON is valid."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main() -> None:
    run()


if __name__ == "__main__":
    main()
