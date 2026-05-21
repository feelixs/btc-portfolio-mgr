from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from btc_portfolio_mgr.data.storage import write_parquet
from btc_portfolio_mgr.features.pipeline import compose_features
from tests.fixtures.synthetic_prices import hourly


def test_run_backtest_produces_scenario_table(tmp_path: Path) -> None:
    from scripts import run_backtest as rb

    # 9000 hourly rows ≈ 375 days. After 90d warmup + 7d horizon, ~270 usable days.
    n = 9000
    rng = np.random.default_rng(11)
    rets = rng.normal(0, 0.01, n - 1)
    prices_list = [100.0]
    for r in rets:
        prices_list.append(prices_list[-1] * float(math.exp(r)))
    prices = hourly(prices_list)
    features = compose_features(prices)

    prices_path = tmp_path / "btc_hourly.parquet"
    features_path = tmp_path / "btc_features.parquet"
    out_path = tmp_path / "backtest_results.json"
    write_parquet(prices, prices_path)
    features.write_parquet(features_path)

    result = rb.run(
        prices_path=prices_path,
        features_path=features_path,
        out_path=out_path,
        eval_days=30,  # short backtest for the test
    )

    # Must contain the four scenario groups
    assert "defaults" in result
    assert "buy_and_hold" in result
    assert "kelly_sweep" in result
    assert "threshold_sweep" in result
    # Defaults must report core metrics
    for key in ("sharpe", "max_drawdown", "total_return"):
        assert key in result["defaults"]
    # Output JSON file exists
    assert out_path.exists()
