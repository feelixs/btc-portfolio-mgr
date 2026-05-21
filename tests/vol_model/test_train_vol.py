from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from btc_portfolio_mgr.data.storage import write_parquet
from tests.fixtures.synthetic_prices import hourly


def test_train_vol_produces_artifact(tmp_path: Path) -> None:
    from scripts import train_vol as tv

    rng = np.random.default_rng(7)
    n = 2000
    returns = rng.normal(0, 0.01, n - 1)
    prices_list = [100.0]
    for r in returns:
        prices_list.append(prices_list[-1] * float(math.exp(r)))
    prices = hourly(prices_list)
    assert prices.height == n

    prices_path = tmp_path / "btc_hourly.parquet"
    artifact_path = tmp_path / "btc_vol.json"
    write_parquet(prices, prices_path)

    result = tv.run(prices_path=prices_path, artifact_path=artifact_path)

    assert artifact_path.exists()
    # Result dict has the 4 metrics + n_training_returns
    assert "qlike" in result
    assert "mse_log" in result
    assert "correlation" in result
    assert "mae" in result
    assert "n_training_returns" in result
    # JSON sidecar has the right structure
    data = json.loads(artifact_path.read_text())
    assert "params" in data
    assert "spec" in data
    assert data["spec"]["vol"] == "GARCH"
    assert data["spec"]["o"] == 1
    assert data["spec"]["dist"] == "t"
    assert "trained_at" in data
    assert "git_sha" in data
    assert "eval_metrics" in data
