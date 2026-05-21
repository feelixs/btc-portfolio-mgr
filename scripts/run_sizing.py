"""Compute the current target portfolio weight from latest data + saved artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.model.inference import load_artifact, predict
from btc_portfolio_mgr.sizing.params import DEFAULT_PARAMS
from btc_portfolio_mgr.sizing.sizer import target_weight
from btc_portfolio_mgr.vol_model.inference import load_vol_artifact, predict_24h_vol
from btc_portfolio_mgr.vol_model.returns import extract_log_returns

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "btc_features.parquet"
DEFAULT_RETURN_MODEL = REPO_ROOT / "models" / "btc_7d.txt"
DEFAULT_RETURN_METADATA = REPO_ROOT / "models" / "btc_7d.metadata.json"
DEFAULT_VOL_ARTIFACT = REPO_ROOT / "models" / "btc_vol.json"


def run(
    prices_path: Path = DEFAULT_PRICES,
    features_path: Path = DEFAULT_FEATURES,
    return_model_path: Path = DEFAULT_RETURN_MODEL,
    return_metadata_path: Path = DEFAULT_RETURN_METADATA,
    vol_artifact_path: Path = DEFAULT_VOL_ARTIFACT,
    current_weight: float = 0.0,
) -> dict[str, Any]:
    """Load latest data + artifacts, compute and print target weight."""
    prices = read_parquet(prices_path)
    features = pl.read_parquet(features_path)
    valid_features = features.drop_nulls()
    if valid_features.height == 0:
        raise RuntimeError("no non-null feature rows available")
    latest_features_row = valid_features.tail(1)
    log_returns = extract_log_returns(prices)["log_return"]

    return_artifact = load_artifact(return_model_path, return_metadata_path)
    vol_artifact = load_vol_artifact(vol_artifact_path)

    mu_series = predict(return_artifact, latest_features_row)
    mu = float(mu_series.to_numpy()[0])
    sigma = predict_24h_vol(vol_artifact, log_returns)
    w = target_weight(
        mu=mu, sigma=sigma, current_weight=current_weight, params=DEFAULT_PARAMS
    )

    return_horizon = return_artifact.target_horizon_hours
    vol_horizon = vol_artifact.horizon_hours
    if return_horizon != vol_horizon:
        print(
            f"WARNING: return horizon ({return_horizon}h) != vol horizon "
            f"({vol_horizon}h). Kelly formula assumes same horizon."
        )
    print(f"mu ({return_horizon}h log return forecast):    {mu:+.6f}")
    print(f"sigma ({vol_horizon}h vol forecast):           {sigma:.6f}")
    print(f"current_weight:                                {current_weight:+.4f}")
    print(f"target_weight (after threshold):               {w:+.4f}")
    return {
        "mu": mu,
        "sigma": sigma,
        "current_weight": current_weight,
        "target_weight": w,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
