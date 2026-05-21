"""Single-cycle inference: latest features -> mu, sigma, target_weight."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl

from btc_portfolio_mgr.model.inference import load_artifact, predict
from btc_portfolio_mgr.sizing.params import SizingParams
from btc_portfolio_mgr.sizing.sizer import target_weight as sizer_target_weight
from btc_portfolio_mgr.vol_model.garch import forecast_24h_vol
from btc_portfolio_mgr.vol_model.inference import load_vol_artifact
from btc_portfolio_mgr.vol_model.returns import extract_log_returns


@dataclass(frozen=True)
class LiveInferenceResult:
    timestamp: datetime
    mu: float
    sigma: float
    target_weight: float
    price_usdt: float


def run_inference(
    prices_path: Path,
    features_path: Path,
    model_path: Path,
    model_metadata_path: Path,
    vol_path: Path,
    sizing_params: SizingParams,
    current_weight: float,
) -> LiveInferenceResult:
    """Load latest features + artifacts, return today's (mu, sigma, target_weight, price).

    `current_weight` is passed in (computed from realized position notional / equity at
    the caller's mark price) and used only by the sizer's rebalance-threshold check.
    `price_usdt` returned here is the historical price at the latest feature timestamp;
    callers should re-fetch mark price at order time for actual sizing.
    """
    prices = pl.read_parquet(prices_path)
    features = pl.read_parquet(features_path)

    nonnull = features.drop_nulls()
    if nonnull.height == 0:
        raise RuntimeError("no usable feature rows")
    latest_row = nonnull.tail(1)
    ts = cast(datetime, latest_row["timestamp"][0])

    matched_price = prices.filter(pl.col("timestamp") == ts)
    if matched_price.height == 0:
        raise RuntimeError(f"no price for feature timestamp {ts}")
    price_usdt = float(matched_price["price"][0])

    model_artifact = load_artifact(model_path, model_metadata_path)
    mu_series = predict(model_artifact, latest_row)
    mu = float(mu_series.to_numpy()[0])

    vol_artifact = load_vol_artifact(vol_path)
    rets_df = extract_log_returns(prices.filter(pl.col("timestamp") <= ts))
    sigma = forecast_24h_vol(
        params=vol_artifact.params,
        log_returns=rets_df["log_return"],
        spec=vol_artifact.spec,
        scale_factor=vol_artifact.scale_factor,
        horizon_hours=vol_artifact.horizon_hours,
    )

    tgt = sizer_target_weight(
        mu=mu, sigma=sigma, current_weight=current_weight, params=sizing_params
    )
    return LiveInferenceResult(
        timestamp=ts,
        mu=float(mu),
        sigma=float(sigma),
        target_weight=float(tgt),
        price_usdt=price_usdt,
    )
