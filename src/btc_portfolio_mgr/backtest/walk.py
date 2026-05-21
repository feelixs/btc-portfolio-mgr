"""Walk-forward forecast generation: refit weekly, forecast daily."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import polars as pl

from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS
from btc_portfolio_mgr.model.target import build_dataset
from btc_portfolio_mgr.model.train import train_lightgbm
from btc_portfolio_mgr.vol_model.garch import (
    fit_gjr_garch,
    forecast_integrated_vols_batch,
)
from btc_portfolio_mgr.vol_model.returns import extract_log_returns
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR

RETURN_HORIZON_HOURS = 168
VOL_HORIZON_HOURS = 168


def _refit_dates(
    eval_start: datetime, eval_end: datetime, cadence_days: int
) -> list[datetime]:
    """Refit dates anchored at midnight UTC, walking from before eval_start."""
    # Snap eval_start back to the most recent midnight, then step back at least
    # one cadence so the first refit covers eval_start.
    floor_start = eval_start.replace(hour=0, minute=0, second=0, microsecond=0)
    first_refit = floor_start - timedelta(
        days=(floor_start.weekday() % cadence_days)
    )
    if first_refit > eval_start:
        first_refit -= timedelta(days=cadence_days)
    dates: list[datetime] = []
    cursor = first_refit
    while cursor <= eval_end:
        dates.append(cursor)
        cursor += timedelta(days=cadence_days)
    return dates


def walk_forward_forecasts(
    prices: pl.DataFrame,
    features: pl.DataFrame,
    eval_start: datetime,
    eval_end: datetime,
    refit_cadence_days: int = 7,
) -> pl.DataFrame:
    """Walk through eval window, refit at each cadence, produce daily (μ̂, σ̂).

    Returns a DataFrame with columns:
      timestamp        — eval bar (00:00 UTC each day)
      mu               — 7d-ahead log return forecast from the refit-anchor model
      sigma            — 168h integrated vol forecast (vol model evaluated at this bar)
      refit_anchor     — refit date whose models produced this row

    Raises ValueError if eval_start is too early for any non-null feature row
    (i.e. before the 90d zscore warmup has filled in).
    """
    # Validate that we have any non-null features before eval_start.
    nonnull_features = features.drop_nulls()
    if nonnull_features.height == 0:
        raise ValueError("no non-null feature rows available in features parquet")
    earliest_usable = cast(datetime, nonnull_features["timestamp"].min())
    if eval_start < earliest_usable:
        raise ValueError(
            f"eval_start {eval_start} is before earliest usable feature row "
            f"{earliest_usable}; need 90d warmup + at least one refit"
        )

    log_returns_df = extract_log_returns(prices)
    log_returns = log_returns_df["log_return"]
    return_timestamps = log_returns_df["timestamp"]
    feature_timestamps = features["timestamp"]

    out_rows: list[dict[str, Any]] = []
    for refit_date in _refit_dates(eval_start, eval_end, refit_cadence_days):
        # Return model: labels reach 7d forward, so training features must end at
        # refit_date - 7d to avoid using any future data.
        return_train_end = refit_date - timedelta(hours=RETURN_HORIZON_HOURS)
        prices_train = prices.filter(pl.col("timestamp") <= return_train_end)
        features_train = features.filter(pl.col("timestamp") <= return_train_end)
        dataset = build_dataset(
            features_train, prices_train, horizon_hours=RETURN_HORIZON_HOURS
        )
        # Resample to daily (one row per day at midnight UTC) — same as train_model.py.
        dataset = dataset.filter(pl.col("timestamp").dt.hour() == 0)
        if dataset.height < 14:
            # Not enough daily samples this early; skip this refit window.
            continue
        X = dataset.select(FEATURE_COLUMNS).to_numpy()
        y = dataset["target"].to_numpy()
        booster = train_lightgbm(X, y, num_boost_round=500)

        # Vol model: uses all returns up to refit_date (no future labels needed).
        vol_train_mask = return_timestamps <= refit_date
        vol_train_returns = log_returns.filter(vol_train_mask)
        if vol_train_returns.len() < 500:
            continue
        vol_params = fit_gjr_garch(vol_train_returns, spec=DEFAULT_SPEC)

        # Batched vol forecast across all eval dates in this refit window.
        # We need σ̂ at each eval bar t given returns up to t. Build the arch
        # model on returns up to eval_end (which covers all the bars we need),
        # then index into the per-anchor forecast array.
        next_refit_idx_start = max(
            refit_date,
            eval_start,
        )
        next_refit_end = min(
            refit_date + timedelta(days=refit_cadence_days),
            eval_end + timedelta(days=1),
        )

        # Returns array for this refit's vol inference. Use returns up to eval_end
        # so the model sees enough to forecast at every anchor in this window.
        infer_returns_mask = return_timestamps <= next_refit_end
        infer_returns = log_returns.filter(infer_returns_mask)
        infer_return_ts = return_timestamps.filter(infer_returns_mask)

        # Pre-compute integrated vol at every anchor index from 0 onward.
        # `start_index` must be < n; we want forecasts at *every* index for the
        # eval bars in this window. Pick start_index = first index whose
        # timestamp >= next_refit_idx_start - 1h (safe lower bound).
        rt_min = cast(datetime, return_timestamps.min())
        anchor_min_ts = max(
            rt_min, next_refit_idx_start - timedelta(hours=1)
        )
        start_index = int(
            cast(int, (infer_return_ts >= anchor_min_ts).arg_max())
        )
        sigmas_batch = forecast_integrated_vols_batch(
            params=vol_params,
            log_returns=infer_returns,
            horizon_hours=VOL_HORIZON_HOURS,
            start_index=start_index,
            spec=DEFAULT_SPEC,
            scale_factor=SCALE_FACTOR,
        )

        # Daily eval bars in [next_refit_idx_start, next_refit_end)
        eval_cursor = next_refit_idx_start.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        while eval_cursor < next_refit_end and eval_cursor <= eval_end:
            if eval_cursor < eval_start:
                eval_cursor += timedelta(days=1)
                continue
            # μ̂: predict from the eval-date feature row.
            feat_row = features.filter(pl.col("timestamp") == eval_cursor)
            if feat_row.height != 1:
                eval_cursor += timedelta(days=1)
                continue
            # If any feature is null at this eval bar, skip (model can't predict).
            if any(
                feat_row[col].null_count() > 0 for col in FEATURE_COLUMNS
            ):
                eval_cursor += timedelta(days=1)
                continue
            X_pred = feat_row.select(FEATURE_COLUMNS).to_numpy()
            mu = float(cast(np.ndarray, booster.predict(X_pred))[0])

            # σ̂: look up the precomputed integrated vol at this anchor.
            # Find return-series index closest to eval_cursor.
            anchor_idx_series = cast(int, (infer_return_ts == eval_cursor).arg_max())
            anchor_idx = int(anchor_idx_series) - start_index
            if 0 <= anchor_idx < len(sigmas_batch):
                sigma = float(sigmas_batch[anchor_idx])
            else:
                eval_cursor += timedelta(days=1)
                continue
            if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
                eval_cursor += timedelta(days=1)
                continue

            out_rows.append(
                {
                    "timestamp": eval_cursor,
                    "mu": mu,
                    "sigma": sigma,
                    "refit_anchor": refit_date,
                }
            )
            eval_cursor += timedelta(days=1)

    schema = {
        "timestamp": pl.Datetime("us", "UTC"),
        "mu": pl.Float64(),
        "sigma": pl.Float64(),
        "refit_anchor": pl.Datetime("us", "UTC"),
    }
    return pl.DataFrame(out_rows, schema=schema)
