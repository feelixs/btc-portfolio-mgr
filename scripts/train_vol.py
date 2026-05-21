"""Train the BTC GJR-GARCH vol model at HORIZON_HOURS: fit, in-sample eval, save."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.model.git_info import current_git_sha
from btc_portfolio_mgr.vol_model.garch import (
    fit_gjr_garch,
    forecast_integrated_vols_batch,
)
from btc_portfolio_mgr.vol_model.inference import VolArtifact, save_vol_artifact
from btc_portfolio_mgr.vol_model.metrics import (
    compute_mae_vol,
    compute_mse_log,
    compute_qlike,
    compute_vol_correlation,
)
from btc_portfolio_mgr.vol_model.returns import extract_log_returns
from btc_portfolio_mgr.vol_model.spec import DEFAULT_SPEC, SCALE_FACTOR
from btc_portfolio_mgr.vol_model.target import compute_realized_24h_vol

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_ARTIFACT = REPO_ROOT / "models" / "btc_vol.json"

HORIZON_HOURS = 168  # 7 days — matches the return model's prediction horizon
WARMUP_OBS = 500
# One eval per week. At 168h horizon, daily stride would produce 7x redundant
# evals (each forecast's target window covers the next 6 stride steps); weekly
# stride yields ~independent samples and cuts the eval loop to ~1-2 minutes.
EVAL_STRIDE = 168


def _evaluate_in_sample(
    params: dict[str, float],
    log_returns: Any,
    realized_arr: np.ndarray,
    horizon_hours: int,
) -> dict[str, float]:
    # Single batched arch call computes integrated vol forecasts at every anchor
    # from WARMUP_OBS onward — far faster than rebuilding the arch model in a
    # per-anchor loop.
    n = log_returns.len()
    pred_vols_all = forecast_integrated_vols_batch(
        params=params,
        log_returns=log_returns,
        horizon_hours=horizon_hours,
        start_index=WARMUP_OBS,
        spec=DEFAULT_SPEC,
        scale_factor=SCALE_FACTOR,
    )
    # pred_vols_all[i] is the forecast anchored at obs WARMUP_OBS + i.
    # Walk by EVAL_STRIDE and align realized vol over [t+1, t+horizon].
    preds: list[float] = []
    realized: list[float] = []
    for t in range(WARMUP_OBS, n - horizon_hours, EVAL_STRIDE):
        pred_vol = float(pred_vols_all[t - WARMUP_OBS])
        realized_vol = float(realized_arr[t + horizon_hours])
        if not np.isfinite(realized_vol) or not np.isfinite(pred_vol):
            continue
        preds.append(pred_vol)
        realized.append(realized_vol)
    pred_arr = np.array(preds)
    real_arr = np.array(realized)
    return {
        "qlike": compute_qlike(real_arr, pred_arr),
        "mse_log": compute_mse_log(real_arr, pred_arr),
        "correlation": compute_vol_correlation(real_arr, pred_arr),
        "mae": compute_mae_vol(real_arr, pred_arr),
    }


def run(
    prices_path: Path = DEFAULT_PRICES,
    artifact_path: Path = DEFAULT_ARTIFACT,
    horizon_hours: int = HORIZON_HOURS,
) -> dict[str, Any]:
    prices = read_parquet(prices_path)
    rets_df = extract_log_returns(prices)
    n = rets_df.height
    print(f"returns: {n} rows after gap drop")
    log_returns = rets_df["log_return"]
    params = fit_gjr_garch(log_returns, spec=DEFAULT_SPEC)
    print(
        "fitted GJR-GARCH(1,1,1)-t: "
        f"omega={params['omega']:.6f} "
        f"alpha={params['alpha[1]']:.4f} "
        f"gamma={params['gamma[1]']:.4f} "
        f"beta={params['beta[1]']:.4f} "
        f"nu={params.get('nu', float('nan')):.2f}"
    )
    realized = compute_realized_24h_vol(log_returns, window_hours=horizon_hours)
    metrics = _evaluate_in_sample(
        params, log_returns, realized.to_numpy(), horizon_hours
    )
    print(f"in-sample {horizon_hours}h vol forecast metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    artifact = VolArtifact(
        params=params,
        spec=DEFAULT_SPEC,
        scale_factor=SCALE_FACTOR,
        trained_at=datetime.now(tz=timezone.utc),
        git_sha=current_git_sha(REPO_ROOT),
        eval_metrics=metrics,
        n_training_returns=n,
        horizon_hours=horizon_hours,
    )
    save_vol_artifact(artifact, artifact_path)
    print(f"saved vol artifact to {artifact_path}")
    return {**metrics, "n_training_returns": n, "horizon_hours": horizon_hours}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
