"""Train the BTC GJR-GARCH 24h vol model: fit, in-sample eval, save JSON artifact."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.model.git_info import current_git_sha
from btc_portfolio_mgr.vol_model.garch import fit_gjr_garch, forecast_24h_vol
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

WARMUP_OBS = 500
EVAL_STRIDE = 24


def _evaluate_in_sample(
    params: dict[str, float],
    log_returns: Any,
    realized_24h_arr: np.ndarray,
) -> dict[str, float]:
    n = log_returns.len()
    preds: list[float] = []
    realized: list[float] = []
    for t in range(WARMUP_OBS, n - 24, EVAL_STRIDE):
        forecast_vol = forecast_24h_vol(
            params=params,
            log_returns=log_returns,
            spec=DEFAULT_SPEC,
            scale_factor=SCALE_FACTOR,
            last_obs_index=t,
        )
        realized_vol = float(realized_24h_arr[t + 24])
        if not np.isfinite(realized_vol):
            continue
        preds.append(forecast_vol)
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
    realized_24h = compute_realized_24h_vol(log_returns, window_hours=24)
    metrics = _evaluate_in_sample(params, log_returns, realized_24h.to_numpy())
    print("in-sample 24h vol forecast metrics:")
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
    )
    save_vol_artifact(artifact, artifact_path)
    print(f"saved vol artifact to {artifact_path}")
    return {**metrics, "n_training_returns": n}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
