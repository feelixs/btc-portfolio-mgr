"""VolArtifact JSON persistence and 24h vol inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from btc_portfolio_mgr.vol_model.garch import forecast_24h_vol
from btc_portfolio_mgr.vol_model.spec import GarchSpec


@dataclass(frozen=True)
class VolArtifact:
    params: dict[str, float]
    spec: GarchSpec
    scale_factor: float
    trained_at: datetime
    git_sha: str
    eval_metrics: dict[str, float]
    n_training_returns: int
    horizon_hours: int = 24  # forecast horizon (e.g. 24, 168). GARCH params are
    # horizon-agnostic, but this stamps the artifact with the horizon it was
    # evaluated against and the default `predict_24h_vol` will use.


def save_vol_artifact(artifact: VolArtifact, path: Path) -> None:
    """Write VolArtifact as JSON. No pickle — all fields are JSON-serializable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "params": artifact.params,
        "spec": artifact.spec.to_dict(),
        "scale_factor": artifact.scale_factor,
        "trained_at": artifact.trained_at.isoformat(),
        "git_sha": artifact.git_sha,
        "eval_metrics": artifact.eval_metrics,
        "n_training_returns": artifact.n_training_returns,
        "horizon_hours": artifact.horizon_hours,
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_vol_artifact(path: Path) -> VolArtifact:
    """Restore VolArtifact from JSON.

    Note: the key order in `data["params"]` is load-bearing — am.fix()
    in forecast_24h_vol receives the params as a positional array, so
    hand-editing the JSON to reorder keys will produce silently wrong
    forecasts. Treat the artifact as opaque.
    """
    with path.open() as f:
        data = json.load(f)
    return VolArtifact(
        params={str(k): float(v) for k, v in data["params"].items()},
        spec=GarchSpec.from_dict(data["spec"]),
        scale_factor=float(data["scale_factor"]),
        trained_at=datetime.fromisoformat(data["trained_at"]),
        git_sha=str(data["git_sha"]),
        eval_metrics={str(k): float(v) for k, v in data["eval_metrics"].items()},
        n_training_returns=int(data["n_training_returns"]),
        horizon_hours=int(data.get("horizon_hours", 24)),
    )


def predict_24h_vol(
    artifact: VolArtifact,
    log_returns: pl.Series,
    last_obs_index: int | None = None,
) -> float:
    """Forecast integrated vol over `artifact.horizon_hours` from saved params.

    The artifact intentionally does NOT store historical returns — the caller
    must provide `log_returns` from the data layer on every call. This keeps
    the artifact small and auditable; Phase 5 sizing fetches fresh returns
    each cycle anyway.

    Despite the historical name, this forecasts over whatever horizon the
    artifact was trained for (e.g. 168h for the 7d-return model pairing).

    Backtest optimization: pass the FULL `log_returns` series and use
    `last_obs_index` to walk forward instead of slicing the series each step.
    arch's `am.fix()` then reconstructs the model once and forecasts at
    different anchor points — much faster than rebuilding O(n) times.
    """
    return forecast_24h_vol(
        params=artifact.params,
        log_returns=log_returns,
        spec=artifact.spec,
        scale_factor=artifact.scale_factor,
        last_obs_index=last_obs_index,
        horizon_hours=artifact.horizon_hours,
    )
