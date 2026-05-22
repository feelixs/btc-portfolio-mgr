"""Live IC = Spearman correlation between predicted mu and realized 7d log return."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def compute_live_ic(df: pl.DataFrame) -> dict[str, float]:
    """df must have columns: timestamp, mu, realized_return."""
    n = df.height
    if n < 2:
        return {"n": n, "ic": math.nan}
    mu = df["mu"].to_numpy()
    ret = df["realized_return"].to_numpy()
    mask = np.isfinite(mu) & np.isfinite(ret)
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "ic": math.nan}
    ic, _ = spearmanr(mu[mask], ret[mask])
    return {"n": int(mask.sum()), "ic": float(ic)}
