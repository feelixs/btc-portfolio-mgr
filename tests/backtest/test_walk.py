from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast

import math
import numpy as np
import polars as pl
import pytest

from btc_portfolio_mgr.backtest.walk import walk_forward_forecasts
from btc_portfolio_mgr.data.storage import write_parquet
from btc_portfolio_mgr.features.pipeline import compose_features
from tests.fixtures.synthetic_prices import hourly


def _synth_prices(n: int = 9000, seed: int = 7) -> pl.DataFrame:
    """Random-walk hourly prices long enough for 90d warmup + 7d target + several refits."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n - 1)
    prices_list = [100.0]
    for r in rets:
        prices_list.append(prices_list[-1] * float(math.exp(r)))
    return hourly(prices_list)


def test_walk_forward_returns_dataframe_with_expected_columns() -> None:
    prices = _synth_prices()
    features = compose_features(prices)
    # Pick eval window after the 90d warmup so features are non-null.
    start = cast(datetime, prices["timestamp"].min()) + timedelta(days=120)
    end = start + timedelta(days=30)
    forecasts = walk_forward_forecasts(
        prices=prices,
        features=features,
        eval_start=start,
        eval_end=end,
        refit_cadence_days=7,
    )
    assert forecasts.columns == ["timestamp", "mu", "sigma", "refit_anchor"]
    # Daily evals -> roughly 30 rows
    assert 25 <= forecasts.height <= 31


def test_walk_forward_no_future_leakage() -> None:
    """Every eval bar's refit_anchor must precede or equal its timestamp - 7d (return
    model needs labels reaching back to the data end)."""
    prices = _synth_prices()
    features = compose_features(prices)
    start = cast(datetime, prices["timestamp"].min()) + timedelta(days=120)
    end = start + timedelta(days=21)
    forecasts = walk_forward_forecasts(
        prices=prices,
        features=features,
        eval_start=start,
        eval_end=end,
        refit_cadence_days=7,
    )
    for row in forecasts.iter_rows(named=True):
        assert row["refit_anchor"] <= row["timestamp"], (
            f"refit_anchor {row['refit_anchor']} must be <= eval {row['timestamp']}"
        )


def test_walk_forward_forecasts_are_finite() -> None:
    prices = _synth_prices()
    features = compose_features(prices)
    start = cast(datetime, prices["timestamp"].min()) + timedelta(days=120)
    end = start + timedelta(days=14)
    forecasts = walk_forward_forecasts(
        prices=prices,
        features=features,
        eval_start=start,
        eval_end=end,
        refit_cadence_days=7,
    )
    mus = forecasts["mu"].to_numpy()
    sigmas = forecasts["sigma"].to_numpy()
    assert np.all(np.isfinite(mus)), "mu must be finite for every eval bar"
    assert np.all(np.isfinite(sigmas)), "sigma must be finite for every eval bar"
    assert np.all(sigmas > 0), "sigma must be positive"


def test_walk_forward_raises_when_eval_window_too_early() -> None:
    """If eval_start is before the 90d feature warmup completes, the walk
    can't produce any usable training data and should raise."""
    prices = _synth_prices()
    features = compose_features(prices)
    too_early = cast(datetime, prices["timestamp"].min()) + timedelta(days=10)
    with pytest.raises(ValueError):
        walk_forward_forecasts(
            prices=prices,
            features=features,
            eval_start=too_early,
            eval_end=too_early + timedelta(days=7),
            refit_cadence_days=7,
        )
