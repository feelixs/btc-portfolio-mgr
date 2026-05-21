from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from btc_portfolio_mgr.data.storage import write_parquet
from btc_portfolio_mgr.features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA
from tests.fixtures.synthetic_prices import hourly


def test_build_writes_features_parquet(tmp_path: Path) -> None:
    from scripts import build_features as bf

    prices_path = tmp_path / "btc_hourly.parquet"
    features_path = tmp_path / "btc_features.parquet"

    # 2200 hourly bars so even zscore_90d (2160h lookback) has non-null tail.
    write_parquet(hourly([100.0 + i * 0.01 for i in range(2200)]), prices_path)

    bf.run(prices_path=prices_path, features_path=features_path)

    loaded = pl.read_parquet(features_path)
    assert loaded.schema == FEATURE_SCHEMA
    assert loaded.height == 2200
    assert loaded.columns == ["timestamp"] + FEATURE_COLUMNS
