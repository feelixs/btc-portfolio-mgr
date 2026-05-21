from __future__ import annotations

import math

import polars as pl

from btc_portfolio_mgr.vol_model.returns import extract_log_returns
from tests.fixtures.synthetic_prices import hourly, with_gap


def test_extract_log_returns_basic() -> None:
    prices = hourly([100.0, 110.0, 121.0, 133.1])
    df = extract_log_returns(prices)
    assert df.height == 3
    assert df.columns == ["timestamp", "log_return"]
    values = df["log_return"].to_list()
    assert math.isclose(values[0], math.log(110 / 100))
    assert math.isclose(values[1], math.log(121 / 110))
    assert math.isclose(values[2], math.log(133.1 / 121))


def test_extract_log_returns_drops_nulls_from_gap() -> None:
    prices = with_gap([100.0, 101.0, 102.0], gap_hours=2, prices_after=[110.0, 111.0])
    df = extract_log_returns(prices)
    # Reindexed: [100, 101, 102, null, null, 110, 111]
    # 1h returns: [None, r1, r2, None, None, None, r3]; after dropna -> 3 rows.
    assert df.height == 3
    values = df["log_return"].to_list()
    assert math.isclose(values[0], math.log(101 / 100))
    assert math.isclose(values[1], math.log(102 / 101))
    assert math.isclose(values[2], math.log(111 / 110))


def test_extract_log_returns_empty_input_returns_empty() -> None:
    empty = pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "price": pl.Float64(), "volume": pl.Float64()}
    )
    df = extract_log_returns(empty)
    assert df.height == 0
    assert df.columns == ["timestamp", "log_return"]
