"""Materialize the feature matrix from hourly prices to parquet."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from btc_portfolio_mgr.data.storage import read_parquet
from btc_portfolio_mgr.features.pipeline import compose_features

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "data" / "btc_hourly.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "btc_features.parquet"


def run(
    prices_path: Path = DEFAULT_PRICES,
    features_path: Path = DEFAULT_FEATURES,
) -> None:
    prices = read_parquet(prices_path)
    features = compose_features(prices)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(features_path)
    print(f"wrote {features.height} rows to {features_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
