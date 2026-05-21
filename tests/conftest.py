from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def coingecko_range_response() -> dict:
    with (FIXTURES_DIR / "coingecko_range_response.json").open() as f:
        return json.load(f)


@pytest.fixture
def binance_bookticker_lines() -> list[str]:
    with (FIXTURES_DIR / "binance_bookticker_stream.jsonl").open() as f:
        return [line.rstrip("\n") for line in f if line.strip()]
