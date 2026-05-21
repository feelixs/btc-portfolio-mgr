# btc-portfolio-mgr

Systematic BTC portfolio management with hourly rebalancing.
Hourly return forecast (LightGBM) + GJR-GARCH vol model → fractional-Kelly vol-targeted sizing.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env  # fill in COINGECKO_API_KEY
```

## Backfill historical

```bash
python scripts/backfill_historical.py
```

## Run tests

```bash
pytest -v
```
