# Helsinki Deployment Checklist

## One-time setup

1. SSH to Helsinki: `ssh root@clob-proxy.mindthegap.bet`
2. Install Python 3.12 + git if not present.
3. Create the runtime user: `useradd -m -s /bin/bash btcbot`
4. Clone the repo: `git clone <repo-url> /opt/btc-portfolio-mgr`
5. `chown -R btcbot:btcbot /opt/btc-portfolio-mgr`
6. As btcbot: `cd /opt/btc-portfolio-mgr && python3 -m venv .venv && .venv/bin/pip install -e .`
7. Copy `deploy/helsinki/.env.example` to `/opt/btc-portfolio-mgr/.env` and fill in keys.
   **IMPORTANT:** start with `BINANCE_NETWORK=testnet`.
8. Generate features + models on the box:
   ```
   sudo -u btcbot .venv/bin/python scripts/backfill_historical.py
   sudo -u btcbot .venv/bin/python scripts/build_features.py
   sudo -u btcbot .venv/bin/python scripts/train_model.py
   sudo -u btcbot .venv/bin/python scripts/train_vol.py
   ```
9. Read-only connectivity check: `sudo -u btcbot .venv/bin/python scripts/verify_binance.py`
10. Dry-run cycle: `sudo -u btcbot .venv/bin/python scripts/run_live.py --dry-run`
11. Install systemd: `cp deploy/helsinki/btc-portfolio.{service,timer} /etc/systemd/system/`
12. Enable timer: `systemctl daemon-reload && systemctl enable --now btc-portfolio.timer`
13. Watch logs: `journalctl -u btc-portfolio.service -f`

## Flip from testnet to mainnet

1. Confirm at least 1 week of green testnet cycles in `data/live_log.parquet`.
2. Edit `.env`: change `BINANCE_NETWORK=mainnet` and update keys to a mainnet API key.
3. Fund the mainnet futures account with $10.
4. Manual dry-run first: `sudo -u btcbot .venv/bin/python scripts/run_live.py --dry-run`
5. If dry-run looks right: `sudo -u btcbot .venv/bin/python scripts/run_live.py`
6. Inspect the live log: `sudo -u btcbot .venv/bin/python -c "import polars as pl; print(pl.read_parquet('data/live_log.parquet').tail(20))"`
7. Verify position via Binance UI.

## Emergency procedures

- **Halt the bot:** `touch /opt/btc-portfolio-mgr/HALT`
- **Resume:** `rm /opt/btc-portfolio-mgr/HALT`
- **Flatten the position:** `sudo -u btcbot /opt/btc-portfolio-mgr/.venv/bin/python scripts/flatten_position.py`
- **Stop the timer:** `systemctl disable --now btc-portfolio.timer`
- **Restart the timer:** `systemctl enable --now btc-portfolio.timer`

## Daily monitoring

- Tail log: `tail -n 100 /var/log/btc-portfolio.log`
- IC report: `sudo -u btcbot .venv/bin/python scripts/live_ic_report.py`
- Last 10 cycles: `sudo -u btcbot .venv/bin/python -c "import polars as pl; print(pl.read_parquet('data/live_log.parquet').tail(10))"`

## Common failure modes + recovery

| Symptom | Cause | Recovery |
|---|---|---|
| `data stale by ...` halt | Features not refreshed | Run `scripts/build_features.py` manually, check cron for feature refresh |
| `drawdown ... exceeds 0.15` halt | Strategy in DD | Investigate; either reduce capital or wait/recalibrate |
| `5 consecutive losing days` halt | Model degradation | Check live IC report; consider retraining |
| Order error log row | Network blip during submit | Check Binance UI for order with `mtg-<id>`; manually reconcile if needed |
