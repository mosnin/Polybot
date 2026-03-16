# Polybot — BTC 5-Minute Binary Options Trading Bot

Automated trading bot for Polymarket's 5-minute BTC Up/Down binary markets. Uses Bayesian inference on sub-20ms Binance perpetual ticks, Monte Carlo validation, and Kelly criterion sizing to exploit the 1-3 second information gap between centralized exchange prices and Polymarket CLOB pricing.

## Quick Start (DigitalOcean NYC3)

### 1. Create Droplet

- Log in to [DigitalOcean](https://cloud.digitalocean.com)
- **Create** → **Droplets**
- Image: **Ubuntu 22.04 LTS**
- Plan: **Basic $6/mo** (1 vCPU, 1GB RAM — sufficient for the bot)
- Region: **NYC3** (critical — lowest latency to Polygon RPC and Polymarket)
- Authentication: SSH key or password
- Click **Create Droplet**

### 2. Run Setup

Open the **DigitalOcean browser console** (or SSH in) and run:

```bash
cd /home/user/Polybot
nano .env   # paste your real keys (see .env.example)
chmod +x setup.sh
sudo ./setup.sh
```

`setup.sh` installs Python 3.11, supervisor, Redis, Caddy, all pip dependencies, runs the one-time USDC approval, and starts the bot under supervisor.

If `.env` is missing, setup.sh copies the template and exits with instructions. Fill in your keys and re-run.

### 3. Fund Your Wallet

Setup prints your Polygon wallet address. Send USDC to it:

- Network: **Polygon** (not Ethereum mainnet)
- Token: **USDC** (bridged USDC.e)
- Amount: **Any amount** — the bot works with whatever you deposit
- Minimum recommended: $100 (matches the `safety_floor_usdc` setting)

Every chart, metric, risk calculation, position size, and withdrawal limit in the system dynamically reflects your real on-chain balance at all times. There are no hardcoded amounts anywhere.

### 4. USDC Approval

`setup.sh` runs this automatically. If it failed, run manually:

```bash
cd /home/user/Polybot
venv/bin/python approve_usdc.py
```

One-time operation. Approves both Polymarket exchange contracts. Costs ~$0.01 gas.

### 5. Backtest First

Before going live, validate the edge on historical data:

```bash
venv/bin/python backtest.py
venv/bin/python backtest.py --windows 2000  # higher confidence
```

This replays 1000+ historical 5-minute windows through the exact same model pipeline. Look for:

| Metric | Go-Live Threshold |
|--------|-------------------|
| Win Rate | > 55% |
| Average Edge | > 1% |
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < 15% |

Only set `TEST_MODE=false` after the backtest confirms positive edge.

### 6. Start Trading

The bot starts automatically via supervisor after `setup.sh`. Check status:

```bash
supervisorctl status polybot        # should show RUNNING
tail -f /var/log/polybot/bot.log    # live trading logs
supervisorctl restart polybot       # restart after config changes
```

To switch from test mode to live:

```bash
nano .env                           # set TEST_MODE=false
supervisorctl restart polybot       # restart picks up new config
```

### 7. Dashboard Access

**Local network:**
```
http://<droplet-ip>:8501
```

**Remote (phone/laptop) via ngrok:**
```bash
export NGROK_AUTHTOKEN=your_token   # get free at ngrok.com
venv/bin/python ngrok_integration.py
```

Prints a public HTTPS URL accessible from any device.

**With custom domain (Caddy):**

Add to `/etc/caddy/Caddyfile`:
```
your-domain.com {
    reverse_proxy localhost:8501
}
```
Then `systemctl restart caddy`. Caddy auto-provisions TLS via Let's Encrypt.

## Architecture

Five concurrent async coroutines in a single process:

```
Binance WebSocket → tick_queue → BayesianModel → Signal → OrderExecutor → CLOB
     ↑                                ↑                        ↑
  ccxt.pro                    Monte Carlo (1000 GBM)     Maker limit orders
  BTC/USDT:USDT               + Kelly criterion          midpoint - 0.01

GammaMarketFinder → market_queue → model.reset()
                                   (new 5-min window)

Stale Order Canceller → every 10s → frees locked capital
```

Target: **tick-to-order-decision in under 80ms** on a standard VPS.

## Live Balance — Everything is Dynamic

Every display, calculation, and decision in the system uses the **real USDC balance from your Polygon wallet**, fetched live via web3 on every tick cycle:

- **Dashboard balance card**: live from `executor.get_current_balance()`
- **Equity curve chart**: every point is a live balance snapshot
- **Position sizing**: `_compute_order_size()` uses live balance minus safety floor
- **Drawdown calculation**: `(peak_balance - current_balance) / peak_balance` — all live
- **Withdraw max**: `state.balance - $1.00` buffer — updates every 5 seconds
- **Compounding gate**: scales with live balance after activation
- **Risk slider**: exposure percentage applied to live balance

`config.starting_capital` ($100) is only a fallback if the very first RPC call fails on startup. It is never used in steady-state operation.

## Key Features

### Auto-Compounding

Position sizes are gated until the bot proves its edge:

1. **First 200 trades**: sizes capped to `starting_capital` (conservative)
2. **After 200 trades**: activates only if ALL three conditions are met:
   - Win rate > 50%
   - Average daily return > 1%
   - Sufficient equity history for calculation
3. **After activation**: Kelly sizes scale with full live balance — profits compound

### Dynamic Z-Score (Liquidity-Aware)

The z-score (edge confidence metric) adjusts based on real-time CLOB order book depth:

- **Thin order book** (< 500 tokens): z-score scaled down by 1.5x → higher bar to trade
- **Thick order book** (> 5000 tokens): z-score scaled up by 0.8x → edge more reliable
- **Unavailable**: defaults to 1.0x (no adjustment)

This prevents false signals in illiquid markets where small orders move the midpoint.

### Batch Orders (Low Volatility)

When per-tick volatility drops below `low_vol_threshold` (0.05%), the bot splits orders across 3 price levels:

| Level | Price | Purpose |
|-------|-------|---------|
| L0 | midpoint - $0.01 | Best fill probability |
| L1 | midpoint - $0.02 | Better price, moderate fill |
| L2 | midpoint - $0.03 | Best price for us, lowest fill |

Total position size is identical to a single order — it's just distributed. This maximizes maker rebate capture during calm markets.

### Redis State Persistence

Bayesian model state (alpha, beta, price history, tick count) is persisted to Redis on every tick update. On restart, the model resumes from saved state instead of starting from an uninformative prior. This preserves winning edge continuity across deployments.

Set in `.env`:
```
REDIS_URL=redis://localhost:6379/0
```

## Configuration

All settings are in `config.py` (frozen dataclass) with secrets in `.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `starting_capital` | $100 | Fallback only — live balance is always used |
| `min_exposure_pct` | 5% | Floor for position size as % of balance |
| `max_exposure_pct` | 10% | Cap for position size (adjustable via dashboard) |
| `min_edge_threshold` | 2% | Minimum net edge after fees to trade |
| `safety_floor_usdc` | $100 | Never trade below this balance |
| `drawdown_hard_stop_pct` | 25% | Halt if drawdown from peak exceeds this |
| `max_consecutive_losses` | 5 | Skip window after this many losses in a row |
| `round_trip_cost_pct` | 0.3% | Worst-case fee assumption (maker is lower) |
| `low_vol_threshold` | 0.0005 | Per-tick vol below which batch orders activate |
| `mc_paths` | 1000 | Monte Carlo simulation paths |

## Troubleshooting

### Latency > 80ms

1. **Verify NYC3 region**: Polygon validators and Alchemy infrastructure are concentrated on the US East Coast. NYC3 gives sub-30ms RPC latency.
   ```bash
   curl -w "time_total: %{time_total}s\n" -o /dev/null -s \
     -X POST -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
     $ALCHEMY_RPC_URL
   ```
   Should show < 0.03s. If > 0.05s, check your Alchemy plan or region.

2. **Check Redis**: `redis-cli ping` should return `PONG`. If Redis is down, model persistence adds 0ms overhead (fails silently), but check if something else is wrong.

3. **Reduce MC paths**: Set `mc_paths: int = 500` in `config.py` — saves ~2ms per evaluation. Still statistically valid for edge detection.

4. **Check system load**: `htop` — the bot should use < 20% CPU. If higher, check for runaway processes.

### Drawdown Recovery

The bot auto-halts at 25% drawdown from peak balance. To recover:

1. **Check logs**: `tail -100 /var/log/polybot/bot.log | grep -i "drawdown\|halt\|error"`
2. **Run backtest on recent data**: `venv/bin/python backtest.py --days 1` — does the edge still hold?
3. **If temporary dip**: Resume via dashboard "Resume Trading" button, or restart:
   ```bash
   supervisorctl restart polybot
   ```
4. **If persistent**: The market regime may have changed. Keep the bot paused and monitor.

### Bot Won't Start

```bash
# Check supervisor logs
supervisorctl tail polybot stderr

# Validate .env
cat .env | head -5

# Test config loading
venv/bin/python -c "from config import load_config; c = load_config(); print(f'Config OK: {c.chain_id}')"

# Check Redis
systemctl status redis-server

# Check Python venv
venv/bin/python --version  # should be 3.11.x
```

### Connection Issues

- **Binance WebSocket disconnects**: The bot auto-reconnects with exponential backoff (up to 10 retries). Check logs for `WebSocket` entries.
- **CLOB API errors**: Usually transient. The bot skips the tick and retries on the next cycle.
- **Alchemy RPC rate limits**: Free tier allows 300 req/s. If hitting limits, upgrade to Growth plan.

## File Structure

```
Polybot/
├── bot.py                 # Main orchestrator — 5 async coroutines
├── model.py               # Bayesian inference + Monte Carlo + Kelly
├── executor.py            # CLOB order execution (maker limits, batch orders)
├── data_feed.py           # Binance WebSocket + Gamma market discovery
├── config.py              # Frozen dataclass configuration
├── dashboard.py           # Streamlit real-time dashboard (port 8501)
├── backtest.py            # Historical backtesting (standalone)
├── approve_usdc.py        # One-time USDC approval for exchange contracts
├── withdraw.py            # USDC withdrawal from Polygon wallet
├── ngrok_integration.py   # HTTPS tunnel for remote dashboard access
├── setup.sh               # DigitalOcean VPS deployment script
├── supervisor.conf        # Process management configuration
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

## Security

- **Zero custody**: Your private key never leaves the VPS. All signing is local via web3.py.
- **No external key storage**: Keys are in `.env` on your server only.
- **.env is gitignored**: Never committed to version control.
- **Supervisor isolation**: Bot runs as a managed process, not a background shell job.

## License

Private. Not for redistribution.
