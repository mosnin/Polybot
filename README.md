# Polybot — BTC 5-Minute Binary Options Trading Bot

Automated trading bot for Polymarket's 5-minute BTC Up/Down binary markets. Uses Bayesian inference on sub-20ms Bybit perpetual ticks, Monte Carlo validation, and Kelly criterion sizing to exploit the 1-3 second information gap between centralized exchange prices and Polymarket CLOB pricing.

---

## Step-by-Step Deployment on DigitalOcean

### Prerequisites

Before you begin, you need:

1. **A DigitalOcean account** — Sign up at [digitalocean.com](https://www.digitalocean.com) (they often have $200 free credit for new users)
2. **A Polygon wallet private key** — Export from MetaMask: Settings → Security → Reveal Private Key
3. **An Alchemy API key** — Free at [dashboard.alchemy.com](https://dashboard.alchemy.com) (select Polygon network)
4. **USDC on Polygon** — Bridge from Ethereum or buy directly on Polygon. Minimum $100 recommended.

---

### Step 1: Create a DigitalOcean Droplet

1. Log in to [cloud.digitalocean.com](https://cloud.digitalocean.com)
2. Click the green **Create** button in the top right → **Droplets**
3. Configure your droplet:

   | Setting | Value | Why |
   |---------|-------|-----|
   | **Region** | **New York — NYC3** | Lowest latency to Polygon RPC and Polymarket servers |
   | **Image** | **Ubuntu 22.04 (LTS) x64** | Stable, well-supported |
   | **Size** | **Basic — $6/mo** (1 vCPU, 1 GB RAM, 25 GB SSD) | More than enough for the bot |
   | **Authentication** | **Password** (simplest) or **SSH Key** (more secure) | Your choice |
   | **Hostname** | `polybot` (or anything you like) | Just a label |

4. Click **Create Droplet** and wait ~60 seconds for it to spin up
5. Copy the **IP address** shown on the droplet page — you'll need it

---

### Step 2: Connect to Your Droplet

**Option A — DigitalOcean Browser Console (easiest, no software needed):**
1. Click on your droplet name
2. Click **Console** in the top right
3. A terminal opens in your browser — you're logged in as root

**Option B — SSH from your computer:**
```bash
ssh root@YOUR_DROPLET_IP
```

---

### Step 3: Upload the Bot Code

If the code isn't already on the server, upload it:

```bash
# From your LOCAL computer (not the droplet):
scp -r /path/to/Polybot root@YOUR_DROPLET_IP:/home/user/
```

Or clone from your repository:
```bash
cd /home/user
git clone YOUR_REPO_URL Polybot
```

---

### Step 4: Configure Your .env File

```bash
cd /home/user/Polybot
cp .env.example .env
nano .env
```

Edit these two required fields — replace the placeholder text with your actual keys:

```
POLYGON_PRIVATE_KEY=0xYourActualPrivateKeyHere
ALCHEMY_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YourActualApiKey
```

Optional but recommended settings:
```
STARTING_CAPITAL=100          # Your initial USDC deposit amount
TEST_MODE=true                # Keep true until backtest passes
ENABLE_DASHBOARD=true         # Web dashboard on port 8501
REDIS_URL=redis://localhost:6379/0   # State persistence across restarts
```

Save: press `Ctrl+X`, then `Y`, then `Enter`.

Lock down file permissions:
```bash
chmod 600 .env
```

---

### Step 5: Run the Setup Script

```bash
chmod +x setup.sh
sudo ./setup.sh
```

This takes 2-3 minutes and automatically:
- Installs Python 3.11, Redis, Caddy, Supervisor
- Creates a virtual environment with all dependencies
- Validates your .env configuration
- Runs the one-time USDC approval (costs ~$0.01 gas)
- Starts the bot under Supervisor (auto-restarts on crash)

If setup fails on the .env step, edit your .env file and re-run `sudo ./setup.sh`.

---

### Step 6: Fund Your Wallet

The setup script prints your Polygon wallet address. Send USDC to it:

- **Network**: Polygon (NOT Ethereum mainnet — you'll lose funds)
- **Token**: USDC (bridged USDC.e on Polygon)
- **Amount**: Minimum $100 recommended

You can also find your wallet address anytime:
```bash
cd /home/user/Polybot
venv/bin/python -c "from eth_account import Account; from dotenv import load_dotenv; import os; load_dotenv(); print(Account.from_key(os.getenv('POLYGON_PRIVATE_KEY')).address)"
```

---

### Step 7: Run a Backtest First

Before risking real money, validate the strategy on historical data:

```bash
cd /home/user/Polybot
venv/bin/python backtest.py
```

This downloads 30 days of 1-second BTC ticks and replays ~8,600 five-minute windows through the full pipeline. Look for:

| Metric | Go-Live Threshold |
|--------|-------------------|
| Win Rate | > 55% |
| Average Edge | > 1% |
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < 15% |

Only proceed to live trading after confirming positive edge.

---

### Step 8: Go Live

Switch from test mode to live trading:

```bash
cd /home/user/Polybot
nano .env
# Change: TEST_MODE=false
# Save: Ctrl+X → Y → Enter

supervisorctl restart polybot
```

Check that the bot is running:
```bash
supervisorctl status polybot         # Should show RUNNING
tail -f /var/log/polybot/bot.log     # Live trading logs
```

---

### Step 9: Access the Dashboard

**From any browser on your network:**
```
http://YOUR_DROPLET_IP:8501
```

**From your phone or laptop anywhere (via ngrok):**
```bash
cd /home/user/Polybot
export NGROK_AUTHTOKEN=your_token    # Free at ngrok.com
venv/bin/python ngrok_integration.py
```
This prints a public HTTPS URL you can open on any device.

**With a custom domain (automatic HTTPS via Caddy):**

Edit `/etc/caddy/Caddyfile`:
```
your-domain.com {
    reverse_proxy localhost:8501
}
```
Then restart Caddy:
```bash
systemctl restart caddy
```
Point your domain's DNS A record to your droplet's IP. Caddy auto-provisions Let's Encrypt TLS.

---

## Daily Operations

### Common Commands

```bash
supervisorctl status polybot          # Check bot status
supervisorctl restart polybot         # Restart (picks up .env changes)
supervisorctl stop polybot            # Stop trading
tail -f /var/log/polybot/bot.log      # Watch live logs
tail -100 /var/log/polybot/bot.log    # Last 100 log lines
```

### Updating Configuration

1. Edit `.env` with `nano /home/user/Polybot/.env`
2. Restart: `supervisorctl restart polybot`

The bot reloads all settings from `.env` on restart. No code changes needed.

### Withdrawing Profits

Use the dashboard sidebar **Withdraw** section, or run:
```bash
cd /home/user/Polybot
venv/bin/python withdraw.py
```

---

## Architecture

Five concurrent async coroutines in a single process:

```
Bybit WebSocket → tick_queue → BayesianModel → Signal → OrderExecutor → CLOB
     ↑                              ↑                        ↑
  ccxt.pro                  Monte Carlo (1000 GBM)     Maker limit orders
  BTC/USDT:USDT              + Kelly criterion          midpoint - 0.01

GammaMarketFinder → market_queue → model.reset()
                                   (new 5-min window)

Stale Order Canceller → every 10s → frees locked capital
```

Target: **tick-to-order-decision in under 80ms** on a standard VPS.

## Live Balance — Everything is Dynamic

Every display, calculation, and decision uses the **real USDC balance from your Polygon wallet**, fetched live via web3:

- **Dashboard balance card**: live from `executor.get_current_balance()`
- **Equity curve chart**: every point is a live balance snapshot
- **Position sizing**: `_compute_order_size()` uses live balance minus safety floor
- **Drawdown calculation**: `(peak_balance - current_balance) / peak_balance` — all live
- **Withdraw max**: `state.balance - $1.00` buffer — updates every 5 seconds
- **Compounding gate**: scales with live balance after activation
- **Risk slider**: exposure percentage applied to live balance

## Configuration

All settings are in `config.py` with secrets in `.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `starting_capital` | $100 | Fallback for projections — live balance is always used |
| `min_exposure_pct` | 5% | Floor for position size as % of balance |
| `max_exposure_pct` | 10% | Cap for position size (adjustable via dashboard) |
| `min_edge_threshold` | 2% | Minimum net edge after fees to trade |
| `safety_floor_usdc` | $100 | Never trade below this balance |
| `drawdown_hard_stop_pct` | 25% | Halt if drawdown from peak exceeds this |
| `max_consecutive_losses` | 5 | Skip window after this many losses in a row |
| `round_trip_cost_pct` | 0.3% | Worst-case fee assumption (maker is lower) |
| `mc_paths` | 1000 | Monte Carlo simulation paths |

## Troubleshooting

### Bot Won't Start

```bash
# Check supervisor logs for error messages
supervisorctl tail polybot stderr

# Validate .env is configured correctly
cat .env | head -5

# Test config loading
cd /home/user/Polybot
venv/bin/python -c "from config import load_config; c = load_config(); print(f'Config OK: chain_id={c.chain_id}')"

# Check Python version
venv/bin/python --version   # Should be 3.11.x

# Check Redis
systemctl status redis-server
redis-cli ping              # Should return PONG
```

### .env Changes Not Taking Effect

The bot only reads `.env` at startup. After editing:
```bash
supervisorctl restart polybot
```

Make sure you're editing the right file (`/home/user/Polybot/.env`, not `.env.example`).

### Latency > 80ms

1. Verify NYC3 region — check RPC latency:
   ```bash
   curl -w "time_total: %{time_total}s\n" -o /dev/null -s \
     -X POST -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
     $ALCHEMY_RPC_URL
   ```
   Should show < 0.03s.

2. Check system load: `htop` — bot should use < 20% CPU.

3. Reduce MC paths: set `mc_paths` to 500 in config.py (saves ~2ms).

### Connection Issues

- **Bybit WebSocket disconnects**: Auto-reconnects with exponential backoff (up to 10 retries). Check logs for `BybitWS` entries.
- **CLOB API errors**: Usually transient. Bot skips the tick and retries next cycle.
- **Alchemy RPC rate limits**: Free tier allows 300 req/s. Upgrade if hitting limits.

### Drawdown Recovery

The bot auto-halts at 25% drawdown from peak. To recover:
1. Check logs: `tail -100 /var/log/polybot/bot.log | grep -i "drawdown\|halt"`
2. Run a quick backtest: `venv/bin/python backtest.py --days 1`
3. Resume via dashboard or restart: `supervisorctl restart polybot`

## File Structure

```
Polybot/
├── bot.py                 # Main orchestrator — 5 async coroutines
├── model.py               # Bayesian inference + Monte Carlo + Kelly
├── executor.py            # CLOB order execution (maker limits, batch orders)
├── data_feed.py           # Bybit WebSocket + Gamma market discovery
├── config.py              # Frozen dataclass configuration
├── dashboard.py           # Streamlit real-time dashboard (port 8501)
├── backtest.py            # Historical backtesting (standalone)
├── approve_usdc.py        # One-time USDC approval for exchange contracts
├── withdraw.py            # USDC withdrawal from Polygon wallet
├── performance.py         # Decay weights, adaptive z-score, rebate optimizer
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
- **File permissions**: `chmod 600 .env` prevents other users from reading your keys.
- **Supervisor isolation**: Bot runs as a managed process, not a background shell job.

## License

Private. Not for redistribution.
