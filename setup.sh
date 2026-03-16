#!/usr/bin/env bash
# =============================================================================
# setup.sh — DigitalOcean NYC3 VPS deployment for Polymarket BTC trading bot.
#
# Optimized for sub-30ms Polygon RPC latency from NYC3 datacenter.
# Installs all dependencies, configures process supervision, and starts
# the bot with zero terminal dependency after initial setup.
#
# Target: Fresh Ubuntu 22.04/24.04 droplet on DigitalOcean.
# Execute from: DigitalOcean browser console or SSH session.
#
# Usage:
#     chmod +x setup.sh
#     sudo ./setup.sh
#
# After setup:
#     supervisorctl status          — check bot status
#     tail -f /var/log/polybot/bot.log  — watch live logs
#     supervisorctl restart polybot — restart after config change
#
# The bot runs under supervisor and auto-restarts on crash.
# No terminal or SSH session required after initial setup.
# =============================================================================
set -euo pipefail

POLYBOT_DIR="/home/user/Polybot"
VENV_DIR="${POLYBOT_DIR}/venv"
LOG_DIR="/var/log/polybot"

echo "============================================================"
echo "  Polybot — DigitalOcean NYC3 Deployment"
echo "============================================================"
echo ""

# ---- Phase 1: System Packages ----
# Note for DigitalOcean browser console: this may take 2-3 minutes.
# The console may appear frozen during apt operations — this is normal.
echo "[1/8] Installing system packages..."
apt update -qq
apt install -y -qq software-properties-common > /dev/null 2>&1

# Python 3.11 from deadsnakes PPA (Ubuntu 22.04 ships 3.10)
add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
apt update -qq

apt install -y -qq \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    supervisor \
    redis-server \
    curl \
    > /dev/null 2>&1

echo "  System packages installed."

# ---- Phase 2: Caddy Web Server (for HTTPS reverse proxy) ----
echo "[2/8] Installing Caddy..."
if ! command -v caddy &> /dev/null; then
    apt install -y -qq debian-keyring debian-archive-keyring apt-transport-https > /dev/null 2>&1
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg 2>/dev/null
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list > /dev/null
    apt update -qq
    apt install -y -qq caddy > /dev/null 2>&1
    echo "  Caddy installed."
else
    echo "  Caddy already installed."
fi

# ---- Phase 3: Python Virtual Environment ----
echo "[3/8] Setting up Python virtual environment..."
python3.11 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip

# Install all project dependencies
"${VENV_DIR}/bin/pip" install --quiet -r "${POLYBOT_DIR}/requirements.txt"

# Additional deployment dependencies
"${VENV_DIR}/bin/pip" install --quiet redis pyngrok

echo "  Python venv ready at ${VENV_DIR}"

# ---- Phase 4: Directory Setup ----
echo "[4/8] Creating log directory..."
mkdir -p "${LOG_DIR}"
echo "  Logs → ${LOG_DIR}"

# ---- Phase 5: .env Validation ----
echo "[5/8] Checking .env configuration..."
if [ ! -f "${POLYBOT_DIR}/.env" ]; then
    cp "${POLYBOT_DIR}/.env.example" "${POLYBOT_DIR}/.env"
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║  ACTION REQUIRED: Upload your .env file                 ║"
    echo "  ║                                                         ║"
    echo "  ║  A template has been created at:                        ║"
    echo "  ║  ${POLYBOT_DIR}/.env                      ║"
    echo "  ║                                                         ║"
    echo "  ║  Edit it with your real keys:                           ║"
    echo "  ║    nano ${POLYBOT_DIR}/.env                ║"
    echo "  ║                                                         ║"
    echo "  ║  Required:                                              ║"
    echo "  ║    POLYGON_PRIVATE_KEY=0xYourActualKey                  ║"
    echo "  ║    ALCHEMY_RPC_URL=https://polygon-mainnet.g.alchemy.. ║"
    echo "  ║                                                         ║"
    echo "  ║  Optional:                                              ║"
    echo "  ║    REDIS_URL=redis://localhost:6379/0                   ║"
    echo "  ║    NGROK_AUTHTOKEN=your_ngrok_token                    ║"
    echo "  ║                                                         ║"
    echo "  ║  Then re-run: sudo ./setup.sh                           ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi

# Validate that .env has been customized (not still the template values)
if grep -q "0xYOUR_PRIVATE_KEY_HERE" "${POLYBOT_DIR}/.env"; then
    echo ""
    echo "  ERROR: .env still contains placeholder values."
    echo "  Edit ${POLYBOT_DIR}/.env with your real keys."
    echo "  Then re-run: sudo ./setup.sh"
    echo ""
    exit 1
fi
echo "  .env validated."

# ---- Phase 6: Redis Setup ----
echo "[6/8] Configuring Redis..."
systemctl enable redis-server > /dev/null 2>&1
systemctl start redis-server
# Verify Redis is running
if redis-cli ping > /dev/null 2>&1; then
    echo "  Redis running on localhost:6379"
else
    echo "  WARNING: Redis not responding. Model state persistence disabled."
fi

# ---- Phase 7: One-Time USDC Approval ----
echo "[7/8] Running USDC approval..."
echo "  This approves the Polymarket exchange contracts to trade USDC."
echo "  (One-time operation — costs ~$0.01 in gas)"
echo ""

# Run approval script — it checks existing allowance and skips if already approved
cd "${POLYBOT_DIR}"
"${VENV_DIR}/bin/python" approve_usdc.py <<< "y" || {
    echo ""
    echo "  WARNING: USDC approval failed. You can run it manually later:"
    echo "    cd ${POLYBOT_DIR} && ${VENV_DIR}/bin/python approve_usdc.py"
    echo ""
}

# ---- Phase 8: Supervisor Setup ----
echo "[8/8] Configuring Supervisor..."
cp "${POLYBOT_DIR}/supervisor.conf" /etc/supervisor/conf.d/polybot.conf
supervisorctl reread > /dev/null 2>&1
supervisorctl update > /dev/null 2>&1
supervisorctl start polybot > /dev/null 2>&1 || true

echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Bot Status:"
supervisorctl status polybot 2>/dev/null || echo "    (starting...)"
echo ""
echo "  Commands:"
echo "    supervisorctl status          — check status"
echo "    supervisorctl restart polybot — restart bot"
echo "    supervisorctl stop polybot    — stop bot"
echo "    tail -f ${LOG_DIR}/bot.log    — live logs"
echo ""
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "  For HTTPS access from your phone (optional):"
echo "    ${VENV_DIR}/bin/python ${POLYBOT_DIR}/ngrok_integration.py"
echo ""

# ---- Caddy Config Snippet ----
echo "  ── Caddy HTTPS Reverse Proxy (optional) ──"
echo ""
echo "  To serve the dashboard over HTTPS with a domain, add this to"
echo "  /etc/caddy/Caddyfile and run: systemctl restart caddy"
echo ""
echo "    your-domain.com {"
echo "        reverse_proxy localhost:8501"
echo "    }"
echo ""
echo "  Caddy auto-provisions Let's Encrypt TLS certificates."
echo "  Point your domain's DNS A record to this server's IP first."
echo ""
echo "============================================================"
echo "  Bot is running. You can close this terminal."
echo "============================================================"
