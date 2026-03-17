#!/usr/bin/env bash
# =============================================================================
# setup.sh — DigitalOcean VPS deployment for Polymarket BTC trading bot.
#
# Paths are detected automatically from this script's location.
# Works regardless of where the repo is cloned (/root/polybot, /home/user/Polybot, etc.)
#
# Target: Fresh Ubuntu 22.04/24.04 droplet on DigitalOcean (NYC3 recommended).
#
# Usage:
#     chmod +x setup.sh
#     sudo ./setup.sh
#
# After setup:
#     supervisorctl status          — check bot status
#     tail -f /var/log/polybot/bot.log  — watch live logs
#     supervisorctl restart polybot — restart after config change
# =============================================================================
set -euo pipefail

# --- Dynamic path detection ---
# POLYBOT_DIR = wherever this script lives. No hardcoded paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLYBOT_DIR="${SCRIPT_DIR}"
VENV_DIR="${POLYBOT_DIR}/venv"
LOG_DIR="/var/log/polybot"

echo "============================================================"
echo "  Polybot — Automated Deployment"
echo "============================================================"
echo ""
echo "  Detected project directory: ${POLYBOT_DIR}"
echo ""

# ---- Phase 1: System Packages ----
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
"${VENV_DIR}/bin/pip" install --quiet pyngrok

echo "  Python venv ready at ${VENV_DIR}"

# ---- Phase 4: Directory Setup ----
echo "[4/8] Creating log directory..."
mkdir -p "${LOG_DIR}"
echo "  Logs → ${LOG_DIR}"

# ---- Phase 5: .env Validation ----
echo "[5/8] Checking .env configuration..."
if [ ! -f "${POLYBOT_DIR}/.env" ]; then
    cp "${POLYBOT_DIR}/.env.example" "${POLYBOT_DIR}/.env"
    chmod 600 "${POLYBOT_DIR}/.env"
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║  ACTION REQUIRED: Configure your .env file              ║"
    echo "  ║                                                         ║"
    echo "  ║  A template has been created at:                        ║"
    echo "  ║  ${POLYBOT_DIR}/.env"
    echo "  ║                                                         ║"
    echo "  ║  EASIEST METHOD (paste-friendly, no editor needed):     ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Run this command (replace the placeholder values):"
    echo ""
    echo "  cat > ${POLYBOT_DIR}/.env << 'EOF'"
    echo "  POLYGON_PRIVATE_KEY=0xYourActualPrivateKeyHere"
    echo "  ALCHEMY_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YourActualApiKey"
    echo "  STARTING_CAPITAL=100"
    echo "  TEST_MODE=true"
    echo "  ENABLE_DASHBOARD=true"
 echo "  SMTP_HOST="
    echo "  SMTP_PORT=587"
    echo "  SMTP_USER="
    echo "  SMTP_PASS="
    echo "  ALERT_EMAIL="
    echo "  NGROK_AUTHTOKEN="
    echo "  EOF"
    echo ""
    echo "  Then re-run: sudo ./setup.sh"
    echo ""
    exit 1
fi

# Robust placeholder check — extract only the VALUE after the = sign,
# ignoring comments and whitespace. Prevents false positives.
PKEY_VAL=$(grep -E '^POLYGON_PRIVATE_KEY=' "${POLYBOT_DIR}/.env" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d '[:space:]')
RPC_VAL=$(grep -E '^ALCHEMY_RPC_URL=' "${POLYBOT_DIR}/.env" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d '[:space:]')

if [ -z "$PKEY_VAL" ] || [ "$PKEY_VAL" = "0xYOUR_PRIVATE_KEY_HERE" ]; then
    echo ""
    echo "  ERROR: POLYGON_PRIVATE_KEY is not set (still placeholder or empty)."
    echo ""
    echo "  Edit ${POLYBOT_DIR}/.env with your real private key."
    echo "  Or use the cat method shown above."
    echo "  Then re-run: sudo ./setup.sh"
    echo ""
    exit 1
fi

if [ -z "$RPC_VAL" ] || echo "$RPC_VAL" | grep -q "YOUR_API_KEY"; then
    echo ""
    echo "  ERROR: ALCHEMY_RPC_URL is not set (still placeholder or empty)."
    echo ""
    echo "  Get a free key at https://dashboard.alchemy.com"
    echo "  Edit ${POLYBOT_DIR}/.env with your real RPC URL."
    echo "  Then re-run: sudo ./setup.sh"
    echo ""
    exit 1
fi

echo "  .env validated — keys are set."

# ---- Phase 6: One-Time USDC Approval ----
echo "[6/8] Running USDC approval..."
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

# ---- Phase 7: Supervisor Setup ----
echo "[7/8] Configuring Supervisor..."

# Rewrite template paths to match this deployment's actual location
sed -e "s|/home/user/Polybot|${POLYBOT_DIR}|g" \
    "${POLYBOT_DIR}/supervisor.conf" > /etc/supervisor/conf.d/polybot.conf

# Ensure supervisor service is running (fixes "socket missing" errors)
systemctl enable supervisor > /dev/null 2>&1 || true
systemctl restart supervisor > /dev/null 2>&1 || true
sleep 2

supervisorctl reread > /dev/null 2>&1
supervisorctl update > /dev/null 2>&1
supervisorctl start polybot > /dev/null 2>&1 || true

echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Project:  ${POLYBOT_DIR}"
echo "  Venv:     ${VENV_DIR}"
echo "  Logs:     ${LOG_DIR}"
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

# Try to detect server IP for dashboard URL
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "YOUR_SERVER_IP")
echo "  Dashboard: http://${SERVER_IP}:8501"
echo ""
echo "  For remote HTTPS access (phone/laptop):"
echo "    ${VENV_DIR}/bin/python ${POLYBOT_DIR}/ngrok_integration.py"
echo ""
echo "============================================================"
echo "  Bot is running. You can close this terminal."
echo "============================================================"
