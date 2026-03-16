"""
config.py — Central configuration for the Polymarket BTC 5-minute bot.

Loads secrets from .env and defines all runtime constants in a frozen dataclass.
Every parameter is tuned for positive expectancy on 5-minute binary BTC markets:
- Conservative Kelly sizing (quarter-Kelly cap) prevents ruin
- 0.3% round-trip cost assumption is worst-case; maker rebates reduce this
- Compounding only activates after statistical significance (200 trades)
"""

import logging
import os
import stat
from dataclasses import dataclass
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration. Frozen to prevent accidental mutation.

    All monetary values are in USDC. All probabilities are 0-1 floats.
    All time values are in seconds unless noted otherwise.
    """

    # --- Secrets (loaded from .env) ---
    private_key: str
    alchemy_rpc_url: str

    # --- Risk Parameters ---
    # Starting capital — the bot is designed for small initial stakes
    starting_capital: float = 100.0  # USDC

    # Per-window exposure as fraction of current balance.
    # Kelly output is clamped to this band to prevent oversizing on noisy signals.
    min_exposure_pct: float = 0.05  # 5% floor — ensures meaningful position sizes
    max_exposure_pct: float = 0.10  # 10% cap — limits single-window risk

    # Minimum edge (true_prob - implied_prob - costs) required to trade.
    # Below this threshold, signal-to-noise is too low for reliable profit.
    min_edge_threshold: float = 0.02  # 2% net edge after all fees

    # Target win rate — used for performance monitoring, not signal gating.
    # The Bayesian model + MC simulation determine actual trade signals.
    target_win_rate: float = 0.60  # 60%

    # Consecutive loss streak circuit breaker. If hit, skip current window
    # and wait for next market to avoid tilt-driven losses.
    max_consecutive_losses: int = 5

    # Compounding activates only after sufficient sample size proves edge is real.
    # Before activation, position sizes stay flat relative to starting capital.
    compounding_activation_trades: int = 200

    # --- Cost Constants ---
    # Gas buffer reserved per trade for Polygon transaction fees.
    # Polygon gas is cheap (~0.01 MATIC) but we pad for spikes.
    gas_buffer_usdc: float = 0.05

    # Maker rebate in basis points — Polymarket rewards passive liquidity.
    # We target maker fills by pricing orders inside the book.
    maker_rebate_bps: float = 0.0  # conservative: assume 0 until confirmed

    # Round-trip cost (taker fee both legs worst case).
    # Actual cost is lower with maker fills, giving hidden edge.
    round_trip_cost_pct: float = 0.003  # 0.3%

    # Never let balance drop below this floor. Preserves capital for recovery.
    safety_floor_usdc: float = 10.0

    # Hard stop: if drawdown from peak exceeds this, halt all trading.
    # Protects against regime changes or model breakdown.
    drawdown_hard_stop_pct: float = 0.25  # 25%

    # --- Polymarket CLOB Connection ---
    clob_host: str = "https://clob.polymarket.com"
    chain_id: int = 137  # Polygon mainnet
    signature_type: int = 0  # EOA wallet (standard private key signing)

    # --- Timing Parameters ---
    # How often to poll Gamma API for new 5-minute market windows
    market_poll_interval: int = 60  # seconds

    # Cancel unfilled orders after this duration to free up capital
    stale_order_cancel_interval: int = 10  # seconds

    # Don't enter positions too close to window expiry — insufficient
    # time for maker fill + price movement to realize edge
    min_remaining_window_secs: int = 10

    # --- Monte Carlo Simulation ---
    # 1000 paths balances accuracy vs latency. Vectorized numpy keeps this <5ms.
    mc_paths: int = 1000

    # Target computation budget for full model.evaluate() call
    mc_target_ms: int = 30

    # Per-tick volatility below which batch orders are used instead of single.
    # 0.0005 = std(returns) < 5 bps per tick — very calm market where a single
    # maker order at midpoint-0.01 may not fill. Spreading across multiple levels
    # increases fill probability and captures rebates on each.
    low_vol_threshold: float = 0.0005

    # --- Performance Tuning (performance.py) ---
    # Exponential decay factor for Bayesian tick weighting.
    # 0.9 means each tick older than the newest is weighted 10% less.
    decay_factor: float = 0.9

    # MC paths during high volatility (>2% per-tick std).
    # Doubling to 2000 improves tail-risk accuracy in volatile regimes.
    high_vol_mc_paths: int = 2000

    # Dollar depth threshold for rebate-optimized batch ordering.
    # When CLOB depth exceeds this, orders are placed at tighter offset
    # to maximize maker rebate classification probability.
    rebate_depth_threshold: float = 50000.0

    # Price offset from midpoint in rebate-optimized mode.
    # 0.005 (half a cent) is tighter than the normal 0.01, increasing
    # fill probability while still qualifying as passive maker orders.
    rebate_price_offset: float = 0.005

    # --- Backtesting ---
    # Number of days of historical 1-second BTC ticks to download for backtesting.
    # 30 days provides ~8,600 five-minute windows for statistically meaningful results.
    historical_data_days: int = 30

    # --- Polygon Contract Addresses ---
    # USDC.e (bridged) on Polygon — primary collateral for all Polymarket trading.
    # 6 decimals: 1 USDC = 1_000_000 wei units.
    usdc_token_address: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    usdc_decimals: int = 6

    # Polymarket CTF Exchange — needs USDC approval for standard market order matching.
    # This is the main exchange contract that escrows collateral when placing orders.
    ctf_exchange_address: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

    # Neg Risk CTF Exchange — needs USDC approval for multi-outcome / neg-risk markets.
    # BTC 5-min markets may use either exchange depending on market structure.
    neg_risk_ctf_exchange_address: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

    # Conditional Tokens (ERC1155) — Polymarket's token framework contract.
    # Handles split, merge, and redeem operations for conditional outcome tokens.
    conditional_tokens_address: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

    # Bridge API base URL for withdrawal operations
    bridge_api_base: str = "https://clob.polymarket.com"

    # Redis URL for Bayesian model state persistence across restarts (optional).
    # When set, the model saves alpha/beta/tick state to Redis on every update
    # and restores it on startup, preserving winning edge continuity.
    redis_url: str = ""

    # --- Alerting (optional SMTP) ---
    # When configured, the bot sends email alerts on drawdown warnings,
    # sustained latency breaches, and circuit breaker activations.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    alert_email: str = ""


def load_config() -> Config:
    """Factory function that reads .env and returns a validated Config.

    SECURITY: private_key is loaded here ONCE, passed only to ClobClient (EIP-712 signing)
    and dashboard process (for approve/withdraw). It is NEVER logged, printed, or transmitted.

    Raises ValueError if required secrets are missing. This fails fast
    at startup rather than mid-trade.
    """
    _logger = logging.getLogger("Config")

    # Check .env file permissions — warn if readable by group/others
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        mode = os.stat(env_path).st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            _logger.warning(
                "SECURITY: .env is readable by group/others. "
                "Run: chmod 600 .env"
            )

    private_key: str = os.getenv("POLYGON_PRIVATE_KEY", "")
    rpc_url: str = os.getenv("ALCHEMY_RPC_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_pass: str = os.getenv("SMTP_PASS", "")
    alert_email: str = os.getenv("ALERT_EMAIL", "")

    if not private_key or private_key == "0xYOUR_PRIVATE_KEY_HERE":
        raise ValueError(
            "POLYGON_PRIVATE_KEY must be set in .env — "
            "copy .env.example to .env and fill in your key"
        )
    if not rpc_url or "YOUR_API_KEY" in rpc_url:
        raise ValueError(
            "ALCHEMY_RPC_URL must be set in .env — "
            "get a free key at https://dashboard.alchemy.com"
        )

    # Validate contract address checksums (EIP-55) to catch tampering or copy-paste errors
    for name, addr in [
        ("usdc_token_address", Config.usdc_token_address),
        ("ctf_exchange_address", Config.ctf_exchange_address),
        ("neg_risk_ctf_exchange_address", Config.neg_risk_ctf_exchange_address),
        ("conditional_tokens_address", Config.conditional_tokens_address),
    ]:
        expected = Web3.to_checksum_address(addr)
        if addr != expected:
            raise ValueError(
                f"Contract address checksum mismatch for {name}: "
                f"got {addr}, expected {expected}"
            )

    return Config(
        private_key=private_key,
        alchemy_rpc_url=rpc_url,
        redis_url=redis_url,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_pass=smtp_pass,
        alert_email=alert_email,
    )
