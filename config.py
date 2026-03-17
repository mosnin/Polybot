"""
config.py — Central configuration for the Polymarket BTC 5-minute bot.

Loads secrets from .env and defines all runtime constants in a frozen dataclass.
Explosive compounding mode: full Kelly sizing, tighter spread hunting, instant
compounding, and 15% per-side MM exposure for maximum growth velocity.
"""

import logging
import os
import stat
from dataclasses import dataclass
from dotenv import load_dotenv
from web3 import Web3

load_dotenv(override=True)


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
    max_exposure_pct: float = 0.15  # 15% cap — aggressive per-window sizing

    # Minimum edge (true_prob - implied_prob - costs) required to trade.
    # Below this threshold, signal-to-noise is too low for reliable profit.
    min_edge_threshold: float = 0.04  # 4% net edge after all fees — filters noise

    # Target win rate — used for performance monitoring, not signal gating.
    # The Bayesian model + MC simulation determine actual trade signals.
    target_win_rate: float = 0.60  # 60%

    # Consecutive loss streak circuit breaker. Set to 999 to effectively disable —
    # explosive mode accepts variance for maximum compounding.
    max_consecutive_losses: int = 999

    # Instant compounding: 0 = activate from first trade, no waiting period.
    compounding_activation_trades: int = 0

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
    # 40% accepts higher variance for explosive upside.
    drawdown_hard_stop_pct: float = 0.40  # 40%

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

    # --- Market-Making Mode (Explosive) ---
    # Tighter spread threshold hunts more opportunities. Batch orders (up to 5/side)
    # at staggered levels maximize fill probability. 15% per side for aggressive sizing.
    mm_enabled: bool = True
    mm_spread_threshold: float = 0.985    # tighter than 0.99 — hunt more spreads
    mm_exposure_pct: float = 0.15         # 15% of balance PER SIDE (30% total)
    mm_cancel_delay_secs: float = 30.0    # faster decisions — 30s not 45s
    mm_check_interval_secs: float = 2.0   # check every 2s for maximum responsiveness
    mm_min_remaining_secs: float = 60.0   # enter closer to expiry
    mm_batch_size: int = 5                # up to 5 simultaneous orders per side

    # --- Order Flow Imbalance ---
    # When bid/ask depth delta > threshold, ride the dominant side harder.
    orderflow_imbalance_threshold: float = 0.20  # 20% delta triggers side bias

    # --- Stoikov Reservation Price Engine ---
    # Avellaneda-Stoikov: r = s - q * gamma * sigma^2 * (T-t)
    # gamma controls risk aversion (higher = tighter quotes, lower inventory risk)
    stoikov_gamma: float = 0.15
    stoikov_check_interval_secs: float = 1.5  # run Stoikov calc every 1.5s

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

    # --- Resolution Sniper ---
    # Late-window strategy: in the last N seconds, BTC direction is nearly decided.
    # The CLOB still has shares at 0.70-0.85 when they should be 0.95+.
    # We buy near-certain outcomes at a discount. This is the highest win-rate edge.
    sniper_enabled: bool = True
    sniper_window_secs: float = 60.0      # activate in last 60 seconds of window
    sniper_min_confidence: float = 0.80    # minimum P(direction) from Bayesian model
    sniper_min_price_discount: float = 0.05  # only buy if CLOB price is 5%+ below our estimate
    sniper_max_exposure_pct: float = 0.20  # can go bigger on high-confidence snipes (20%)
    sniper_min_ticks: int = 15             # need at least 15 ticks of data before sniping

    # --- Window Selector ---
    # Not all 5-min windows have edge. Filter for high-volatility sessions where
    # Polymarket lags hardest. Skip dead zones where spreads are tight and edge is noise.
    window_selector_enabled: bool = True
    # High-edge hours (UTC). These correspond to major market opens and overlap periods:
    # 8-10 = London open, 13-16 = US open + overlap, 0-2 = Asia session open
    high_edge_hours_utc: str = "0,1,2,8,9,10,13,14,15,16"
    # Minimum per-tick volatility to trade outside high-edge hours.
    # During "dead" hours, only trade if vol is spiking (news event, flash crash).
    min_vol_for_off_hours: float = 0.005   # 0.5% per-tick vol = something is happening
    # Minimum BTC price change (%) since window open to consider the window tradeable.
    # Flat windows = no directional edge to capture.
    min_window_move_pct: float = 0.03      # 0.03% = ~$20 on $67k BTC

    # --- Multi-Timeframe Momentum ---
    # EMA spans (in ticks) for multi-timeframe confirmation.
    # Fast (~30s), medium (~1.5m), slow (~5m) at ~1 tick/3sec.
    mtf_fast_span: int = 10       # ~30 seconds of ticks
    mtf_medium_span: int = 30     # ~1.5 minutes
    mtf_slow_span: int = 60       # ~3 minutes — initializes within 80% of window
    # Minimum agreement score (0-1) across timeframes to confirm signal.
    # 1.0 = all three must agree. 0.66 = at least 2 of 3.
    mtf_min_agreement: float = 1.0

    # --- Volatility-Adjusted Sizing ---
    # Scale Kelly fraction by inverse normalized volatility.
    # High vol → smaller size (uncertainty), low vol + signal → bigger size.
    vol_sizing_enabled: bool = True
    vol_sizing_lookback: int = 50   # ticks for baseline vol estimation
    vol_sizing_floor: float = 0.3   # minimum scaling factor (never below 30% of Kelly)
    vol_sizing_ceiling: float = 1.5  # maximum scaling factor (cap at 150% of Kelly)

    # --- Mean Reversion Detection ---
    # Detect when model-vs-market spread is abnormally wide (opportunity)
    # or narrow (noise). Uses rolling z-score of the spread history.
    mean_reversion_enabled: bool = True
    spread_history_maxlen: int = 200  # rolling window for spread z-score
    mean_reversion_z_threshold: float = 1.5  # spread z > this = mean reversion opportunity
    mean_reversion_boost: float = 1.3  # multiply Kelly by this when MR detected

    # --- Performance Tuning (performance.py) ---
    # Exponential decay factor for Bayesian tick weighting.
    # 0.65 balances recent momentum with overall window trend.
    # Higher values (0.9) cause over-fitting to last few ticks → mean reversion losses.
    decay_factor: float = 0.65

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

    # Market-making overrides from .env (explosive defaults)
    mm_enabled: bool = os.getenv("MM_ENABLED", "true").lower() == "true"
    mm_spread_threshold: float = float(os.getenv("MM_SPREAD_THRESHOLD", "0.985"))
    mm_exposure_pct: float = float(os.getenv("MM_EXPOSURE_PCT", "0.15"))
    mm_cancel_delay_secs: float = float(os.getenv("MM_CANCEL_DELAY_SECS", "30.0"))
    mm_batch_size: int = int(os.getenv("MM_BATCH_SIZE", "5"))
    orderflow_imbalance_threshold: float = float(
        os.getenv("ORDERFLOW_IMBALANCE_THRESHOLD", "0.20")
    )
    stoikov_gamma: float = float(os.getenv("STOIKOV_GAMMA", "0.15"))
    starting_capital: float = float(os.getenv("STARTING_CAPITAL", "100.0"))

    return Config(
        private_key=private_key,
        alchemy_rpc_url=rpc_url,
        redis_url=redis_url,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_pass=smtp_pass,
        alert_email=alert_email,
        mm_enabled=mm_enabled,
        mm_spread_threshold=mm_spread_threshold,
        mm_exposure_pct=mm_exposure_pct,
        mm_cancel_delay_secs=mm_cancel_delay_secs,
        mm_batch_size=mm_batch_size,
        orderflow_imbalance_threshold=orderflow_imbalance_threshold,
        stoikov_gamma=stoikov_gamma,
        starting_capital=starting_capital,
    )
