"""
performance.py — Advanced performance and latency fine-tuning for PolyBot.

Four optimization modules:
1. latency_monitor: Decorator timing tick-to-order cycles, auto-adjusts poll interval
2. volatility_clustering: Exponential decay weighting for Bayesian model + dynamic MC paths
3. rebate_optimizer: Intelligent batch ordering based on CLOB liquidity depth
4. adaptive_z_score: Dynamic min-edge threshold based on real-time order book depth
"""

import functools
import logging
import time
from collections import deque
from typing import Deque, Optional


# ---------------------------------------------------------------------------
# 1. Latency Monitor — Decorator + poll interval auto-tuning
# ---------------------------------------------------------------------------

class LatencyMonitor:
    """Tracks tick-to-order cycle latency and auto-adjusts poll interval.

    Usage:
        monitor = LatencyMonitor(logger)
        # In the trade loop, wrap timing:
        monitor.record(cycle_ms)
        monitor.auto_adjust_poll_interval(volatility)
    """

    WARN_THRESHOLD_MS: float = 80.0  # Log warning if cycle exceeds this
    HIGH_VOL_THRESHOLD: float = 0.02  # 2% per-tick vol = high volatility regime
    MIN_POLL_INTERVAL: float = 0.5    # Fastest poll interval during high-vol
    DEFAULT_POLL_INTERVAL: float = 1.0  # Normal poll interval

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger: logging.Logger = logger or logging.getLogger("LatencyMonitor")
        self.cycle_times: Deque[float] = deque(maxlen=100)
        self._current_poll_interval: float = self.DEFAULT_POLL_INTERVAL

    def record(self, cycle_ms: float) -> None:
        """Record a cycle time and log warning if it exceeds 80ms."""
        self.cycle_times.append(cycle_ms)
        if cycle_ms > self.WARN_THRESHOLD_MS:
            self.logger.warning(
                f"Slow cycle: {cycle_ms:.1f}ms (threshold={self.WARN_THRESHOLD_MS}ms)"
            )

    @property
    def avg_ms(self) -> float:
        """Rolling average cycle time in milliseconds."""
        if not self.cycle_times:
            return 0.0
        return sum(self.cycle_times) / len(self.cycle_times)

    @property
    def is_high_latency(self) -> bool:
        """True if average cycle time exceeds 60ms target."""
        return self.avg_ms > 60.0

    @property
    def poll_interval(self) -> float:
        """Current auto-adjusted poll interval."""
        return self._current_poll_interval

    def auto_adjust_poll_interval(self, volatility: float) -> float:
        """Auto-adjust poll interval based on current volatility regime.

        During high-vol (>2% per-tick), reduce poll interval to 0.5s
        to capture faster-moving signals. Revert to 1.0s in calm markets.

        Args:
            volatility: Current per-tick volatility from BayesianModel

        Returns:
            The adjusted poll interval in seconds
        """
        if volatility > self.HIGH_VOL_THRESHOLD:
            self._current_poll_interval = self.MIN_POLL_INTERVAL
            self.logger.debug(
                f"High-vol regime (vol={volatility:.4f}): "
                f"poll interval → {self.MIN_POLL_INTERVAL}s"
            )
        else:
            self._current_poll_interval = self.DEFAULT_POLL_INTERVAL
        return self._current_poll_interval


def latency_monitor(logger: Optional[logging.Logger] = None):
    """Decorator factory that times async function calls and logs slow cycles.

    Usage:
        @latency_monitor(logger)
        async def process_tick(...):
            ...
    """
    monitor = LatencyMonitor(logger)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            monitor.record(elapsed_ms)
            return result
        wrapper._monitor = monitor
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 2. Volatility Clustering — Exponential decay tick weighting + dynamic MC
# ---------------------------------------------------------------------------

def apply_decay_weights(
    tick_directions: list,
    decay_factor: float = 0.9,
) -> tuple:
    """Recompute alpha/beta from tick direction history using exponential decay.

    Recent ticks are weighted heavier than older ticks:
        weight(i) = decay_factor^i  where i=0 is the most recent tick

    This captures volatility clustering — recent momentum is more predictive
    of near-term direction than stale observations from window start.

    Args:
        tick_directions: List of +1 (up), -1 (down), 0 (flat) in chronological order
        decay_factor: Exponential decay rate (0.9 = 10% decay per tick)

    Returns:
        Tuple of (alpha, beta) with decay-weighted counts + 1.0 base prior
    """
    alpha: float = 1.0  # base prior
    beta: float = 1.0

    n = len(tick_directions)
    for i, direction in enumerate(tick_directions):
        # i=0 is oldest, i=n-1 is newest
        # weight for position i: decay^(n-1-i), so newest gets weight 1.0
        weight = decay_factor ** (n - 1 - i)
        if direction == 1:
            alpha += weight
        elif direction == -1:
            beta += weight

    return (alpha, beta)


def get_dynamic_mc_paths(volatility: float, base_paths: int = 1000, high_vol_paths: int = 2000) -> int:
    """Return MC path count based on current volatility regime.

    During high volatility (>2% per-tick), double MC paths to 2000
    for more accurate tail-risk estimation. In calm markets, 1000
    paths suffice and keep latency under 5ms.

    Args:
        volatility: Current per-tick vol from BayesianModel
        base_paths: Default MC path count (1000)
        high_vol_paths: Elevated count during high-vol (2000)

    Returns:
        Number of MC simulation paths to run
    """
    if volatility > 0.02:
        return high_vol_paths
    return base_paths


# ---------------------------------------------------------------------------
# 3. Rebate Optimizer — Liquidity-aware batch ordering
# ---------------------------------------------------------------------------

def compute_dollar_depth(order_book: Optional[dict]) -> float:
    """Compute total dollar depth from CLOB order book.

    Sums (price * size) across all bid and ask levels to get
    the total dollar liquidity available in the book.

    Args:
        order_book: Dict with 'bids' and 'asks' arrays

    Returns:
        Total dollar depth, or 0.0 if order_book is None/invalid
    """
    if order_book is None:
        return 0.0
    try:
        total: float = 0.0
        for side in ("bids", "asks"):
            for level in order_book.get(side, []):
                if isinstance(level, dict):
                    price = float(level.get("price", level.get("p", 0)))
                    size = float(level.get("size", level.get("s", 0)))
                else:
                    price = float(level[0])
                    size = float(level[1])
                total += price * size
        return total
    except Exception:
        return 0.0


def rebate_optimizer(
    order_book: Optional[dict],
    midpoint: float,
    depth_threshold: float = 50000.0,
    rebate_offset: float = 0.005,
) -> dict:
    """Determine optimal order placement strategy based on CLOB liquidity.

    When liquidity depth exceeds $50k, the book is thick enough to batch
    3 limit orders at a wider offset (0.005 instead of 0.01). The slightly
    worse price maximizes maker classification probability — orders further
    from mid are more likely to be classified as passive, earning rebates.

    In thin books, stick to the standard single-order strategy at -0.01
    to prioritize fill probability over rebate capture.

    Args:
        order_book: CLOB L2 order book dict
        midpoint: Current CLOB midpoint price
        depth_threshold: Dollar depth threshold for rebate mode ($50k)
        rebate_offset: Price offset from midpoint in rebate mode (0.005)

    Returns:
        Dict with keys:
            use_rebate_batch: bool — whether to use rebate-optimized batching
            price_offset: float — offset from midpoint for order pricing
            levels: int — number of order levels
            dollar_depth: float — computed dollar depth for logging
    """
    dollar_depth: float = compute_dollar_depth(order_book)

    if dollar_depth > depth_threshold:
        return {
            "use_rebate_batch": True,
            "price_offset": rebate_offset,
            "levels": 3,
            "dollar_depth": dollar_depth,
        }
    return {
        "use_rebate_batch": False,
        "price_offset": 0.01,
        "levels": 1,
        "dollar_depth": dollar_depth,
    }


# ---------------------------------------------------------------------------
# 4. Adaptive Z-Score — Dynamic min-edge based on real-time liquidity
# ---------------------------------------------------------------------------

def adaptive_z_score(
    order_book: Optional[dict],
    base_min_edge: float = 0.02,
    high_depth_threshold: float = 50000.0,
    low_depth_threshold: float = 10000.0,
) -> float:
    """Dynamically adjust minimum edge threshold based on CLOB liquidity.

    Deep order books indicate stable markets where smaller edges are
    more likely to be real (not noise from thin book manipulation).
    We lower the entry bar to capture more trades in liquid conditions.

    Shallow books require a higher bar because:
    - Small orders move the midpoint → noisy implied probability
    - Wider spreads increase effective round-trip cost
    - Fill probability drops for maker orders

    Scaling:
    - Depth > $50k: min_edge *= 0.5 (1% threshold — aggressive)
    - Depth $10k-$50k: linear interpolation between 0.5x and 1.0x
    - Depth < $10k: min_edge *= 1.0 (full 2% — conservative)

    Args:
        order_book: CLOB L2 order book dict
        base_min_edge: Default minimum edge threshold (0.02 = 2%)
        high_depth_threshold: Dollar depth above which edge is halved
        low_depth_threshold: Dollar depth below which full edge required

    Returns:
        Adjusted minimum edge threshold
    """
    dollar_depth: float = compute_dollar_depth(order_book)

    if dollar_depth >= high_depth_threshold:
        return base_min_edge * 0.5

    if dollar_depth <= low_depth_threshold:
        return base_min_edge

    # Linear interpolation between low_depth (1.0x) and high_depth (0.5x)
    t: float = (dollar_depth - low_depth_threshold) / (
        high_depth_threshold - low_depth_threshold
    )
    multiplier: float = 1.0 - t * 0.5  # 1.0 → 0.5
    return base_min_edge * multiplier
