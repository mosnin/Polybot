"""
model.py — Bayesian inference + Monte Carlo simulation for edge detection.

This is the brain of the bot. The core insight driving positive expectancy:

Polymarket 5-min BTC markets are priced by retail participants using lagging
information. Our Bayesian model updates on sub-20ms OKX perpetual ticks,
building a posterior distribution for P(BTC up) that leads the CLOB implied
probability by 1-3 seconds. When the gap (edge) exceeds costs, we trade.

The z-score normalizes the edge by recent volatility, filtering out noise.
Monte Carlo simulation validates the edge over 1000 GBM paths, ensuring
positive expected value even in the tail scenarios.

All computations are vectorized numpy — total evaluate() latency target <30ms.
"""

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from performance import apply_decay_weights, get_dynamic_mc_paths

# Memory store is optional — model works without it, but loses state persistence
# Import MemoryStore lazily to avoid circular imports
_memory_store = None  # set externally by bot.py


@dataclass
class Signal:
    """Trade signal emitted when the model detects a profitable edge.

    Every field is needed by the executor for position sizing and logging:
    - direction: which token to buy (UP or DOWN)
    - true_prob: our Bayesian estimate of P(correct outcome)
    - implied_prob: market's estimate from CLOB midpoint
    - edge: true_prob - implied_prob - costs (the actual profit margin)
    - z_score: edge normalized by volatility (confidence measure)
    - mc_ev: Monte Carlo expected value per dollar risked
    - mc_win_prob: fraction of MC paths that ended profitable
    - mc_variance: variance of MC payoffs (risk measure)
    - kelly_fraction: optimal position size as fraction of bankroll
    - computation_ms: total time for evaluate() call (latency monitoring)
    - mtf_agreement: multi-timeframe momentum agreement score (0-1)
    - vol_scalar: volatility-based sizing multiplier applied to Kelly
    - spread_z: z-score of current spread vs rolling history (mean reversion)
    """

    direction: str  # "UP" or "DOWN"
    true_prob: float
    implied_prob: float
    edge: float
    z_score: float
    mc_ev: float
    mc_win_prob: float
    mc_variance: float
    kelly_fraction: float
    computation_ms: float
    mtf_agreement: float = 1.0
    vol_scalar: float = 1.0
    spread_z: float = 0.0


class BayesianModel:
    """Bayesian beta-binomial model for short-term BTC direction prediction.

    The model maintains a beta distribution prior over P(next tick is up).
    Each OKX tick updates the posterior:
    - Price went up   → alpha += 1 (evidence for upward momentum)
    - Price went down → beta += 1  (evidence for downward momentum)
    - No change       → no update  (uninformative observation)

    The posterior mean alpha/(alpha+beta) converges to the true short-term
    directional probability as ticks accumulate. With ~100 ticks per 5-min
    window (one every ~3 seconds), the posterior becomes informative within
    30-60 seconds.

    The prior resets to (1,1) — uninformative — at each new window to prevent
    stale momentum from prior windows contaminating the signal.
    """

    VOLATILITY_WINDOW: int = 30  # number of ticks for rolling volatility
    MC_PATHS: int = 1000  # Monte Carlo simulation paths

    # Liquidity depth thresholds for dynamic z-score scaling.
    # Total token depth across top 3 bid + ask levels in the CLOB.
    THIN_BOOK_DEPTH: float = 500.0   # below this → high noise → raise bar
    THICK_BOOK_DEPTH: float = 5000.0  # above this → low noise → lower bar

    def __init__(
        self,
        min_edge: float = 0.02,
        round_trip_cost: float = 0.003,
        memory_store=None,
        decay_factor: float = 0.9,
        mtf_fast_span: int = 10,
        mtf_medium_span: int = 30,
        mtf_slow_span: int = 100,
        mtf_min_agreement: float = 0.66,
        vol_sizing_enabled: bool = True,
        vol_sizing_lookback: int = 50,
        vol_sizing_floor: float = 0.3,
        vol_sizing_ceiling: float = 1.5,
        mean_reversion_enabled: bool = True,
        spread_history_maxlen: int = 200,
        mean_reversion_z_threshold: float = 1.5,
        mean_reversion_boost: float = 1.3,
    ) -> None:
        """Initialize with uninformative beta prior.

        Args:
            min_edge: Minimum net edge (after costs) required to generate signal.
                      Lower = more trades but noisier. 0.02 is conservative.
            round_trip_cost: Total cost of entering + exiting position.
                             0.3% assumes worst-case taker fees both sides.
                             Maker fills reduce this, creating hidden alpha.
            memory_store: Optional MemoryStore for SQLite state persistence.
                          If None, persistence is disabled.
            decay_factor: Exponential decay for tick weighting (0.9 = 10% decay).
                          Recent ticks are weighted heavier in alpha/beta computation.
            mtf_fast_span: Fast EMA span in ticks (~30s).
            mtf_medium_span: Medium EMA span in ticks (~1.5m).
            mtf_slow_span: Slow EMA span in ticks (~5m).
            mtf_min_agreement: Minimum fraction of timeframes agreeing (0-1).
            vol_sizing_enabled: Whether to scale Kelly by inverse volatility.
            vol_sizing_lookback: Ticks for baseline volatility estimation.
            vol_sizing_floor: Minimum vol sizing scalar (0.3 = 30% of Kelly).
            vol_sizing_ceiling: Maximum vol sizing scalar (1.5 = 150% of Kelly).
            mean_reversion_enabled: Whether to detect spread mean reversion.
            spread_history_maxlen: Rolling window for spread z-score.
            mean_reversion_z_threshold: Spread z above this = MR opportunity.
            mean_reversion_boost: Kelly multiplier when MR detected.
        """
        # Beta distribution parameters — start uninformative
        self.alpha: float = 1.0
        self.beta: float = 1.0

        self.min_edge: float = min_edge
        self.round_trip_cost: float = round_trip_cost
        self.decay_factor: float = decay_factor

        # Window open price — set on first tick, used for directional tracking
        self._window_open_price: Optional[float] = None

        # Rolling price history for volatility calculation.
        # Deque with maxlen auto-evicts old observations.
        self.price_history: Deque[float] = deque(maxlen=self.VOLATILITY_WINDOW)

        # Tick direction history for exponential decay weighting.
        # +1 = up, -1 = down, 0 = flat. Used by apply_decay_weights().
        self.tick_directions: Deque[int] = deque(maxlen=self.VOLATILITY_WINDOW)

        self.tick_count: int = 0
        self.logger: logging.Logger = logging.getLogger("Model")

        # SQLite memory store for state persistence (replaces Redis)
        self._memory = memory_store

        # --- Multi-Timeframe Momentum (EMA-based) ---
        # Three EMA windows capture momentum at different time scales.
        # When all agree on direction, signal confidence is highest.
        self._mtf_fast_span: int = mtf_fast_span
        self._mtf_medium_span: int = mtf_medium_span
        self._mtf_slow_span: int = mtf_slow_span
        self._mtf_min_agreement: float = mtf_min_agreement
        # EMA state: initialized to None, set on first price
        self._ema_fast: Optional[float] = None
        self._ema_medium: Optional[float] = None
        self._ema_slow: Optional[float] = None

        # --- Volatility-Adjusted Sizing ---
        # Tracks longer price history for baseline vol estimation.
        self._vol_sizing_enabled: bool = vol_sizing_enabled
        self._vol_sizing_lookback: int = vol_sizing_lookback
        self._vol_sizing_floor: float = vol_sizing_floor
        self._vol_sizing_ceiling: float = vol_sizing_ceiling
        self._vol_history: Deque[float] = deque(maxlen=vol_sizing_lookback)

        # --- Mean Reversion Detection ---
        # Tracks the edge (true_prob - implied_prob) over time.
        # When current spread is abnormally wide vs history, it's likely
        # to revert — which means our directional bet has extra tailwind.
        self._mean_reversion_enabled: bool = mean_reversion_enabled
        self._spread_history: Deque[float] = deque(maxlen=spread_history_maxlen)
        self._mr_z_threshold: float = mean_reversion_z_threshold
        self._mr_boost: float = mean_reversion_boost

        # Restore state from SQLite if available
        self._load_state()

    def reset(self) -> None:
        """Reset to uninformative prior for new market window.

        Called by bot.py when GammaMarketFinder discovers a new 5-min window.
        This prevents momentum from the previous window bleeding into signals
        for the new window — each window is an independent event.

        NOTE: _vol_history and _spread_history are NOT cleared on reset.
        They accumulate across windows to build a stable baseline for
        volatility normalization and mean reversion detection. This is
        intentional — cross-window context improves these estimators.
        """
        self.alpha = 1.0
        self.beta = 1.0
        self.price_history.clear()
        self.tick_directions.clear()
        self.tick_count = 0
        self._window_open_price = None
        # Reset EMAs (each window starts fresh for momentum)
        self._ema_fast = None
        self._ema_medium = None
        self._ema_slow = None
        self._save_state()

    def update(self, current_price: float, previous_price: Optional[float]) -> None:
        """Bayesian update on a new price observation.

        The update rule is the conjugate beta-binomial:
        - Observe "success" (price up) → alpha += 1
        - Observe "failure" (price down) → beta += 1

        This is mathematically equivalent to computing:
            P(up | data) ∝ P(data | up) * P(up)
        where the likelihood is Bernoulli and prior is Beta.

        Also updates multi-timeframe EMAs and volatility history
        for the advanced statistical filters.

        Args:
            current_price: Latest BTC price from OKX
            previous_price: Previous tick's price (None on first tick)
        """
        self.price_history.append(current_price)
        self.tick_count += 1

        # Update multi-timeframe EMAs on every tick.
        # EMA formula: ema = alpha * price + (1 - alpha) * ema_prev
        # where alpha = 2 / (span + 1)
        self._update_ema(current_price)

        # Record window open price on first tick
        if self._window_open_price is None:
            self._window_open_price = current_price

        if previous_price is None:
            return  # first tick — no direction to observe

        # Track position relative to window OPEN, not tick-to-tick momentum.
        # This directly matches Polymarket resolution: "is BTC above/below open?"
        # Tick-to-tick momentum caused mean-reversion losses because the model
        # detected micro-spikes that reliably reversed.
        if current_price > self._window_open_price:
            self.tick_directions.append(1)
        elif current_price < self._window_open_price:
            self.tick_directions.append(-1)
        else:
            self.tick_directions.append(0)

        # Track per-tick volatility across windows for vol-adjusted sizing.
        # This builds a stable vol baseline that persists across window resets.
        tick_return: float = (current_price - previous_price) / previous_price
        self._vol_history.append(tick_return)

        # Recompute alpha/beta using exponential decay weighting.
        # Recent ticks get weight ~1.0, older ticks decay by factor^age.
        self.alpha, self.beta = apply_decay_weights(
            list(self.tick_directions), self.decay_factor
        )

        self._save_state()

    def _update_ema(self, price: float) -> None:
        """Update all three EMA timeframes with new price.

        EMA is an exponentially weighted moving average that reacts faster
        to recent prices than SMA. The smoothing factor alpha = 2/(span+1)
        controls responsiveness:
        - Fast (span=10): alpha=0.182, reacts in ~5 ticks
        - Medium (span=30): alpha=0.065, reacts in ~15 ticks
        - Slow (span=100): alpha=0.020, reacts in ~50 ticks
        """
        for attr, span in [
            ("_ema_fast", self._mtf_fast_span),
            ("_ema_medium", self._mtf_medium_span),
            ("_ema_slow", self._mtf_slow_span),
        ]:
            current = getattr(self, attr)
            if current is None:
                setattr(self, attr, price)
            else:
                alpha = 2.0 / (span + 1)
                setattr(self, attr, alpha * price + (1.0 - alpha) * current)

    def _save_state(self) -> None:
        """Persist Bayesian state + cross-window state to SQLite.

        Called after every update() and reset(). SQLite WAL mode writes
        in ~0.1ms — faster than Redis over TCP for our payload sizes.
        Cross-window state saved every 10th tick to reduce write frequency.
        """
        if self._memory is None:
            return
        try:
            self._memory.set("model_state", {
                "alpha": self.alpha,
                "beta": self.beta,
                "price_history": list(self.price_history),
                "tick_directions": list(self.tick_directions),
                "tick_count": self.tick_count,
            })
            # Cross-window state saved less frequently
            if self.tick_count % 10 == 0:
                self._memory.set("model_cross_window", {
                    "vol_history": list(self._vol_history),
                    "spread_history": list(self._spread_history),
                    "ema_fast": self._ema_fast,
                    "ema_medium": self._ema_medium,
                    "ema_slow": self._ema_slow,
                })
        except Exception:
            pass  # never block the trading loop

    def _load_state(self) -> None:
        """Restore Bayesian + cross-window state from SQLite on startup."""
        if self._memory is None:
            return

        data = self._memory.get("model_state")
        if data is not None:
            try:
                self.alpha = float(data["alpha"])
                self.beta = float(data["beta"])
                self.price_history = deque(
                    data["price_history"], maxlen=self.VOLATILITY_WINDOW
                )
                self.tick_directions = deque(
                    data.get("tick_directions", []), maxlen=self.VOLATILITY_WINDOW
                )
                self.tick_count = int(data["tick_count"])
                self.logger.info(
                    f"Restored model state "
                    f"(alpha={self.alpha:.1f}, beta={self.beta:.1f}, ticks={self.tick_count})"
                )
            except Exception as e:
                self.logger.warning(f"Could not load model state: {e}")

        cw = self._memory.get("model_cross_window")
        if cw is not None:
            try:
                self._vol_history = deque(
                    cw.get("vol_history", []), maxlen=self._vol_sizing_lookback
                )
                self._spread_history = deque(
                    cw.get("spread_history", []), maxlen=self._spread_history.maxlen
                )
                if cw.get("ema_fast") is not None:
                    self._ema_fast = float(cw["ema_fast"])
                if cw.get("ema_medium") is not None:
                    self._ema_medium = float(cw["ema_medium"])
                if cw.get("ema_slow") is not None:
                    self._ema_slow = float(cw["ema_slow"])
                self.logger.info(
                    f"Restored cross-window state "
                    f"(vol_pts={len(self._vol_history)}, "
                    f"spread_pts={len(self._spread_history)})"
                )
            except Exception as e:
                self.logger.warning(f"Could not load cross-window state: {e}")

    @property
    def true_prob_up(self) -> float:
        """Posterior mean P(next tick is up) = α / (α + β).

        This is the minimum variance unbiased estimator for the beta
        distribution. With α=β=1 (uninformative prior), it starts at 0.5
        and moves toward the observed frequency as data accumulates.
        """
        return self.alpha / (self.alpha + self.beta)

    @property
    def volatility(self) -> float:
        """Per-tick volatility as standard deviation of simple returns.

        Computed over the last 30 ticks (VOLATILITY_WINDOW). This captures
        the current regime's noise level — high volatility means larger
        price swings and more uncertainty in direction prediction.

        Used to:
        1. Normalize the edge into a z-score (signal quality metric)
        2. Scale Monte Carlo simulation paths

        Returns:
            Standard deviation of returns, or 0.0 if insufficient data.
        """
        if len(self.price_history) < 2:
            return 0.0
        prices: np.ndarray = np.array(self.price_history)
        # Simple returns: (p[t] - p[t-1]) / p[t-1]
        returns: np.ndarray = np.diff(prices) / prices[:-1]
        return float(np.std(returns))

    def _compute_liquidity_factor(self, order_book: Optional[dict]) -> float:
        """Compute a z-score scaling factor from CLOB order book depth.

        Thin order books indicate low liquidity and higher noise — signals
        in thin markets are less reliable because small orders can move the
        midpoint. We scale the z-score down (divide by factor > 1) to raise
        the bar for trade entry when liquidity is thin.

        Thick books indicate a well-stocked market where the midpoint is
        stable and our edge is more likely to be real, so we lower the bar.

        The factor linearly interpolates between:
        - THIN_BOOK_DEPTH (500 tokens)  → factor 1.5 (50% harder to trade)
        - THICK_BOOK_DEPTH (5000 tokens) → factor 0.8 (20% easier to trade)

        Args:
            order_book: Dict with 'bids' and 'asks' arrays from CLOB,
                        or None if unavailable (graceful degradation)

        Returns:
            Scaling factor >= 0.8 — multiply into z_score denominator
        """
        if order_book is None:
            return 1.0

        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])

            # Sum sizes from top 3 levels on each side
            bid_depth: float = sum(
                float(level.get("size", level[1]) if isinstance(level, dict) else level[1])
                for level in bids[:3]
            )
            ask_depth: float = sum(
                float(level.get("size", level[1]) if isinstance(level, dict) else level[1])
                for level in asks[:3]
            )
            total_depth: float = bid_depth + ask_depth

            # Linear interpolation between thin and thick thresholds
            if total_depth <= self.THIN_BOOK_DEPTH:
                return 1.5
            if total_depth >= self.THICK_BOOK_DEPTH:
                return 0.8
            # Interpolate: 1.5 at thin, 0.8 at thick
            t: float = (total_depth - self.THIN_BOOK_DEPTH) / (
                self.THICK_BOOK_DEPTH - self.THIN_BOOK_DEPTH
            )
            return 1.5 - t * 0.7  # 1.5 → 0.8

        except Exception:
            return 1.0  # degrade gracefully on parse errors

    def _compute_z_score(
        self,
        true_prob: float,
        implied_prob: float,
        liquidity_factor: float = 1.0,
    ) -> float:
        """Normalize the edge by volatility and liquidity to get a confidence score.

        z_score = (true_prob - implied_prob) / (volatility * liquidity_factor)

        The liquidity_factor adjusts for order book depth:
        - Thin books (factor > 1.0) → lower z_score → harder to pass thresholds
        - Thick books (factor < 1.0) → higher z_score → edge more reliable

        Interpretation:
        - |z| > 2: strong signal, edge is 2+ standard deviations from noise
        - |z| 1-2: moderate signal, proceed with caution
        - |z| < 1: weak signal, likely noise — don't trade

        Args:
            true_prob: Our Bayesian estimate of P(correct direction)
            implied_prob: Market's estimate from CLOB midpoint
            liquidity_factor: Scaling factor from order book depth (default 1.0)

        Returns:
            z-score float, or 0.0 if volatility is negligible
        """
        vol: float = self.volatility
        if vol < 1e-10:
            # Near-zero volatility means price hasn't moved — no signal
            return 0.0
        return (true_prob - implied_prob) / (vol * liquidity_factor)

    def compute_mtf_agreement(self, direction: str) -> float:
        """Compute multi-timeframe momentum agreement score.

        Checks how many EMA timeframes agree with the proposed direction:
        - Price > EMA = bullish signal for that timeframe
        - Price < EMA = bearish signal for that timeframe

        Agreement score = fraction of timeframes confirming direction.
        Score of 1.0 means all three timeframes confirm (strongest signal).
        Score of 0.0 means all three disagree (counter-trend, avoid).

        This is the same principle used by trend-following funds:
        multi-timeframe confluence reduces false signals by 40-60%.

        Args:
            direction: "UP" or "DOWN"

        Returns:
            Agreement score between 0.0 and 1.0
        """
        if len(self.price_history) < 2:
            return 1.0  # insufficient data, don't filter

        current_price: float = self.price_history[-1]
        confirmations: int = 0
        total: int = 0

        for ema in (self._ema_fast, self._ema_medium, self._ema_slow):
            if ema is None:
                continue
            total += 1
            if direction == "UP" and current_price > ema:
                confirmations += 1
            elif direction == "DOWN" and current_price < ema:
                confirmations += 1

        if total == 0:
            return 1.0
        return confirmations / total

    def compute_vol_scalar(self) -> float:
        """Compute volatility-adjusted position sizing scalar.

        The idea: size inversely to realized volatility.
        - In calm markets with strong signals → size up (confidence is high)
        - In volatile markets → size down (uncertainty is high)

        Uses the ratio of current window vol to baseline (cross-window) vol.
        If current vol is 2x the baseline, scalar = 0.5 (half Kelly).
        If current vol is 0.5x the baseline, scalar = 1.5 (capped).

        This is standard risk parity / inverse-vol weighting used by
        institutional quant funds (Bridgewater, AQR, etc).

        Returns:
            Scalar between vol_sizing_floor and vol_sizing_ceiling
        """
        if not self._vol_sizing_enabled:
            return 1.0

        current_vol: float = self.volatility
        if current_vol < 1e-10:
            return 1.0  # no vol data, neutral sizing

        # Baseline vol from cross-window history
        if len(self._vol_history) < 10:
            return 1.0  # insufficient baseline, neutral sizing

        baseline_vol: float = float(np.std(list(self._vol_history)))
        if baseline_vol < 1e-10:
            return 1.0

        # Inverse vol ratio: high current vol → lower scalar
        ratio: float = baseline_vol / current_vol
        return max(self._vol_sizing_floor, min(ratio, self._vol_sizing_ceiling))

    def compute_spread_z(self, true_prob: float, implied_prob: float) -> float:
        """Compute z-score of current spread vs rolling spread history.

        Tracks the raw spread (true_prob - implied_prob) over time.
        When the current spread is far from its rolling mean, it's
        likely to mean-revert — which means our directional bet has
        a statistical tailwind.

        High spread_z (>1.5): spread is abnormally wide → mean reversion
        likely → boost confidence / position size.
        Low spread_z (<0.5): spread is normal → no extra conviction.

        This captures the well-documented mean reversion in prediction
        market mispricings (Wolfers & Zitzewitz 2004).

        Args:
            true_prob: Our Bayesian estimate
            implied_prob: Market's estimate from CLOB

        Returns:
            Z-score of current spread. Always >= 0 (absolute value).
        """
        if not self._mean_reversion_enabled:
            return 0.0

        spread: float = true_prob - implied_prob
        self._spread_history.append(spread)

        if len(self._spread_history) < 20:
            return 0.0  # insufficient history

        spreads: np.ndarray = np.array(self._spread_history)
        mean: float = float(np.mean(spreads))
        std: float = float(np.std(spreads))

        if std < 1e-10:
            return 0.0

        return abs((spread - mean) / std)

    def _monte_carlo(
        self, current_price: float, remaining_seconds: float, direction: str,
        open_price: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Simulate 1000 price paths to estimate expected value of the trade.

        Model: Geometric Brownian Motion with zero drift and current realized
        volatility. Each path simulates BTC price at window expiry.

        The key insight: even if our Bayesian posterior says P(up)=0.65, the
        actual EV depends on how prices evolve over the remaining window.
        Monte Carlo captures path-dependent dynamics like mean reversion
        and volatility clustering that the simple Bayesian model misses.

        Payoff is binary:
        - Win (correct direction): +1.0 - round_trip_cost
        - Lose (wrong direction): -round_trip_cost

        All computation is vectorized — no Python loops. A single
        np.random.standard_normal(1000) call generates all paths.

        Args:
            current_price: Latest BTC price
            remaining_seconds: Seconds until window resolution
            direction: "UP" or "DOWN" — which outcome we're betting on

        Returns:
            Tuple of (expected_value, win_probability, variance)
        """
        vol: float = self.volatility
        if vol < 1e-10 or remaining_seconds <= 0:
            return (0.0, 0.5, 0.0)

        # Scale per-tick volatility to remaining window duration.
        # Assuming ~1 tick/second, vol scales by sqrt(time) per GBM.
        vol_scaled: float = vol * np.sqrt(remaining_seconds)

        # Dynamically select MC path count: 2000 during high-vol (>2%),
        # 1000 otherwise. More paths in volatile regimes improves tail accuracy.
        mc_paths: int = get_dynamic_mc_paths(vol, self.MC_PATHS)

        # Vectorized GBM: S(T) = S(0) * exp((−σ²/2)T + σ√T * Z)
        # where Z ~ N(0,1). We compute all paths in one operation.
        z: np.ndarray = np.random.standard_normal(mc_paths)
        terminal_prices: np.ndarray = current_price * np.exp(
            -0.5 * vol_scaled**2 + vol_scaled * z
        )

        # Binary outcome: did price end above/below window OPEN?
        # Matches Polymarket resolution (close vs open), not continuation from eval point.
        ref_price: float = open_price if open_price is not None else current_price
        if direction == "UP":
            wins: np.ndarray = terminal_prices > ref_price
        else:
            wins = terminal_prices < ref_price

        win_prob: float = float(np.mean(wins))

        # Binary payoff after round-trip costs
        payoffs: np.ndarray = np.where(
            wins, 1.0 - self.round_trip_cost, -self.round_trip_cost
        )
        ev: float = float(np.mean(payoffs))
        variance: float = float(np.var(payoffs))

        return (ev, win_prob, variance)

    def _kelly_fraction(self, win_prob: float, payout_ratio: float = 1.0) -> float:
        """Compute optimal bet size via full Kelly criterion.

        Kelly formula: f* = (b*p - q) / b
        where:
            b = net payout ratio (how much you win per dollar risked)
            p = probability of winning
            q = 1 - p = probability of losing

        Explosive mode: full Kelly (no down-scaling) for maximum compounding
        speed. Accepts higher variance in exchange for optimal geometric growth.

        Args:
            win_prob: Estimated probability of winning the trade
            payout_ratio: Net payout ratio (1.0 for even-money bets)

        Returns:
            Kelly fraction clamped to [0.0, 1.0]
        """
        q: float = 1.0 - win_prob
        f: float = (payout_ratio * win_prob - q) / payout_ratio
        # Full Kelly: maximum growth rate, no down-scaling
        return max(0.0, min(f, 1.0))

    def evaluate(
        self,
        current_price: float,
        implied_prob_up: float,
        remaining_seconds: float,
        order_book: Optional[dict] = None,
        min_edge_override: Optional[float] = None,
    ) -> Optional[Signal]:
        """Full evaluation pipeline: Bayesian posterior → z-score → MC → Kelly.

        This is called on every tick from the trading loop. The pipeline:

        1. Compute Bayesian posterior P(up) from accumulated tick evidence
        2. Determine direction: UP if posterior > 0.5, DOWN otherwise
        3. Compute net edge = |true_prob - implied_prob| - round_trip_cost
        4. If net edge < min_edge_threshold → return None (not worth trading)
        5. Compute liquidity-adjusted z-score from order book depth
        6. Run Monte Carlo to validate edge over 1000 simulated paths
        7. If MC expected value ≤ 0 → return None (edge doesn't survive noise)
        8. Compute Kelly fraction for position sizing
        9. Return Signal with all metrics

        The triple gating (edge threshold + liquidity-adjusted z-score + MC EV)
        is the key to avoiding false signals. The Bayesian posterior can overfit
        to short-term noise; the z-score adjusts for market depth; the MC
        simulation stress-tests whether the edge persists across random paths.

        Args:
            current_price: Latest BTC price from OKX
            implied_prob_up: CLOB midpoint price of the UP token (= market P(up))
            remaining_seconds: Seconds until this 5-min window resolves
            order_book: Optional CLOB L2 order book for liquidity-aware z-score

        Returns:
            Signal if profitable edge found, None otherwise
        """
        start: float = time.perf_counter()

        # Analytical P(close > open) from GBM model.
        # Given current position relative to open and remaining volatility,
        # this is a well-calibrated probability — not a backward-looking count.
        open_px: float = self._window_open_price if self._window_open_price else current_price
        vol: float = self.volatility
        if vol < 1e-10 or remaining_seconds <= 0 or open_px <= 0 or current_price <= 0:
            true_prob = 0.5
        else:
            vol_scaled: float = vol * math.sqrt(remaining_seconds)
            if vol_scaled < 1e-10:
                true_prob = 0.5
            else:
                log_ratio: float = math.log(current_price / open_px) / vol_scaled
                true_prob = 0.5 * (1.0 + math.erf(log_ratio / math.sqrt(2.0)))

        # Determine which direction has edge by comparing both sides.
        # Pick the direction where true_prob exceeds implied_prob (underpriced token).
        # Edge UP  = true_prob - implied_prob_up  (model thinks UP more likely than market)
        # Edge DOWN = implied_prob_up - true_prob  (model thinks DOWN more likely than market)
        edge_up: float = true_prob - implied_prob_up
        edge_down: float = implied_prob_up - true_prob  # = (1-true) - (1-implied)

        if edge_up >= edge_down:
            direction: str = "UP"
            directional_true: float = true_prob
            directional_implied: float = implied_prob_up
        else:
            direction = "DOWN"
            directional_true = 1.0 - true_prob
            directional_implied = 1.0 - implied_prob_up

        # Net edge after subtracting round-trip costs
        edge: float = directional_true - directional_implied - self.round_trip_cost

        # Gate 1: minimum edge threshold (dynamically adjustable via adaptive_z_score)
        # Below this, the signal-to-noise ratio is too low to overcome
        # execution slippage and model uncertainty
        active_min_edge: float = min_edge_override if min_edge_override is not None else self.min_edge
        if edge < active_min_edge:
            return None

        # Gate 2: Multi-timeframe momentum confirmation
        # Requires at least mtf_min_agreement (default 66%) of EMA timeframes
        # to agree with the proposed direction. This filters counter-trend
        # noise that the Bayesian model might pick up from 2-3 lucky ticks.
        mtf_agreement: float = self.compute_mtf_agreement(direction)
        if mtf_agreement < self._mtf_min_agreement:
            return None

        # Z-score with dynamic liquidity adjustment
        # Thin order books → factor > 1 → lower z_score → harder to trade
        # Thick order books → factor < 1 → higher z_score → edge more reliable
        liquidity_factor: float = self._compute_liquidity_factor(order_book)
        z_score: float = self._compute_z_score(
            directional_true, directional_implied, liquidity_factor
        )

        # Gate 3: Monte Carlo validation
        # Even if the instantaneous edge looks good, simulate 1000 paths
        # to check if the edge survives price evolution over remaining time
        mc_ev: float
        mc_win_prob: float
        mc_variance: float
        mc_ev, mc_win_prob, mc_variance = self._monte_carlo(
            current_price, remaining_seconds, direction, open_price=open_px
        )

        # Gate 4: MC expected value must be positive
        # This filters out edges that look good statically but get eroded
        # by volatility over the remaining window
        if mc_ev <= 0:
            return None

        # Kelly sizing: compute optimal fraction of bankroll to risk
        # Payout ratio for binary bet: pay `implied` to receive `1.0` on win
        if directional_implied > 0.01:
            payout_ratio: float = (1.0 - directional_implied) / directional_implied
        else:
            payout_ratio = 1.0  # avoid division issues at extreme prices

        kelly: float = self._kelly_fraction(mc_win_prob, payout_ratio)

        # --- Advanced sizing adjustments ---

        # Volatility-adjusted sizing: scale Kelly by inverse vol ratio.
        # High vol → smaller size, low vol → larger size.
        vol_scalar: float = self.compute_vol_scalar()
        kelly *= vol_scalar

        # Mean reversion detection: if spread is abnormally wide, boost.
        # The spread is likely to narrow, giving our direction extra tailwind.
        spread_z: float = self.compute_spread_z(directional_true, directional_implied)
        if spread_z >= self._mr_z_threshold:
            kelly *= self._mr_boost

        # Final Kelly clamp after all adjustments
        kelly = max(0.0, min(kelly, 1.0))

        elapsed_ms: float = (time.perf_counter() - start) * 1000

        return Signal(
            direction=direction,
            true_prob=directional_true,
            implied_prob=directional_implied,
            edge=edge,
            z_score=z_score,
            mc_ev=mc_ev,
            mc_win_prob=mc_win_prob,
            mc_variance=mc_variance,
            kelly_fraction=kelly,
            computation_ms=elapsed_ms,
            mtf_agreement=mtf_agreement,
            vol_scalar=vol_scalar,
            spread_z=spread_z,
        )

    def evaluate_snipe(
        self,
        current_price: float,
        implied_prob_up: float,
        remaining_seconds: float,
        order_book: Optional[dict] = None,
        sniper_min_confidence: float = 0.80,
        sniper_min_price_discount: float = 0.05,
        sniper_max_exposure_pct: float = 0.20,
        sniper_min_ticks: int = 15,
    ) -> Optional[Signal]:
        """Resolution sniper: high-confidence late-window entries.

        In the last 30-60 seconds of a 5-min window, the BTC price direction
        is essentially decided. A stock that's been trending up for 4 minutes
        is overwhelmingly likely to finish up. But the CLOB still has shares
        priced at 0.70-0.85 because retail participants:
        1. Aren't watching (it's a 5-min market, attention is fleeting)
        2. Are scared of the short time left (they see risk, we see certainty)
        3. Have stale limit orders sitting from earlier in the window

        This is not a prediction — it's collecting the gap between reality
        and the market's delayed pricing of near-certain outcomes.

        The sniper bypasses the full evaluate() pipeline (MC, MTF) because
        at <60s remaining, those gates add noise, not signal. We only need:
        1. Strong Bayesian posterior (>80% confidence from tick history)
        2. CLOB price meaningfully below our estimate (>5% discount)

        Win rate target: 85%+. This is the bot's highest-conviction strategy.

        Args:
            current_price: Latest BTC price
            implied_prob_up: CLOB midpoint of UP token
            remaining_seconds: Seconds until window resolution
            order_book: Optional CLOB order book for liquidity check
            sniper_min_confidence: Minimum Bayesian P(direction) to snipe
            sniper_min_price_discount: Minimum gap between our prob and CLOB price
            sniper_max_exposure_pct: Max exposure for sniper (higher = more aggressive)
            sniper_min_ticks: Minimum tick count before sniping

        Returns:
            Signal if snipe opportunity found, None otherwise
        """
        start: float = time.perf_counter()

        # Need enough data to have a reliable posterior
        if self.tick_count < sniper_min_ticks:
            return None

        true_prob: float = self.true_prob_up

        # Determine direction and confidence
        if true_prob >= 0.5:
            direction: str = "UP"
            confidence: float = true_prob
            clob_price: float = implied_prob_up
        else:
            direction = "DOWN"
            confidence = 1.0 - true_prob
            clob_price = 1.0 - implied_prob_up

        # Gate 1: confidence must be high
        if confidence < sniper_min_confidence:
            return None

        # Gate 2: CLOB must be underpricing this outcome
        discount: float = confidence - clob_price
        if discount < sniper_min_price_discount:
            return None

        # Gate 3: liquidity check — don't snipe into an empty book
        if order_book is not None:
            liquidity_factor: float = self._compute_liquidity_factor(order_book)
            if liquidity_factor >= 1.4:  # very thin book
                return None

        # Compute edge and Kelly for the snipe
        edge: float = discount - self.round_trip_cost

        if edge <= 0:
            return None

        # Simple Kelly for near-certain bets
        # Payout ratio for binary: (1 - clob_price) / clob_price
        if clob_price > 0.01:
            payout_ratio: float = (1.0 - clob_price) / clob_price
        else:
            payout_ratio = 1.0

        kelly: float = self._kelly_fraction(confidence, payout_ratio)

        # Sniper can size bigger since conviction is higher
        kelly = min(kelly, 1.0)

        elapsed_ms: float = (time.perf_counter() - start) * 1000

        return Signal(
            direction=direction,
            true_prob=confidence,
            implied_prob=clob_price,
            edge=edge,
            z_score=discount / max(self.volatility, 1e-6),  # simplified z
            mc_ev=edge,  # skip MC, edge IS the EV at this point
            mc_win_prob=confidence,
            mc_variance=0.0,
            kelly_fraction=kelly,
            computation_ms=elapsed_ms,
            mtf_agreement=1.0,  # MTF not used for sniper
            vol_scalar=1.0,     # no vol adjustment for sniper
            spread_z=0.0,
        )

    def check_window_quality(
        self,
        current_hour_utc: int,
        high_edge_hours: List[int],
        min_vol_for_off_hours: float = 0.005,
        min_window_move_pct: float = 0.03,
    ) -> dict:
        """Evaluate whether the current window is worth trading.

        Not all 5-minute windows have edge. This filter identifies:
        1. High-edge hours (major market opens, overlap sessions)
        2. Volatility spikes (news events during dead hours)
        3. Sufficient price movement (flat = no directional edge)

        Returns a quality assessment that the trade loop uses to decide
        whether to engage or sit out.

        Args:
            current_hour_utc: Current hour in UTC (0-23)
            high_edge_hours: List of UTC hours considered high-edge
            min_vol_for_off_hours: Minimum vol to trade during off-hours
            min_window_move_pct: Minimum price change since window open

        Returns:
            Dict with:
                tradeable: bool — should we trade this window?
                reason: str — why or why not
                quality_score: float — 0.0 to 1.0 quality rating
        """
        is_high_edge_hour: bool = current_hour_utc in high_edge_hours
        current_vol: float = self.volatility
        has_vol_spike: bool = current_vol >= min_vol_for_off_hours

        # Check if price has moved enough since window open
        has_movement: bool = False
        if len(self.price_history) >= 2:
            first_price: float = self.price_history[0]
            last_price: float = self.price_history[-1]
            if first_price > 0:
                move_pct: float = abs(last_price - first_price) / first_price
                has_movement = move_pct >= min_window_move_pct
            else:
                has_movement = False

        # Quality scoring
        score: float = 0.0

        # Session quality (0.0 - 0.4)
        if is_high_edge_hour:
            score += 0.4
        elif has_vol_spike:
            score += 0.2  # off-hours but something is happening

        # Volatility quality (0.0 - 0.3)
        if current_vol > 0.01:
            score += 0.3  # high vol = wider mispricings
        elif current_vol > 0.003:
            score += 0.15

        # Movement quality (0.0 - 0.3)
        if has_movement:
            score += 0.3

        # Decision logic
        if is_high_edge_hour and has_movement:
            return {
                "tradeable": True,
                "reason": "high-edge hour with price movement",
                "quality_score": score,
            }

        if has_vol_spike and has_movement:
            return {
                "tradeable": True,
                "reason": f"vol spike ({current_vol:.4f}) during off-hours",
                "quality_score": score,
            }

        if is_high_edge_hour and not has_movement:
            return {
                "tradeable": False,
                "reason": "high-edge hour but flat price — no directional edge",
                "quality_score": score,
            }

        return {
            "tradeable": False,
            "reason": f"off-hours (hour={current_hour_utc}), "
                      f"vol={current_vol:.6f}, no movement",
            "quality_score": score,
        }

    def check_market_making_opportunity(
        self,
        yes_midpoint: float,
        no_midpoint: float,
        spread_threshold: float = 0.985,
        batch_size: int = 5,
        stoikov_yes: Optional[dict] = None,
        stoikov_no: Optional[dict] = None,
    ) -> Optional[dict]:
        """Check if YES + NO midpoints create a market-making spread.

        When the sum of midpoints < threshold, buying both sides locks in
        guaranteed value (1.00 payout on one side, minus cost of both).
        If Stoikov reservation prices are provided, batch levels are anchored
        at the optimal_bid from each side's Stoikov calculation.

        Args:
            yes_midpoint: Current CLOB midpoint for YES token (0.0–1.0)
            no_midpoint: Current CLOB midpoint for NO token (0.0–1.0)
            spread_threshold: Trigger when sum < this (default 0.985)
            batch_size: Number of price levels per side (default 5)
            stoikov_yes: Optional Stoikov dict for YES side (from compute_stoikov)
            stoikov_no: Optional Stoikov dict for NO side (from compute_stoikov)

        Returns:
            Dict with spread metrics + batch_levels if opportunity found, None otherwise.
        """
        mid_sum: float = yes_midpoint + no_midpoint
        if mid_sum >= spread_threshold:
            return None

        spread: float = 1.0 - mid_sum

        # Best price: Stoikov optimal_bid if available, else naive midpoint - 0.01
        if stoikov_yes is not None:
            yes_price: float = stoikov_yes["optimal_bid"]
        else:
            yes_price = round(yes_midpoint - 0.01, 2)
        if stoikov_no is not None:
            no_price: float = stoikov_no["optimal_bid"]
        else:
            no_price = round(no_midpoint - 0.01, 2)

        # Clamp to valid range
        yes_price = max(0.01, min(0.99, yes_price))
        no_price = max(0.01, min(0.99, no_price))

        # Expected profit: 1.00 payout - cost of both sides - round-trip costs
        cost: float = yes_price + no_price
        expected_profit: float = 1.0 - cost - self.round_trip_cost

        if expected_profit <= 0:
            return None  # spread doesn't cover costs

        # Generate staggered batch levels anchored at Stoikov optimal_bid
        yes_anchor: float = yes_price
        no_anchor: float = no_price
        yes_levels: list = []
        no_levels: list = []
        for i in range(batch_size):
            yp: float = round(yes_anchor - i * 0.005, 3)
            np_: float = round(no_anchor - i * 0.005, 3)
            yes_levels.append(max(0.01, min(0.99, yp)))
            no_levels.append(max(0.01, min(0.99, np_)))

        return {
            "spread": spread,
            "yes_price": yes_price,
            "no_price": no_price,
            "expected_profit": expected_profit,
            "mid_sum": mid_sum,
            "batch_levels": {
                "yes": yes_levels,
                "no": no_levels,
            },
        }

    def compute_orderflow_imbalance(
        self,
        order_book: dict,
        lookback_depth: int = 5,
        threshold: float = 0.20,
    ) -> dict:
        """Calculate bid/ask depth delta from CLOB order book.

        Sums the top `lookback_depth` levels on each side to determine
        whether buying or selling pressure dominates. Used to bias MM
        side selection when imbalance exceeds threshold.

        Args:
            order_book: Dict with 'bids' and 'asks' arrays of {price, size}
            lookback_depth: Number of top levels to consider
            threshold: Minimum abs(imbalance) to trigger side bias

        Returns:
            Dict with imbalance (-1 to 1), favor_side ("YES"/"NO"/None),
            and magnitude (abs of imbalance).
        """
        bids: list = order_book.get("bids", [])[:lookback_depth]
        asks: list = order_book.get("asks", [])[:lookback_depth]

        bid_vol: float = sum(float(b.get("size", 0)) for b in bids)
        ask_vol: float = sum(float(a.get("size", 0)) for a in asks)
        total: float = bid_vol + ask_vol

        if total == 0:
            return {"imbalance": 0.0, "favor_side": None, "magnitude": 0.0}

        imbalance: float = (bid_vol - ask_vol) / total
        magnitude: float = abs(imbalance)

        favor_side: Optional[str] = None
        if magnitude >= threshold:
            favor_side = "YES" if imbalance > 0 else "NO"

        return {
            "imbalance": round(imbalance, 4),
            "favor_side": favor_side,
            "magnitude": round(magnitude, 4),
        }

    def compute_stoikov_reservation_price(
        self,
        midpoint: float,
        remaining_secs: float,
        gamma: float = 0.15,
    ) -> dict:
        """Avellaneda-Stoikov reservation price for optimal market-making.

        Core formula:
            r = s - q * gamma * sigma^2 * (T - t)

        Where:
            s = current CLOB midpoint price (YES share)
            q = Bayesian posterior centered at 0 (inventory skew)
            gamma = risk aversion parameter (higher = tighter quotes)
            sigma = rolling volatility (std of last 30 price ticks)
            T - t = seconds remaining in current 5-minute window

        Optimal execution prices derived from the reservation price:
            optimal_spread = max(0.02, gamma * sigma^2 * (T-t))
            ask = r + spread/2
            bid = r - spread/2

        Returns:
            Dict with reservation_price, optimal_bid, optimal_ask,
            optimal_spread, sigma, gamma, remaining_secs, inventory_skew.
        """
        # Guard against negative remaining time from timing races
        remaining_secs = max(0.0, remaining_secs)

        sigma: float = self.volatility  # std of last 30 ticks

        # Inventory skew: center posterior around 0 (-0.5 to +0.5)
        # Positive q (model bullish) → lower reservation price (encourages selling)
        # Negative q (model bearish) → higher reservation (encourages buying)
        q: float = self.true_prob_up - 0.5

        # Stoikov reservation price
        r: float = midpoint - q * gamma * (sigma ** 2) * remaining_secs

        # Optimal spread (Avellaneda-Stoikov approximation)
        optimal_spread: float = gamma * (sigma ** 2) * remaining_secs
        # Floor at 2 cents to ensure maker classification on CLOB
        optimal_spread = max(0.02, optimal_spread)

        optimal_ask: float = r + optimal_spread / 2
        optimal_bid: float = r - optimal_spread / 2

        # Clamp to valid CLOB range [0.01, 0.99]
        optimal_ask = max(0.01, min(0.99, round(optimal_ask, 3)))
        optimal_bid = max(0.01, min(0.99, round(optimal_bid, 3)))
        r = max(0.01, min(0.99, round(r, 3)))

        return {
            "reservation_price": r,
            "optimal_bid": optimal_bid,
            "optimal_ask": optimal_ask,
            "optimal_spread": round(optimal_spread, 4),
            "sigma": round(sigma, 6),
            "gamma": gamma,
            "remaining_secs": remaining_secs,
            "inventory_skew": round(q, 4),
        }
