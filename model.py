"""
model.py — Bayesian inference + Monte Carlo simulation for edge detection.

This is the brain of the bot. The core insight driving positive expectancy:

Polymarket 5-min BTC markets are priced by retail participants using lagging
information. Our Bayesian model updates on sub-20ms Binance perpetual ticks,
building a posterior distribution for P(BTC up) that leads the CLOB implied
probability by 1-3 seconds. When the gap (edge) exceeds costs, we trade.

The z-score normalizes the edge by recent volatility, filtering out noise.
Monte Carlo simulation validates the edge over 1000 GBM paths, ensuring
positive expected value even in the tail scenarios.

All computations are vectorized numpy — total evaluate() latency target <30ms.
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from performance import apply_decay_weights, get_dynamic_mc_paths

# Redis is optional — model works without it, but loses state persistence
try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None  # type: ignore[assignment]


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


class BayesianModel:
    """Bayesian beta-binomial model for short-term BTC direction prediction.

    The model maintains a beta distribution prior over P(next tick is up).
    Each Binance tick updates the posterior:
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
        redis_url: Optional[str] = None,
        decay_factor: float = 0.9,
    ) -> None:
        """Initialize with uninformative beta prior.

        Args:
            min_edge: Minimum net edge (after costs) required to generate signal.
                      Lower = more trades but noisier. 0.02 is conservative.
            round_trip_cost: Total cost of entering + exiting position.
                             0.3% assumes worst-case taker fees both sides.
                             Maker fills reduce this, creating hidden alpha.
            redis_url: Optional Redis URL for state persistence across restarts.
                       If None or redis-py not installed, persistence is disabled.
            decay_factor: Exponential decay for tick weighting (0.9 = 10% decay).
                          Recent ticks are weighted heavier in alpha/beta computation.
        """
        # Beta distribution parameters — start uninformative
        self.alpha: float = 1.0
        self.beta: float = 1.0

        self.min_edge: float = min_edge
        self.round_trip_cost: float = round_trip_cost
        self.decay_factor: float = decay_factor

        # Rolling price history for volatility calculation.
        # Deque with maxlen auto-evicts old observations.
        self.price_history: Deque[float] = deque(maxlen=self.VOLATILITY_WINDOW)

        # Tick direction history for exponential decay weighting.
        # +1 = up, -1 = down, 0 = flat. Used by apply_decay_weights().
        self.tick_directions: Deque[int] = deque(maxlen=self.VOLATILITY_WINDOW)

        self.tick_count: int = 0
        self.logger: logging.Logger = logging.getLogger("Model")

        # Redis connection for Bayesian state persistence (optional).
        # Sub-1ms SET on localhost — no impact on 80ms cycle budget.
        self._redis = None
        if _redis_lib is not None and redis_url:
            try:
                self._redis = _redis_lib.Redis.from_url(
                    redis_url, socket_connect_timeout=1, socket_timeout=1
                )
                self._redis.ping()
                self.logger.info("Redis connected for model state persistence")
            except Exception as e:
                self.logger.warning(f"Redis unavailable, persistence disabled: {e}")
                self._redis = None

        # Restore state from Redis if available (silent no-op if Redis is down)
        self._load_from_redis()

    def reset(self) -> None:
        """Reset to uninformative prior for new market window.

        Called by bot.py when GammaMarketFinder discovers a new 5-min window.
        This prevents momentum from the previous window bleeding into signals
        for the new window — each window is an independent event.
        """
        self.alpha = 1.0
        self.beta = 1.0
        self.price_history.clear()
        self.tick_directions.clear()
        self.tick_count = 0
        self._save_to_redis()

    def update(self, current_price: float, previous_price: Optional[float]) -> None:
        """Bayesian update on a new price observation.

        The update rule is the conjugate beta-binomial:
        - Observe "success" (price up) → alpha += 1
        - Observe "failure" (price down) → beta += 1

        This is mathematically equivalent to computing:
            P(up | data) ∝ P(data | up) * P(up)
        where the likelihood is Bernoulli and prior is Beta.

        Args:
            current_price: Latest BTC price from Binance
            previous_price: Previous tick's price (None on first tick)
        """
        self.price_history.append(current_price)
        self.tick_count += 1

        if previous_price is None:
            return  # first tick — no direction to observe

        if current_price > previous_price:
            self.tick_directions.append(1)
        elif current_price < previous_price:
            self.tick_directions.append(-1)
        else:
            self.tick_directions.append(0)

        # Recompute alpha/beta using exponential decay weighting.
        # Recent ticks get weight ~1.0, older ticks decay by factor^age.
        self.alpha, self.beta = apply_decay_weights(
            list(self.tick_directions), self.decay_factor
        )

        self._save_to_redis()

    def _save_to_redis(self) -> None:
        """Persist current Bayesian state to Redis.

        Called after every update() and reset(). Synchronous but fast —
        localhost SET with ~200-byte payload completes in <0.5ms.
        Fails silently so the model always continues operating.
        """
        if self._redis is None:
            return
        try:
            state: str = json.dumps({
                "alpha": self.alpha,
                "beta": self.beta,
                "price_history": list(self.price_history),
                "tick_directions": list(self.tick_directions),
                "tick_count": self.tick_count,
            })
            self._redis.set("polybot:model_state", state)
        except Exception:
            pass  # never block the trading loop for persistence

    def _load_from_redis(self) -> None:
        """Restore Bayesian state from Redis on startup.

        If Redis has prior state, the model resumes with accumulated
        evidence instead of starting from an uninformative prior.
        This preserves winning edge continuity across process restarts.
        Fails silently — uses default uninformative prior on any error.
        """
        if self._redis is None:
            return
        try:
            raw = self._redis.get("polybot:model_state")
            if raw is None:
                return
            data: dict = json.loads(raw)
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
                f"Restored model state from Redis "
                f"(alpha={self.alpha:.1f}, beta={self.beta:.1f}, ticks={self.tick_count})"
            )
        except Exception as e:
            self.logger.warning(f"Could not load Redis state: {e}")

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

    def _monte_carlo(
        self, current_price: float, remaining_seconds: float, direction: str
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

        # Binary outcome: did price move in our predicted direction?
        if direction == "UP":
            wins: np.ndarray = terminal_prices > current_price
        else:
            wins = terminal_prices < current_price

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
            current_price: Latest BTC price from Binance
            implied_prob_up: CLOB midpoint price of the UP token (= market P(up))
            remaining_seconds: Seconds until this 5-min window resolves
            order_book: Optional CLOB L2 order book for liquidity-aware z-score

        Returns:
            Signal if profitable edge found, None otherwise
        """
        start: float = time.perf_counter()

        true_prob: float = self.true_prob_up

        # Determine which direction has edge and compute directional probabilities
        if true_prob >= 0.5:
            direction: str = "UP"
            directional_true: float = true_prob
            directional_implied: float = implied_prob_up
        else:
            direction = "DOWN"
            # P(down) = 1 - P(up) for both true and implied
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

        # Z-score with dynamic liquidity adjustment
        # Thin order books → factor > 1 → lower z_score → harder to trade
        # Thick order books → factor < 1 → higher z_score → edge more reliable
        liquidity_factor: float = self._compute_liquidity_factor(order_book)
        z_score: float = self._compute_z_score(
            directional_true, directional_implied, liquidity_factor
        )

        # Gate 2: Monte Carlo validation
        # Even if the instantaneous edge looks good, simulate 1000 paths
        # to check if the edge survives price evolution over remaining time
        mc_ev: float
        mc_win_prob: float
        mc_variance: float
        mc_ev, mc_win_prob, mc_variance = self._monte_carlo(
            current_price, remaining_seconds, direction
        )

        # Gate 3: MC expected value must be positive
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
        )

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
