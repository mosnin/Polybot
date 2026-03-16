"""
bot.py — Main orchestrator for the Polymarket BTC 5-minute trading bot.

Ties together all components into a single async runtime:
- OKXWebSocket → price ticks → BayesianModel → Signal → OrderExecutor
- GammaMarketFinder → market windows → model reset + token selection
- Safety controls: drawdown hard stop, streak pause, compounding gate
- Rich console logging with real-time performance metrics

The architecture runs 5 concurrent coroutines via asyncio.gather:
1. OKX ticker loop (producer: price ticks)
2. Gamma market finder (producer: market windows)
3. Market discovery consumer (resets model per window)
4. Stale order canceller (maintenance: frees locked capital)
5. Trade loop (consumer: tick → signal → order decision)

Target: tick-to-order-decision in under 80ms on standard VPS.

Usage:
    1. Copy .env.example to .env and fill in your keys
    2. pip install -r requirements.txt
    3. python bot.py

    The bot starts in TEST_MODE by default (no real orders).
    Set TEST_MODE=false in .env for live trading.
"""

import asyncio
import datetime
import logging
import multiprocessing
import os
import queue as queue_mod  # for queue.Empty exception
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from config import Config, load_config
from data_feed import OKXWebSocket, GammaMarketFinder, MarketWindow, PriceTick
from executor import OrderExecutor, OrderResult
from model import BayesianModel, Signal
from performance import LatencyMonitor, adaptive_z_score, rebate_optimizer

# Redis is optional — bot works without it, but loses equity history persistence
try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None  # type: ignore[assignment]


@dataclass
class BotState:
    """Mutable bot state tracking performance and safety metrics.

    Only mutated from the trade loop coroutine (single writer),
    so no locks needed in the async architecture.
    """

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    daily_maker_volume: float = 0.0  # cumulative maker volume for rebate tracking
    is_paused: bool = False
    compounding_active: bool = False
    test_mode: bool = False
    # Market-making state
    mm_trades: int = 0
    mm_spread_profit: float = 0.0
    mm_active: bool = False
    # Balance history for daily return calculation (compounding gate).
    # Stores (unix_timestamp, balance) tuples. Capped at 50k entries (~14 hours).
    equity_history: List[Tuple[float, float]] = field(default_factory=list)


class AsyncBot:
    """Main bot class orchestrating all components.

    The bot's positive expectancy comes from three compounding edges:
    1. Information edge: OKX perpetual ticks lead Polymarket CLOB by 1-3s
    2. Statistical edge: Bayesian + MC filters noise, only trades strong signals
    3. Execution edge: Maker limit orders capture rebates, reducing effective cost

    Each edge alone is marginal (~0.5-2%). Combined and compounded over
    hundreds of trades per day, they produce measurable profit.
    """

    def __init__(
        self,
        config: Config,
        test_mode: bool = False,
        event_queue: Optional[multiprocessing.Queue] = None,
        control_queue: Optional[multiprocessing.Queue] = None,
    ) -> None:
        """Initialize all components with shared communication queues.

        Args:
            config: Frozen Config dataclass with all parameters
            test_mode: If True, log signals but don't place real orders
            event_queue: multiprocessing.Queue for pushing events to dashboard
            control_queue: multiprocessing.Queue for receiving dashboard commands
        """
        self.config: Config = config
        self.console: Console = Console()
        self.logger: logging.Logger = logging.getLogger("Bot")

        # Dashboard inter-process queues (None if dashboard disabled)
        self.event_queue: Optional[multiprocessing.Queue] = event_queue
        self.control_queue: Optional[multiprocessing.Queue] = control_queue

        # Dashboard-adjustable exposure override (None = use config default)
        self._exposure_override: Optional[float] = None

        # Shared asyncio queues — the glue between producer and consumer tasks.
        # maxsize prevents unbounded memory growth if consumers fall behind.
        self.tick_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.market_queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        # Component initialization
        self.okx: OKXWebSocket = OKXWebSocket(self.tick_queue)
        self.gamma: GammaMarketFinder = GammaMarketFinder(self.market_queue)
        self.model: BayesianModel = BayesianModel(
            min_edge=config.min_edge_threshold,
            round_trip_cost=config.round_trip_cost_pct,
            redis_url=config.redis_url,
            decay_factor=config.decay_factor,
        )
        self.executor: OrderExecutor = OrderExecutor(
            private_key=config.private_key,
            alchemy_rpc_url=config.alchemy_rpc_url,
            clob_host=config.clob_host,
            chain_id=config.chain_id,
            signature_type=config.signature_type,
            gas_buffer=config.gas_buffer_usdc,
            safety_floor=config.safety_floor_usdc,
        )

        # Mutable state
        self.state: BotState = BotState(test_mode=test_mode)
        self.current_market: Optional[MarketWindow] = None
        self.last_price: Optional[float] = None

        # Cycle latency tracking for performance monitoring
        self.cycle_times: Deque[float] = deque(maxlen=100)

        # Latency monitor for auto-adjusting poll interval during high-vol
        self.latency_monitor: LatencyMonitor = LatencyMonitor(self.logger)

        # Track last trade's prediction for win/loss outcome resolution.
        # Set when an order is placed; resolved when the window ends (new window arrives).
        self._pending_prediction: Optional[dict] = None  # {"direction": str, "entry_price": float}

        # Circuit breaker: halt trading after consecutive API failures
        self._consecutive_api_failures: int = 0
        self._circuit_breaker_until: float = 0.0
        self._CIRCUIT_BREAKER_THRESHOLD: int = 3
        self._CIRCUIT_BREAKER_COOLDOWN: float = 60.0  # seconds

        # Alert state flags (prevent repeated emails)
        self._drawdown_alert_sent: bool = False
        self._latency_alert_sent: bool = False

        # Market-making state (explosive: batch orders, orderflow tracking)
        self._mm_active: bool = False
        self._mm_yes_orders: List[OrderResult] = []
        self._mm_no_orders: List[OrderResult] = []
        self._mm_entry_time: float = 0.0
        self._mm_spread_profit: float = 0.0
        self._last_mm_check: float = 0.0
        self._mm_pending_cancel: bool = False
        self._orderflow_imbalance: float = 0.0
        self._mm_no_opportunity_count: int = 0

        # Stoikov reservation price engine state
        self._last_stoikov_check: float = 0.0
        self._stoikov_state: Optional[dict] = None

        # Redis connection for equity history + bot state persistence (optional).
        # Shares the same Redis URL as the model for state continuity across restarts.
        self._redis = None
        self._redis_save_interval: float = 10.0  # seconds between equity saves
        self._last_redis_save: float = 0.0
        if _redis_lib is not None and config.redis_url:
            try:
                self._redis = _redis_lib.Redis.from_url(
                    config.redis_url, socket_connect_timeout=1, socket_timeout=1
                )
                self._redis.ping()
                self.logger.info("Redis connected for equity history persistence")
                self._load_equity_from_redis()
            except Exception as e:
                self.logger.warning(f"Redis unavailable for equity: {e}")
                self._redis = None

    # --- Safety Controls ---

    def _check_drawdown(self) -> bool:
        """Check if drawdown from peak balance exceeds hard stop threshold.

        Drawdown = (peak - current) / peak. If this exceeds 25%, the bot
        halts all trading. This protects against:
        - Model breakdown (regime change, API issues)
        - Black swan events (flash crashes, exchange outages)
        - Compounding losses from a bad streak

        Returns:
            True if drawdown exceeds limit (bot should stop)
        """
        if self.state.peak_balance <= 0:
            return False
        drawdown: float = (
            (self.state.peak_balance - self.state.current_balance)
            / self.state.peak_balance
        )

        # Early warning at 15% drawdown
        if drawdown >= 0.15 and not self._drawdown_alert_sent:
            self._send_alert(
                "Drawdown Warning",
                f"Drawdown at {drawdown:.1%}. "
                f"Peak: ${self.state.peak_balance:.2f}, "
                f"Current: ${self.state.current_balance:.2f}"
            )
            self._drawdown_alert_sent = True

        # Reset alert flag on balance recovery
        if drawdown < 0.10:
            self._drawdown_alert_sent = False

        if drawdown >= self.config.drawdown_hard_stop_pct:
            self.logger.critical(
                f"HARD STOP: {drawdown:.1%} drawdown "
                f"(peak=${self.state.peak_balance:.2f}, "
                f"current=${self.state.current_balance:.2f})"
            )
            self._send_alert(
                "HARD STOP — Drawdown Limit",
                f"Bot halted at {drawdown:.1%} drawdown. "
                f"Peak: ${self.state.peak_balance:.2f}, "
                f"Current: ${self.state.current_balance:.2f}. "
                "Manual review required."
            )
            return True
        return False

    def _check_streak(self) -> bool:
        """Check if consecutive loss streak exceeds maximum.

        5 consecutive losses is extremely unlikely (~3% probability at 60%
        win rate). If it happens, it suggests the model is miscalibrated
        for current market conditions. We skip the current window and
        wait for the next one, allowing conditions to change.

        Returns:
            True if streak limit hit (should skip current window)
        """
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self.logger.warning(
                f"Streak limit: {self.state.consecutive_losses} "
                f"consecutive losses (max={self.config.max_consecutive_losses})"
            )
            return True
        return False

    def _send_alert(self, subject: str, body: str) -> None:
        """Send email alert via SMTP. Skips silently if SMTP not configured."""
        if not self.config.smtp_host or not self.config.alert_email:
            return
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body)
            msg["Subject"] = f"[PolyBot] {subject}"
            msg["From"] = self.config.smtp_user or "polybot@localhost"
            msg["To"] = self.config.alert_email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=10) as server:
                server.starttls()
                if self.config.smtp_user and self.config.smtp_pass:
                    server.login(self.config.smtp_user, self.config.smtp_pass)
                server.send_message(msg)
            self.logger.info(f"Alert sent: {subject}")
        except Exception as e:
            self.logger.warning(f"Alert email failed: {e}")

    def _compute_avg_daily_return(self) -> float:
        """Compute average daily net return from equity history.

        Groups balance snapshots by calendar day and computes the
        return for each complete day. Returns the average.

        Used as a compounding gate: only activate compounding when
        the bot demonstrates consistent daily profitability (>1%).

        Returns:
            Average daily return as a fraction (0.01 = 1%), or 0.0
            if insufficient data (< 1 complete day).
        """
        history: List[Tuple[float, float]] = self.state.equity_history
        if len(history) < 2:
            return 0.0

        # Group by calendar day
        daily_balances: dict = {}
        for ts, balance in history:
            day: datetime.date = datetime.date.fromtimestamp(ts)
            if day not in daily_balances:
                daily_balances[day] = {"first": balance, "last": balance}
            else:
                daily_balances[day]["last"] = balance

        # Compute daily returns for complete days (exclude today — incomplete)
        today: datetime.date = datetime.date.today()
        daily_returns: list = []
        for day, bal in daily_balances.items():
            if day == today:
                continue  # incomplete day
            if bal["first"] > 0:
                daily_return: float = (bal["last"] - bal["first"]) / bal["first"]
                daily_returns.append(daily_return)

        if not daily_returns:
            return 0.0
        return sum(daily_returns) / len(daily_returns)

    def _update_compounding(self) -> None:
        """Activate compounding — instant in explosive mode.

        When compounding_activation_trades == 0, compounding activates
        immediately from the first trade for maximum growth velocity.
        Otherwise falls back to the triple gate (trades + win rate + daily return).
        """
        if self.state.compounding_active:
            return

        if self.config.compounding_activation_trades == 0:
            # Explosive mode: instant compounding from trade 1
            self.state.compounding_active = True
            self.logger.info("COMPOUNDING ACTIVATED: explosive mode (instant)")
            return

        if self.state.total_trades >= self.config.compounding_activation_trades:
            win_rate: float = self.state.wins / max(self.state.total_trades, 1)
            avg_daily: float = self._compute_avg_daily_return()
            if win_rate > 0.5 and avg_daily > 0.01:
                self.state.compounding_active = True
                self.logger.info(
                    f"COMPOUNDING ACTIVATED: {self.state.total_trades} trades, "
                    f"{win_rate:.1%} win rate, {avg_daily:.2%} avg daily return"
                )

    # --- Stoikov Reservation Price ---

    async def _compute_stoikov(
        self, midpoint: float, remaining: float,
    ) -> Optional[dict]:
        """Compute Stoikov reservation price. Throttled to every 1.5s."""
        now: float = time.time()
        if now - self._last_stoikov_check < self.config.stoikov_check_interval_secs:
            return self._stoikov_state  # return cached
        self._last_stoikov_check = now

        stoikov: dict = self.model.compute_stoikov_reservation_price(
            midpoint=midpoint,
            remaining_secs=remaining,
            gamma=self.config.stoikov_gamma,
        )
        self._stoikov_state = stoikov
        return stoikov

    # --- Daily Rebate Tracker ---

    def _update_rebate_tracker(self, order: OrderResult) -> None:
        """Track cumulative maker volume for rebate estimation.

        Polymarket offers volume-based maker rebates. By tracking our
        daily maker volume, we can estimate the rebate tier we'll qualify
        for and adjust the round_trip_cost assumption accordingly.

        Args:
            order: Successfully posted order to add to volume tracker
        """
        volume: float = order.size * order.price
        self.state.daily_maker_volume += volume

    # --- Dashboard Communication ---

    def _push_event(self, event: dict) -> None:
        """Non-blocking push to the dashboard event queue.

        Uses put_nowait to never block the trading loop. If the queue
        is full or the dashboard process has died, events are silently
        dropped — the bot's performance is always the priority.
        """
        if self.event_queue is not None:
            try:
                self.event_queue.put_nowait(event)
            except Exception:
                pass  # never block the bot for dashboard updates

    def _save_equity_to_redis(self) -> None:
        """Persist equity history and bot state to Redis.

        Throttled to every 10 seconds to avoid Redis overhead on every tick.
        Saves: equity_history, wins, losses, consecutive_losses, total_trades,
        peak_balance, daily_maker_volume, compounding_active.

        Fails silently — never blocks the trading loop for persistence.
        """
        if self._redis is None:
            return
        now: float = time.time()
        if now - self._last_redis_save < self._redis_save_interval:
            return
        try:
            import json
            state: str = json.dumps({
                "equity_history": self.state.equity_history[-50000:],
                "wins": self.state.wins,
                "losses": self.state.losses,
                "consecutive_losses": self.state.consecutive_losses,
                "total_trades": self.state.total_trades,
                "peak_balance": self.state.peak_balance,
                "daily_maker_volume": self.state.daily_maker_volume,
                "compounding_active": self.state.compounding_active,
            })
            self._redis.set("polybot:bot_state", state)
            self._last_redis_save = now
        except Exception:
            pass  # never block the trading loop for persistence

    def _load_equity_from_redis(self) -> None:
        """Restore equity history and bot state from Redis on startup.

        Ensures zero edge history loss across restarts — the compounding
        gate, drawdown tracking, and daily return calculation all resume
        from where they left off.
        """
        if self._redis is None:
            return
        try:
            import json
            raw = self._redis.get("polybot:bot_state")
            if raw is None:
                return
            data: dict = json.loads(raw)
            self.state.equity_history = data.get("equity_history", [])
            self.state.wins = int(data.get("wins", 0))
            self.state.losses = int(data.get("losses", 0))
            self.state.consecutive_losses = int(data.get("consecutive_losses", 0))
            self.state.total_trades = int(data.get("total_trades", 0))
            self.state.peak_balance = float(data.get("peak_balance", 0.0))
            self.state.daily_maker_volume = float(data.get("daily_maker_volume", 0.0))
            self.state.compounding_active = bool(data.get("compounding_active", False))
            self.logger.info(
                f"Restored bot state from Redis "
                f"(trades={self.state.total_trades}, "
                f"wins={self.state.wins}, "
                f"equity_points={len(self.state.equity_history)})"
            )
        except Exception as e:
            self.logger.warning(f"Could not load Redis bot state: {e}")

    def _poll_controls(self) -> None:
        """Non-blocking poll for dashboard control commands.

        Checks the control queue for pause/resume, exposure adjustment,
        and withdraw requests. Processes all available commands in a
        single pass (drains the queue).
        """
        if self.control_queue is None:
            return
        while True:
            try:
                cmd: dict = self.control_queue.get_nowait()
                cmd_type: str = cmd.get("type", "")
                if cmd_type == "pause":
                    self.state.is_paused = True
                    self.logger.info("Dashboard: PAUSED")
                elif cmd_type == "resume":
                    self.state.is_paused = False
                    self.state.consecutive_losses = 0  # reset streak on manual resume
                    self.logger.info("Dashboard: RESUMED")
                elif cmd_type == "set_exposure":
                    value: float = cmd.get("value", self.config.max_exposure_pct)
                    self._exposure_override = max(0.05, min(value, 0.10))
                    self.logger.info(
                        f"Dashboard: exposure cap set to {self._exposure_override:.0%}"
                    )
                elif cmd_type == "withdraw":
                    amount: float = cmd.get("amount", 0.0)
                    self.logger.info(
                        f"Dashboard: withdraw request ${amount:.2f} "
                        "(not yet implemented — manual transfer required)"
                    )
            except queue_mod.Empty:
                break

    # --- Rich Console Logging ---

    def _build_status_table(
        self, signal: Optional[Signal] = None, cycle_ms: float = 0
    ) -> Table:
        """Build a Rich table showing current bot status and signal metrics.

        This table is printed after every tick processing cycle, giving
        real-time visibility into the bot's decision-making process.

        Args:
            signal: Current signal (if any) for detailed metrics
            cycle_ms: Time taken for this tick-to-decision cycle

        Returns:
            Rich Table object ready for console.print()
        """
        table = Table(title="PolyBot Status", show_lines=True)
        table.add_column("Metric", style="cyan", min_width=18)
        table.add_column("Value", style="green", min_width=20)

        # Portfolio metrics
        table.add_row("Balance", f"${self.state.current_balance:.2f}")
        table.add_row("Peak Balance", f"${self.state.peak_balance:.2f}")
        drawdown: float = 0.0
        if self.state.peak_balance > 0:
            drawdown = (
                (self.state.peak_balance - self.state.current_balance)
                / self.state.peak_balance
            )
        table.add_row("Drawdown", f"{drawdown:.1%}")

        # Performance metrics
        table.add_row("Total Trades", str(self.state.total_trades))
        win_rate: float = self.state.wins / max(self.state.total_trades, 1)
        table.add_row("Win Rate", f"{win_rate:.1%}")
        table.add_row("Consec. Losses", str(self.state.consecutive_losses))
        table.add_row("Compounding", "ON" if self.state.compounding_active else "OFF")

        # Latency metrics (from LatencyMonitor)
        table.add_row("Cycle Latency", f"{cycle_ms:.1f}ms")
        if self.latency_monitor.cycle_times:
            table.add_row("Avg Latency", f"{self.latency_monitor.avg_ms:.1f}ms")
            table.add_row(
                "Poll Interval",
                f"{self.latency_monitor.poll_interval:.1f}s"
                + (" [HIGH-VOL]" if self.latency_monitor.poll_interval < 1.0 else ""),
            )

        # Rebate tracking
        table.add_row("Maker Volume", f"${self.state.daily_maker_volume:.2f}")

        # Signal metrics (if signal was generated this cycle)
        if signal:
            table.add_row("---", "--- Signal ---")
            table.add_row("Direction", signal.direction)
            table.add_row("Edge", f"{signal.edge:.4f}")
            table.add_row("Z-Score", f"{signal.z_score:.2f}")
            table.add_row("MC EV", f"{signal.mc_ev:.4f}")
            table.add_row("MC Win Prob", f"{signal.mc_win_prob:.1%}")
            table.add_row("MC Variance", f"{signal.mc_variance:.6f}")
            table.add_row("Kelly Fraction", f"{signal.kelly_fraction:.3f}")
            table.add_row("Model Latency", f"{signal.computation_ms:.1f}ms")
            # Projected profit per trade = edge * position_size
            projected_profit: float = signal.edge * (
                self.state.current_balance * signal.kelly_fraction
            )
            table.add_row("Projected P/L", f"${projected_profit:.2f}")

        # Market window info
        if self.current_market:
            remaining: float = self.current_market.end_timestamp - time.time()
            table.add_row("---", "--- Market ---")
            table.add_row("Window", self.current_market.slug)
            table.add_row("Remaining", f"{remaining:.0f}s")
        else:
            table.add_row("Market", "Waiting for window...")

        # Mode indicator
        if self.state.test_mode:
            table.add_row("MODE", "[bold red]TEST (no real orders)[/bold red]")

        return table

    # --- Core Async Loops ---

    async def _market_discovery_loop(self) -> None:
        """Consume new market windows from GammaMarketFinder.

        When a new 5-minute window is discovered:
        1. Update current_market reference
        2. Reset the Bayesian model (clear stale prior)
        3. Clear last_price (new window = fresh start)

        This ensures each window is traded independently with no
        information leakage from previous windows.
        """
        while True:
            market: MarketWindow = await self.market_queue.get()

            # Resolve the previous window's prediction outcome if we traded it.
            # Compare exit price (last known price at window end) to entry price.
            if self._pending_prediction is not None and self.last_price is not None:
                pred = self._pending_prediction
                price_went_up: bool = self.last_price > pred["entry_price"]
                if pred["direction"] == "UP":
                    won: bool = price_went_up
                else:
                    won = not price_went_up

                if won:
                    self.state.wins += 1
                    self.state.consecutive_losses = 0
                    self.logger.info("Previous window outcome: WIN")
                else:
                    self.state.losses += 1
                    self.state.consecutive_losses += 1
                    self.logger.info(
                        f"Previous window outcome: LOSS "
                        f"(streak={self.state.consecutive_losses})"
                    )
                self._pending_prediction = None

            self.current_market = market
            self.model.reset()
            self.last_price = None
            self.logger.info(
                f"New market window: {market.slug} "
                f"(ends in {market.end_timestamp - time.time():.0f}s)"
            )

    async def _stale_order_loop(self) -> None:
        """Cancel unfilled orders every 10 seconds.

        In a 5-minute market, an order that hasn't filled within 10 seconds
        is likely at a stale price. Cancelling frees up locked capital for
        new, more accurate signals.

        Uses run_in_executor because py-clob-client cancel is synchronous HTTP.
        """
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        while True:
            await asyncio.sleep(self.config.stale_order_cancel_interval)
            await loop.run_in_executor(None, self.executor.cancel_stale_orders)

    async def _check_mm_opportunity(
        self, loop: asyncio.AbstractEventLoop
    ) -> bool:
        """Check for and execute a market-making spread opportunity (explosive mode).

        Fetches both YES and NO midpoints + order book in parallel, checks spread
        threshold (0.985), computes order-flow imbalance, and places batch orders
        (up to 5 per side) at staggered price levels.

        Throttled to every 2 seconds. Prioritized before directional trading.

        Returns:
            True if MM batch was placed (caller should skip directional)
        """
        if not self.config.mm_enabled:
            return False
        if self.current_market is None:
            return False
        if self._mm_active:
            return False  # already in an MM position

        now: float = time.time()
        if now - self._last_mm_check < self.config.mm_check_interval_secs:
            return False
        self._last_mm_check = now

        remaining: float = self.current_market.end_timestamp - now
        if remaining < self.config.mm_min_remaining_secs:
            return False

        # Fetch both midpoints + YES order book in parallel
        try:
            yes_mid, no_mid, order_book = await asyncio.gather(
                loop.run_in_executor(
                    None, self.executor.get_midpoint,
                    self.current_market.yes_token_id,
                ),
                loop.run_in_executor(
                    None, self.executor.get_midpoint,
                    self.current_market.no_token_id,
                ),
                loop.run_in_executor(
                    None, self.executor.get_order_book,
                    self.current_market.yes_token_id,
                ),
            )
            yes_mid = float(yes_mid)
            no_mid = float(no_mid)
            self._consecutive_api_failures = 0
        except Exception as e:
            self.logger.warning(f"MM midpoint fetch failed: {e}")
            return False

        # Compute Stoikov reservation prices for both sides
        stoikov_yes = self.model.compute_stoikov_reservation_price(
            midpoint=yes_mid,
            remaining_secs=remaining,
            gamma=self.config.stoikov_gamma,
        )
        stoikov_no = self.model.compute_stoikov_reservation_price(
            midpoint=no_mid,
            remaining_secs=remaining,
            gamma=self.config.stoikov_gamma,
        )
        self._stoikov_state = stoikov_yes  # cache for dashboard

        mm_signal = self.model.check_market_making_opportunity(
            yes_midpoint=yes_mid,
            no_midpoint=no_mid,
            spread_threshold=self.config.mm_spread_threshold,
            batch_size=self.config.mm_batch_size,
            stoikov_yes=stoikov_yes,
            stoikov_no=stoikov_no,
        )

        if mm_signal is None:
            self._mm_no_opportunity_count += 1
            return False

        # Compute order-flow imbalance for side bias
        flow = self.model.compute_orderflow_imbalance(
            order_book,
            threshold=self.config.orderflow_imbalance_threshold,
        )
        self._orderflow_imbalance = flow["imbalance"]

        self.logger.info(
            f"MM STOIKOV: spread={mm_signal['spread']:.4f} "
            f"r_yes={stoikov_yes['reservation_price']:.3f} "
            f"r_no={stoikov_no['reservation_price']:.3f} "
            f"σ={stoikov_yes['sigma']:.6f} T-t={remaining:.0f}s "
            f"flow={flow['imbalance']:+.3f}"
        )

        if self.state.test_mode:
            self.logger.info("[TEST MODE] MM opportunity detected, not placing orders")
            return True

        # Place batch orders on both sides
        yes_orders, no_orders = await loop.run_in_executor(
            None,
            self.executor.place_mm_batch,
            self.current_market.yes_token_id,
            self.current_market.no_token_id,
            mm_signal["batch_levels"],
            self.state.current_balance,
            self.config.mm_exposure_pct,
            self.config.mm_batch_size,
        )

        if not yes_orders and not no_orders:
            self.logger.warning("MM batch: all orders failed, aborting")
            return False

        # Need at least one order on each side for valid MM
        if not yes_orders or not no_orders:
            orphans = yes_orders or no_orders
            for orphan in orphans:
                await loop.run_in_executor(
                    None, self.executor.cancel_order, orphan.order_id
                )
            self.logger.warning("MM batch: one side empty, cancelled orphans")
            return False

        self._mm_active = True
        self._mm_yes_orders = yes_orders
        self._mm_no_orders = no_orders
        self._mm_entry_time = time.time()
        self._mm_pending_cancel = False
        self._mm_no_opportunity_count = 0
        self.state.mm_active = True
        self.state.mm_trades += 1
        self.state.total_trades += 1

        total_yes_size: float = sum(o.size for o in yes_orders)
        total_no_size: float = sum(o.size for o in no_orders)
        gas_total: float = self.config.gas_buffer_usdc * (len(yes_orders) + len(no_orders))

        # Push MM trade event to dashboard
        self._push_event({
            "type": "trade",
            "timestamp": time.time(),
            "direction": "MM-SPREAD",
            "edge": mm_signal["spread"],
            "z_score": 0.0,
            "model_prob": 0.0,
            "implied_prob": mm_signal["mid_sum"],
            "kelly_fraction": 0.0,
            "fill_price": mm_signal["yes_price"] + mm_signal["no_price"],
            "size": total_yes_size + total_no_size,
            "mc_ev": mm_signal["expected_profit"],
            "outcome": None,
            "gas_paid": gas_total,
            "net_pnl": None,
            "mm_spread": mm_signal["spread"],
            "orderflow_imbalance": flow["imbalance"],
            "batch_count": len(yes_orders) + len(no_orders),
            "stoikov_r": stoikov_yes["reservation_price"],
            "stoikov_sigma": stoikov_yes["sigma"],
        })

        return True

    async def _manage_mm_position(
        self, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Manage active MM batch position: cancel losing side after delay.

        After cancel_delay_secs (30s), uses BOTH Bayesian model direction
        AND order-flow imbalance to determine winner. If flow imbalance > 20%,
        flow overrides model direction ("ride that side harder").

        Cancels ALL orders on the losing side, keeps all on the winning side.
        """
        if not self._mm_active:
            return

        elapsed: float = time.time() - self._mm_entry_time

        # After cancel delay, determine winner and cancel loser
        if elapsed >= self.config.mm_cancel_delay_secs and not self._mm_pending_cancel:
            self._mm_pending_cancel = True

            # Default: use Bayesian model direction
            keep_direction: str = "UP" if self.model.true_prob_up >= 0.5 else "DOWN"

            # Override with order-flow imbalance if strong enough (>20%)
            if abs(self._orderflow_imbalance) >= self.config.orderflow_imbalance_threshold:
                flow_direction: str = "UP" if self._orderflow_imbalance > 0 else "DOWN"
                if flow_direction != keep_direction:
                    self.logger.info(
                        f"MM FLOW OVERRIDE: model={keep_direction} "
                        f"flow={flow_direction} (imbalance={self._orderflow_imbalance:+.3f})"
                    )
                    keep_direction = flow_direction

            # Cancel all orders on losing side
            if keep_direction == "UP":
                cancel_orders = self._mm_no_orders
                keep_orders = self._mm_yes_orders
            else:
                cancel_orders = self._mm_yes_orders
                keep_orders = self._mm_no_orders

            for order in cancel_orders:
                await loop.run_in_executor(
                    None, self.executor.cancel_order, order.order_id
                )

            # Compute spread profit across all batch levels
            avg_yes_price: float = (
                sum(o.price * o.size for o in self._mm_yes_orders)
                / max(sum(o.size for o in self._mm_yes_orders), 0.01)
            )
            avg_no_price: float = (
                sum(o.price * o.size for o in self._mm_no_orders)
                / max(sum(o.size for o in self._mm_no_orders), 0.01)
            )
            total_gas: float = self.config.gas_buffer_usdc * (
                len(self._mm_yes_orders) + len(self._mm_no_orders)
            )
            spread_pnl: float = (
                1.0 - (avg_yes_price + avg_no_price)
                - self.config.round_trip_cost_pct - total_gas
            )
            yes_value: float = sum(o.size * o.price for o in self._mm_yes_orders)
            no_value: float = sum(o.size * o.price for o in self._mm_no_orders)
            position_value: float = min(yes_value, no_value)
            spread_pnl_usdc: float = spread_pnl * position_value

            self._mm_spread_profit += spread_pnl_usdc
            self.state.mm_spread_profit += spread_pnl_usdc

            self.logger.info(
                f"MM CANCEL: dropped {len(cancel_orders)} orders, "
                f"keeping {keep_direction} ({len(keep_orders)} orders) | "
                f"spread P&L: ${spread_pnl_usdc:.4f}"
            )

            # Record pending prediction so kept side resolves like directional
            self._pending_prediction = {
                "direction": keep_direction,
                "entry_price": self.last_price,
            }

            # Reset MM state
            self._mm_active = False
            self._mm_yes_orders = []
            self._mm_no_orders = []
            self.state.mm_active = False
            return

        # Safety: if window is about to expire, cancel both sides
        if self.current_market is not None:
            remaining: float = self.current_market.end_timestamp - time.time()
            if remaining < 15.0:
                self.logger.warning("MM: window expiring, cancelling all batch orders")
                for order in self._mm_yes_orders + self._mm_no_orders:
                    await loop.run_in_executor(
                        None, self.executor.cancel_order, order.order_id
                    )
                self._mm_active = False
                self._mm_yes_orders = []
                self._mm_no_orders = []
                self.state.mm_active = False

    async def _trade_loop(self) -> None:
        """Main tick-to-decision pipeline. The core trading loop.

        For each tick from OKX WebSocket:
        1. Update Bayesian model with new price observation
        2. Check if active market exists with sufficient time remaining
        3. Run safety checks (drawdown, streak)
        4. Fetch CLOB midpoint for implied probability
        5. Run model.evaluate() — Bayesian + MC + Kelly pipeline
        6. If signal generated: place maker limit order
        7. Log status via Rich table

        Target: <80ms from tick arrival to order decision.
        Bottleneck: CLOB midpoint HTTP call (~20-50ms via run_in_executor).
        Model evaluation: <5ms (vectorized numpy).
        """
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

        while True:
            cycle_start: float = time.perf_counter()

            # Poll dashboard control commands (non-blocking)
            self._poll_controls()

            # If paused by dashboard, idle without consuming ticks aggressively
            if self.state.is_paused:
                await asyncio.sleep(0.5)
                continue

            # Circuit breaker: skip cycle if cooling down after consecutive API failures
            if time.time() < self._circuit_breaker_until:
                await asyncio.sleep(1.0)
                continue

            # Block until next tick arrives from OKX
            tick: PriceTick = await self.tick_queue.get()

            # Step 1: Update Bayesian model with new price evidence
            self.model.update(tick.last_price, self.last_price)
            self.last_price = tick.last_price

            # Step 2: Check for active market
            if self.current_market is None:
                continue  # no market window yet, keep consuming ticks

            remaining: float = self.current_market.end_timestamp - time.time()

            # Step 3: Don't enter too close to expiry — insufficient time
            # for maker fill + price movement to realize the edge
            if remaining < self.config.min_remaining_window_secs:
                continue

            # Step 4: Safety checks
            # Fetch balance via thread executor (sync HTTP call).
            # Fall back to last known balance on RPC failure to avoid crashing.
            try:
                self.state.current_balance = await loop.run_in_executor(
                    None, self.executor.get_current_balance
                )
                self._consecutive_api_failures = 0
            except Exception as e:
                self.logger.warning(
                    f"Balance fetch failed, using last known: {e}"
                )
                self._consecutive_api_failures += 1
                if self._consecutive_api_failures >= self._CIRCUIT_BREAKER_THRESHOLD:
                    self._circuit_breaker_until = time.time() + self._CIRCUIT_BREAKER_COOLDOWN
                    self.logger.critical(
                        f"CIRCUIT BREAKER: {self._consecutive_api_failures} consecutive "
                        f"API failures. Cooling down for {self._CIRCUIT_BREAKER_COOLDOWN:.0f}s."
                    )
                    self._send_alert(
                        "Circuit Breaker Activated",
                        f"{self._consecutive_api_failures} consecutive API failures. "
                        f"Trading paused for {self._CIRCUIT_BREAKER_COOLDOWN:.0f}s."
                    )
                    continue
            self.state.peak_balance = max(
                self.state.peak_balance, self.state.current_balance
            )

            # Track balance for daily return calculation (compounding gate)
            self.state.equity_history.append(
                (time.time(), self.state.current_balance)
            )
            if len(self.state.equity_history) > 50000:
                self.state.equity_history = self.state.equity_history[-50000:]

            # Push live balance to dashboard
            self._push_event({
                "type": "balance",
                "timestamp": time.time(),
                "balance": self.state.current_balance,
            })

            # Hard stop: drawdown exceeds 25%
            if self._check_drawdown():
                self.state.is_paused = True
                self.logger.critical(
                    "Bot HALTED: drawdown hard stop triggered. "
                    "Manual review required."
                )
                # Cancel all open orders before stopping
                await loop.run_in_executor(None, self.executor.cancel_all)
                break

            # Streak check disabled in explosive mode (max_consecutive_losses=999)
            # Kept as safety net — only triggers if config overridden via .env
            if self.config.max_consecutive_losses < 999 and self._check_streak():
                continue

            # Step 4b: Market-making spread check (prioritized before directional)
            # Manages existing MM position if active, then checks for new opportunity.
            if self._mm_active:
                await self._manage_mm_position(loop)

            mm_triggered: bool = await self._check_mm_opportunity(loop)
            if mm_triggered:
                cycle_ms = (time.perf_counter() - cycle_start) * 1000
                signal = None  # no directional signal on MM cycles

            # Directional as tie-breaker: only fire when MM didn't find opportunity
            # for 3+ consecutive checks (MM is the primary strategy)
            if not mm_triggered and self._mm_no_opportunity_count >= 3:
                # Step 5: Get implied probability + order book in parallel
                # Parallelizing these two HTTP calls via asyncio.gather keeps
                # total I/O at max(midpoint_ms, orderbook_ms) instead of the sum,
                # preserving the <80ms cycle target.
                try:
                    token_id_for_fetch: str = self.current_market.yes_token_id
                    midpoint_result, order_book = await asyncio.gather(
                        loop.run_in_executor(
                            None,
                            self.executor.get_midpoint,
                            token_id_for_fetch,
                        ),
                        loop.run_in_executor(
                            None,
                            self.executor.get_order_book,
                            token_id_for_fetch,
                        ),
                    )
                    implied_prob_up: float = float(midpoint_result)
                except Exception as e:
                    self.logger.error(f"Midpoint/orderbook fetch failed: {e}")
                    self._consecutive_api_failures += 1
                    if self._consecutive_api_failures >= self._CIRCUIT_BREAKER_THRESHOLD:
                        self._circuit_breaker_until = time.time() + self._CIRCUIT_BREAKER_COOLDOWN
                        self.logger.critical(
                            f"CIRCUIT BREAKER: {self._consecutive_api_failures} consecutive "
                            f"API failures. Cooling down for {self._CIRCUIT_BREAKER_COOLDOWN:.0f}s."
                        )
                        self._send_alert(
                            "Circuit Breaker Activated",
                            f"{self._consecutive_api_failures} consecutive API failures. "
                            f"Trading paused for {self._CIRCUIT_BREAKER_COOLDOWN:.0f}s."
                        )
                    continue

                # Step 6: Compute adaptive min-edge based on real-time CLOB depth.
                dynamic_min_edge: float = adaptive_z_score(
                    order_book,
                    base_min_edge=self.config.min_edge_threshold,
                    high_depth_threshold=self.config.rebate_depth_threshold,
                )

                # Run full evaluation pipeline with liquidity-aware z-score
                signal: Optional[Signal] = self.model.evaluate(
                    current_price=tick.last_price,
                    implied_prob_up=implied_prob_up,
                    remaining_seconds=remaining,
                    order_book=order_book,
                    min_edge_override=dynamic_min_edge,
                )

                cycle_ms: float = (time.perf_counter() - cycle_start) * 1000

                # Step 7: Execute if signal exists
                if signal:
                    # Select the correct token based on predicted direction
                    if signal.direction == "UP":
                        token_id: str = self.current_market.yes_token_id
                    else:
                        token_id = self.current_market.no_token_id

                    # Compute Stoikov reservation price for directional orders
                    stoikov = await self._compute_stoikov(implied_prob_up, remaining)
                    stoikov_price: Optional[float] = None
                    if stoikov is not None:
                        stoikov_price = stoikov["optimal_bid"]

                    if self.state.test_mode:
                        self.logger.info(
                            f"[TEST MODE] Signal: {signal.direction} | "
                            f"edge={signal.edge:.4f} | "
                            f"kelly={signal.kelly_fraction:.3f} | "
                            f"mc_ev={signal.mc_ev:.4f} | "
                            f"stoikov_r={stoikov['reservation_price'] if stoikov else 'N/A'}"
                        )
                        self.state.total_trades += 1
                    else:
                        # Live mode: place the actual order with Stoikov pricing
                        exposure: float = (
                            self._exposure_override or self.config.max_exposure_pct
                        )

                        balance_cap: Optional[float] = None
                        if not self.state.compounding_active:
                            balance_cap = self.config.starting_capital

                        self._pending_prediction = {
                            "direction": signal.direction,
                            "entry_price": tick.last_price,
                        }

                        rebate_cfg: dict = rebate_optimizer(
                            order_book,
                            midpoint=implied_prob_up,
                            depth_threshold=self.config.rebate_depth_threshold,
                            rebate_offset=self.config.rebate_price_offset,
                        )

                        use_low_vol_batch: bool = (
                            self.model.volatility < self.config.low_vol_threshold
                            and self.model.volatility > 0
                            and not rebate_cfg["use_rebate_batch"]
                        )

                        if rebate_cfg["use_rebate_batch"]:
                            results = await loop.run_in_executor(
                                None,
                                self.executor.place_rebate_optimized_orders,
                                token_id,
                                signal.direction,
                                signal.kelly_fraction,
                                implied_prob_up,
                                rebate_cfg,
                                exposure,
                                balance_cap,
                            )
                        elif use_low_vol_batch:
                            results = await loop.run_in_executor(
                                None,
                                self.executor.place_batch_maker_limits,
                                token_id,
                                signal.direction,
                                signal.kelly_fraction,
                                implied_prob_up,
                                exposure,
                                3,
                                balance_cap,
                            )
                        else:
                            single_result: Optional[OrderResult] = await loop.run_in_executor(
                                None,
                                self.executor.place_maker_limit,
                                token_id,
                                signal.direction,
                                signal.kelly_fraction,
                                implied_prob_up,
                                exposure,
                                balance_cap,
                                stoikov_price,
                            )
                            results = [single_result] if single_result else []

                        for order_result in results:
                            self.state.total_trades += 1
                            self._update_rebate_tracker(order_result)
                            self._update_compounding()
                            self._push_event({
                                "type": "trade",
                                "timestamp": time.time(),
                                "direction": signal.direction,
                                "edge": signal.edge,
                                "z_score": signal.z_score,
                                "model_prob": signal.true_prob,
                                "implied_prob": signal.implied_prob,
                                "kelly_fraction": signal.kelly_fraction,
                                "fill_price": order_result.price,
                                "size": order_result.size,
                                "mc_ev": signal.mc_ev,
                                "outcome": None,
                                "gas_paid": self.config.gas_buffer_usdc,
                                "net_pnl": None,
                            })

                        # Reset MM miss counter — directional trade fired as tie-breaker
                        self._mm_no_opportunity_count = 0

            # Step 8: Latency monitor — record cycle, warn if >80ms, auto-tune poll
            self.latency_monitor.record(cycle_ms)
            self.latency_monitor.auto_adjust_poll_interval(self.model.volatility)

            # Alert on sustained latency breach (>80ms avg over 50+ cycles)
            if (
                len(self.latency_monitor.cycle_times) >= 50
                and self.latency_monitor.avg_ms > 80.0
                and not self._latency_alert_sent
            ):
                self._send_alert(
                    "Latency Breach",
                    f"Avg cycle latency: {self.latency_monitor.avg_ms:.1f}ms "
                    f"over {len(self.latency_monitor.cycle_times)} cycles (target: <60ms)."
                )
                self._latency_alert_sent = True

            # Step 9: Rich console output
            table: Table = self._build_status_table(signal, cycle_ms)
            self.console.print(table)
            self.cycle_times.append(cycle_ms)

            # Step 10: Persist equity history to Redis (throttled to every 10s)
            self._save_equity_to_redis()

            # Push status snapshot to dashboard
            self._push_event({
                "type": "status",
                "timestamp": time.time(),
                "total_trades": self.state.total_trades,
                "wins": self.state.wins,
                "losses": self.state.losses,
                "consecutive_losses": self.state.consecutive_losses,
                "peak_balance": self.state.peak_balance,
                "current_balance": self.state.current_balance,
                "is_paused": self.state.is_paused,
                "compounding_active": self.state.compounding_active,
                "cycle_ms": cycle_ms,
                "mm_active": self.state.mm_active,
                "mm_trades": self.state.mm_trades,
                "mm_spread_profit": self.state.mm_spread_profit,
                "orderflow_imbalance": self._orderflow_imbalance,
                "explosive_mode": self.config.mm_spread_threshold < 0.99,
                "starting_capital": self.config.starting_capital,
                "stoikov_r": self._stoikov_state.get("reservation_price") if self._stoikov_state else None,
                "stoikov_sigma": self._stoikov_state.get("sigma") if self._stoikov_state else None,
                "stoikov_gamma": self.config.stoikov_gamma,
                "stoikov_remaining": self._stoikov_state.get("remaining_secs") if self._stoikov_state else None,
                "stoikov_skew": self._stoikov_state.get("inventory_skew") if self._stoikov_state else None,
            })

    async def run(self) -> None:
        """Main entry point — launches all concurrent async tasks.

        The 5 coroutines run indefinitely via asyncio.gather:
        - 2 producers (OKX ticks, Gamma markets)
        - 3 consumers (market discovery, stale order cleanup, trade loop)

        KeyboardInterrupt (Ctrl+C) triggers graceful shutdown:
        cancel all orders, close WebSocket, stop market finder.
        """
        self.logger.info("=" * 60)
        self.logger.info("  PolyBot Starting...")
        self.logger.info(f"  Test Mode: {self.state.test_mode}")
        self.logger.info(f"  Min Edge: {self.config.min_edge_threshold}")
        self.logger.info(f"  Max Exposure: {self.config.max_exposure_pct:.0%}")
        self.logger.info(f"  Drawdown Stop: {self.config.drawdown_hard_stop_pct:.0%}")
        self.logger.info("=" * 60)

        # Initialize starting balance
        try:
            self.state.current_balance = self.executor.get_current_balance()
        except Exception as e:
            self.logger.warning(
                f"Could not fetch initial balance: {e}. "
                f"Using starting_capital={self.config.starting_capital}"
            )
            self.state.current_balance = self.config.starting_capital

        self.state.peak_balance = self.state.current_balance
        self.logger.info(f"Starting balance: ${self.state.current_balance:.2f}")

        # Connect OKX WebSocket
        await self.okx.connect()

        try:
            # Launch all 5 concurrent tasks
            await asyncio.gather(
                self.okx.run_ticker_loop(),  # Producer: OKX price ticks
                self.gamma.run(),  # Producer: Polymarket market windows
                self._market_discovery_loop(),  # Consumer: update current market
                self._stale_order_loop(),  # Maintenance: cancel stale orders
                self._trade_loop(),  # Consumer: tick → signal → order
            )
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
        finally:
            # Graceful shutdown: clean up all resources
            self.logger.info("Shutting down...")
            await self.okx.close()
            await self.gamma.close()
            self.executor.cancel_all()

            # Final stats
            self.logger.info("=" * 60)
            self.logger.info("  Final Statistics:")
            self.logger.info(f"  Total Trades: {self.state.total_trades}")
            win_rate: float = self.state.wins / max(self.state.total_trades, 1)
            self.logger.info(f"  Win Rate: {win_rate:.1%}")
            self.logger.info(f"  Final Balance: ${self.state.current_balance:.2f}")
            self.logger.info(
                f"  Maker Volume: ${self.state.daily_maker_volume:.2f}"
            )
            self.logger.info("=" * 60)


async def main() -> None:
    """Entry point: load config, create bot, run.

    If ENABLE_DASHBOARD=true in .env, launches a Streamlit dashboard
    in a separate process connected via multiprocessing queues.
    """
    load_dotenv(override=True)

    # Configure logging with timestamps for latency analysis
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config: Config = load_config()
    test_mode: bool = os.getenv("TEST_MODE", "true").lower() == "true"
    enable_dashboard: bool = os.getenv("ENABLE_DASHBOARD", "false").lower() == "true"

    # Create inter-process queues for dashboard communication
    event_queue: Optional[multiprocessing.Queue] = None
    control_queue: Optional[multiprocessing.Queue] = None
    dashboard_proc: Optional[multiprocessing.Process] = None

    if enable_dashboard:
        event_queue = multiprocessing.Queue(maxsize=10000)
        control_queue = multiprocessing.Queue(maxsize=100)

        from dashboard import run_dashboard

        dashboard_proc = multiprocessing.Process(
            target=run_dashboard,
            args=(event_queue, control_queue, config.private_key, config.alchemy_rpc_url),
            daemon=True,  # auto-killed when main process exits
        )
        dashboard_proc.start()
        logging.getLogger("Bot").info(
            "Dashboard launched at http://localhost:8501"
        )

    bot: AsyncBot = AsyncBot(
        config,
        test_mode=test_mode,
        event_queue=event_queue,
        control_queue=control_queue,
    )
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
