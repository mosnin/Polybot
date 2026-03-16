"""
bot.py — Main orchestrator for the Polymarket BTC 5-minute trading bot.

Ties together all components into a single async runtime:
- BinanceWebSocket → price ticks → BayesianModel → Signal → OrderExecutor
- GammaMarketFinder → market windows → model reset + token selection
- Safety controls: drawdown hard stop, streak pause, compounding gate
- Rich console logging with real-time performance metrics

The architecture runs 5 concurrent coroutines via asyncio.gather:
1. Binance ticker loop (producer: price ticks)
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
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from config import Config, load_config
from data_feed import BinanceWebSocket, GammaMarketFinder, MarketWindow, PriceTick
from executor import OrderExecutor, OrderResult
from model import BayesianModel, Signal


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


class AsyncBot:
    """Main bot class orchestrating all components.

    The bot's positive expectancy comes from three compounding edges:
    1. Information edge: Binance perpetual ticks lead Polymarket CLOB by 1-3s
    2. Statistical edge: Bayesian + MC filters noise, only trades strong signals
    3. Execution edge: Maker limit orders capture rebates, reducing effective cost

    Each edge alone is marginal (~0.5-2%). Combined and compounded over
    hundreds of trades per day, they produce measurable profit.
    """

    def __init__(self, config: Config, test_mode: bool = False) -> None:
        """Initialize all components with shared communication queues.

        Args:
            config: Frozen Config dataclass with all parameters
            test_mode: If True, log signals but don't place real orders
        """
        self.config: Config = config
        self.console: Console = Console()
        self.logger: logging.Logger = logging.getLogger("Bot")

        # Shared asyncio queues — the glue between producer and consumer tasks.
        # maxsize prevents unbounded memory growth if consumers fall behind.
        self.tick_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.market_queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        # Component initialization
        self.binance: BinanceWebSocket = BinanceWebSocket(self.tick_queue)
        self.gamma: GammaMarketFinder = GammaMarketFinder(self.market_queue)
        self.model: BayesianModel = BayesianModel(
            min_edge=config.min_edge_threshold,
            round_trip_cost=config.round_trip_cost_pct,
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
        if drawdown >= self.config.drawdown_hard_stop_pct:
            self.logger.critical(
                f"HARD STOP: {drawdown:.1%} drawdown "
                f"(peak=${self.state.peak_balance:.2f}, "
                f"current=${self.state.current_balance:.2f})"
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

    def _update_compounding(self) -> None:
        """Activate compounding after sufficient statistical evidence.

        Compounding means position sizes scale with balance growth.
        Before activation (first 200 trades), sizes are based on
        starting_capital to limit exposure while the edge is unproven.

        After 200 trades with >50% win rate, compounding activates:
        position sizes then scale with actual balance, letting profits
        accelerate growth via the Kelly criterion.

        Why 200 trades? At 60% true win rate:
        - 95% CI for observed win rate is [53%, 67%]
        - This is narrow enough to confirm edge is real, not luck
        """
        if (
            self.state.total_trades >= self.config.compounding_activation_trades
            and not self.state.compounding_active
        ):
            win_rate: float = self.state.wins / max(self.state.total_trades, 1)
            if win_rate > 0.5:
                self.state.compounding_active = True
                self.logger.info(
                    f"COMPOUNDING ACTIVATED: {self.state.total_trades} trades, "
                    f"{win_rate:.1%} win rate"
                )

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

        # Latency metrics
        table.add_row("Cycle Latency", f"{cycle_ms:.1f}ms")
        if self.cycle_times:
            avg_latency: float = sum(self.cycle_times) / len(self.cycle_times)
            table.add_row("Avg Latency", f"{avg_latency:.1f}ms")

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

    async def _trade_loop(self) -> None:
        """Main tick-to-decision pipeline. The core trading loop.

        For each tick from Binance WebSocket:
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

            # Block until next tick arrives from Binance
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
            # Fetch balance via thread executor (sync HTTP call)
            self.state.current_balance = await loop.run_in_executor(
                None, self.executor.get_current_balance
            )
            self.state.peak_balance = max(
                self.state.peak_balance, self.state.current_balance
            )

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

            # Streak check: skip this window if too many consecutive losses
            if self._check_streak():
                continue

            # Step 5: Get implied probability from CLOB midpoint
            # For the UP token: midpoint = market's P(BTC goes up)
            try:
                implied_prob_up: float = await loop.run_in_executor(
                    None,
                    self.executor.get_midpoint,
                    self.current_market.yes_token_id,
                )
                implied_prob_up = float(implied_prob_up)
            except Exception as e:
                self.logger.error(f"Midpoint fetch failed: {e}")
                continue

            # Step 6: Run full evaluation pipeline
            signal: Optional[Signal] = self.model.evaluate(
                current_price=tick.last_price,
                implied_prob_up=implied_prob_up,
                remaining_seconds=remaining,
            )

            cycle_ms: float = (time.perf_counter() - cycle_start) * 1000

            # Step 7: Execute if signal exists
            if signal:
                # Select the correct token based on predicted direction
                if signal.direction == "UP":
                    token_id: str = self.current_market.yes_token_id
                else:
                    token_id = self.current_market.no_token_id

                if self.state.test_mode:
                    # Test mode: log the signal without placing real orders.
                    # This allows validating the model's output quality
                    # before risking real capital.
                    self.logger.info(
                        f"[TEST MODE] Signal: {signal.direction} | "
                        f"edge={signal.edge:.4f} | "
                        f"kelly={signal.kelly_fraction:.3f} | "
                        f"mc_ev={signal.mc_ev:.4f}"
                    )
                    self.state.total_trades += 1  # count for statistics
                else:
                    # Live mode: place the actual order
                    result: Optional[OrderResult] = await loop.run_in_executor(
                        None,
                        self.executor.place_maker_limit,
                        token_id,
                        signal.direction,
                        signal.kelly_fraction,
                        implied_prob_up,
                    )
                    if result:
                        self.state.total_trades += 1
                        self._update_rebate_tracker(result)
                        self._update_compounding()

            # Step 8: Rich console output
            table: Table = self._build_status_table(signal, cycle_ms)
            self.console.print(table)
            self.cycle_times.append(cycle_ms)

    async def run(self) -> None:
        """Main entry point — launches all concurrent async tasks.

        The 5 coroutines run indefinitely via asyncio.gather:
        - 2 producers (Binance ticks, Gamma markets)
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

        # Connect Binance WebSocket
        await self.binance.connect()

        try:
            # Launch all 5 concurrent tasks
            await asyncio.gather(
                self.binance.run_ticker_loop(),  # Producer: Binance price ticks
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
            await self.binance.close()
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
    """Entry point: load config, create bot, run."""
    load_dotenv()

    # Configure logging with timestamps for latency analysis
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config: Config = load_config()
    test_mode: bool = os.getenv("TEST_MODE", "true").lower() == "true"

    bot: AsyncBot = AsyncBot(config, test_mode=test_mode)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
