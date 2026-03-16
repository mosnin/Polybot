"""
backtest.py — Historical backtesting for the Polymarket BTC 5-minute strategy.

Replays historical BTC price data through the exact same BayesianModel pipeline
used in live trading. Fetches 1-minute OHLCV candles from Binance via ccxt,
groups them into 5-minute windows, and simulates the full evaluation pipeline
(Bayesian update → edge gate → Monte Carlo → Kelly) on each window.

This validates that the model produces positive expectancy on real historical
data BEFORE risking capital in live trading. Run this first:

    python backtest.py
    python backtest.py --windows 2000   # more windows for higher confidence
    python backtest.py --days 7         # specify days of history

The backtest uses BayesianModel with redis_url=None (no Redis persistence)
and is completely standalone — it never imports or interacts with bot.py.

Exit criteria for going live:
- Win rate > 55% (not just 50% — need edge above costs)
- Average edge > 1% (covers worst-case taker fees)
- Sharpe ratio > 1.0 (risk-adjusted returns are meaningful)
- Max drawdown < 15% (survivable with starting capital)
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import ccxt
except ImportError:
    print("ERROR: ccxt not installed. Run: pip install ccxt")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None  # type: ignore[assignment, misc]

from model import BayesianModel, Signal


@dataclass
class WindowResult:
    """Result of simulating one 5-minute window."""

    traded: bool  # whether the model generated a signal
    direction: Optional[str]  # "UP" or "DOWN" if traded
    won: Optional[bool]  # whether the prediction was correct
    edge: float  # net edge at time of signal
    pnl: float  # realized P&L for this window
    signal: Optional[Signal]  # full signal object if traded


@dataclass
class BacktestResult:
    """Aggregate results across all simulated windows."""

    total_windows: int = 0
    traded_windows: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    edges: List[float] = field(default_factory=list)
    pnls: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.traded_windows == 0:
            return 0.0
        return self.wins / self.traded_windows

    @property
    def avg_edge(self) -> float:
        if not self.edges:
            return 0.0
        return sum(self.edges) / len(self.edges)

    @property
    def sharpe_ratio(self) -> float:
        if len(self.pnls) < 2:
            return 0.0
        arr = np.array(self.pnls)
        std = float(np.std(arr))
        if std < 1e-10:
            return 0.0
        # Annualize: 288 five-minute windows per day, 252 trading days
        return float(np.mean(arr)) / std * np.sqrt(252 * 288)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak: float = self.equity_curve[0]
        max_dd: float = 0.0
        for balance in self.equity_curve:
            peak = max(peak, balance)
            if peak > 0:
                dd: float = (peak - balance) / peak
                max_dd = max(max_dd, dd)
        return max_dd


def fetch_candles(exchange, days: int = 4) -> list:
    """Fetch historical 1-minute BTC/USDT candles from Binance.

    Args:
        exchange: ccxt exchange instance
        days: Number of days of history to fetch

    Returns:
        List of OHLCV candles [timestamp, open, high, low, close, volume]
    """
    since: int = int((time.time() - days * 86400) * 1000)
    all_candles: list = []
    limit: int = 1000  # Binance max per request

    print(f"  Fetching {days} days of 1m BTC/USDT candles...")

    while True:
        candles = exchange.fetch_ohlcv(
            "BTC/USDT", "1m", since=since, limit=limit
        )
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 60000  # next minute
        if len(candles) < limit:
            break

    print(f"  Fetched {len(all_candles)} candles")
    return all_candles


def candles_to_ticks(five_candles: list) -> List[float]:
    """Convert 5 one-minute candles into synthetic tick prices.

    For each candle, emit 4 ticks: open, high, low, close.
    This gives ~20 ticks per 5-minute window, which is enough for
    the BayesianModel to build a meaningful posterior.

    Args:
        five_candles: List of 5 OHLCV candles

    Returns:
        List of ~20 price floats
    """
    ticks: List[float] = []
    for candle in five_candles:
        # [timestamp, open, high, low, close, volume]
        ticks.extend([candle[1], candle[2], candle[3], candle[4]])
    return ticks


def simulate_window(
    ticks: List[float],
    min_edge: float = 0.02,
    round_trip_cost: float = 0.003,
) -> WindowResult:
    """Simulate one 5-minute window through the full model pipeline.

    Creates a fresh BayesianModel (uninformative prior), feeds it the
    tick sequence, and checks if a trading signal is generated. If so,
    determines the outcome by comparing start and end prices.

    Args:
        ticks: List of ~20 price observations for this window
        min_edge: Minimum net edge threshold
        round_trip_cost: Round-trip cost assumption

    Returns:
        WindowResult with trade outcome
    """
    if len(ticks) < 5:
        return WindowResult(
            traded=False, direction=None, won=None, edge=0.0, pnl=0.0, signal=None
        )

    model = BayesianModel(
        min_edge=min_edge,
        round_trip_cost=round_trip_cost,
        redis_url=None,  # no Redis for backtesting
    )

    # Feed ticks to the model
    prev_price: Optional[float] = None
    last_signal: Optional[Signal] = None

    for i, price in enumerate(ticks):
        model.update(price, prev_price)
        prev_price = price

        # Start evaluating after 10 ticks (enough for posterior to be informative)
        if i >= 10:
            remaining_secs: float = max(1.0, (len(ticks) - i) * 3.0)
            signal = model.evaluate(
                current_price=price,
                implied_prob_up=0.50,  # assume fair market for backtest
                remaining_seconds=remaining_secs,
            )
            if signal is not None:
                last_signal = signal
                break  # take the first valid signal

    if last_signal is None:
        return WindowResult(
            traded=False, direction=None, won=None, edge=0.0, pnl=0.0, signal=None
        )

    # Determine outcome: did price move in the predicted direction?
    start_price: float = ticks[0]
    end_price: float = ticks[-1]
    price_went_up: bool = end_price > start_price

    if last_signal.direction == "UP":
        won: bool = price_went_up
    else:
        won = not price_went_up

    # P&L calculation: binary outcome after costs
    pnl: float = last_signal.edge if won else -round_trip_cost

    return WindowResult(
        traded=True,
        direction=last_signal.direction,
        won=won,
        edge=last_signal.edge,
        pnl=pnl,
        signal=last_signal,
    )


def run_backtest(
    n_windows: int = 1000,
    days: int = 4,
    min_edge: float = 0.02,
    round_trip_cost: float = 0.003,
) -> BacktestResult:
    """Run full backtest across historical 5-minute windows.

    Args:
        n_windows: Target number of windows to simulate
        days: Days of historical data to fetch
        min_edge: Minimum net edge threshold
        round_trip_cost: Round-trip cost assumption

    Returns:
        BacktestResult with aggregate metrics
    """
    exchange = ccxt.binance({"enableRateLimit": True})

    # Fetch enough candles: n_windows * 5 candles each
    needed_days: int = max(days, (n_windows * 5) // (60 * 24) + 1)
    candles = fetch_candles(exchange, needed_days)

    if len(candles) < 10:
        print("ERROR: Not enough historical data. Try again later.")
        sys.exit(1)

    # Group into 5-candle windows
    n_available: int = len(candles) // 5
    n_actual: int = min(n_windows, n_available)

    print(f"  Simulating {n_actual} five-minute windows...")

    result = BacktestResult()
    starting_capital: float = 100.0
    balance: float = starting_capital
    result.equity_curve.append(balance)

    for i in range(n_actual):
        five = candles[i * 5 : (i + 1) * 5]
        ticks = candles_to_ticks(five)

        window_result = simulate_window(ticks, min_edge, round_trip_cost)
        result.total_windows += 1

        if window_result.traded:
            result.traded_windows += 1
            result.edges.append(window_result.edge)
            result.pnls.append(window_result.pnl)

            if window_result.won:
                result.wins += 1
                # Size at 5% of balance (conservative backtest sizing)
                trade_pnl: float = balance * 0.05 * window_result.edge
            else:
                result.losses += 1
                trade_pnl = -balance * 0.05 * round_trip_cost

            balance += trade_pnl
            result.total_pnl += trade_pnl

        result.equity_curve.append(balance)

    return result


def print_results(result: BacktestResult) -> None:
    """Print formatted backtest results."""

    if Console is not None:
        console = Console()
        table = Table(title="Backtest Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total Windows", str(result.total_windows))
        table.add_row("Traded Windows", f"{result.traded_windows} ({result.traded_windows/max(result.total_windows,1)*100:.1f}%)")
        table.add_row("Win Rate", f"{result.win_rate:.1%}")
        table.add_row("Wins / Losses", f"{result.wins} / {result.losses}")
        table.add_row("Avg Edge", f"{result.avg_edge:.4f} ({result.avg_edge*100:.2f}%)")
        table.add_row("Total P&L", f"${result.total_pnl:+.2f}")
        table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        table.add_row("Max Drawdown", f"{result.max_drawdown:.1%}")

        final_balance: float = result.equity_curve[-1] if result.equity_curve else 100.0
        table.add_row("Final Balance", f"${final_balance:.2f}")
        table.add_row("Return", f"{(final_balance / 100.0 - 1) * 100:+.2f}%")

        console.print()
        console.print(table)
        console.print()

        # Go/No-Go assessment
        go_live: bool = (
            result.win_rate > 0.55
            and result.avg_edge > 0.01
            and result.sharpe_ratio > 1.0
            and result.max_drawdown < 0.15
        )
        if go_live:
            console.print("[bold green]✓ EDGE CONFIRMED — ready for live trading[/bold green]")
            console.print("  Set TEST_MODE=false in .env to go live.")
        else:
            console.print("[bold red]✗ EDGE NOT CONFIRMED — do not go live[/bold red]")
            if result.win_rate <= 0.55:
                console.print(f"  Win rate {result.win_rate:.1%} below 55% threshold")
            if result.avg_edge <= 0.01:
                console.print(f"  Avg edge {result.avg_edge:.4f} below 1% threshold")
            if result.sharpe_ratio <= 1.0:
                console.print(f"  Sharpe {result.sharpe_ratio:.2f} below 1.0 threshold")
            if result.max_drawdown >= 0.15:
                console.print(f"  Max drawdown {result.max_drawdown:.1%} above 15% limit")
        console.print()
    else:
        # Fallback without Rich
        print(f"\n{'='*50}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"  Windows: {result.traded_windows}/{result.total_windows}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Avg Edge: {result.avg_edge:.4f}")
        print(f"  Total P&L: ${result.total_pnl:+.2f}")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  Max DD: {result.max_drawdown:.1%}")
        print(f"{'='*50}\n")


def main() -> None:
    """CLI entry point for backtesting."""
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Backtest Polymarket BTC 5-minute strategy on historical data"
    )
    parser.add_argument(
        "--windows", type=int, default=1000,
        help="Number of 5-minute windows to simulate (default: 1000)",
    )
    parser.add_argument(
        "--days", type=int, default=4,
        help="Days of historical data to fetch (default: 4)",
    )
    parser.add_argument(
        "--min-edge", type=float, default=0.02,
        help="Minimum net edge threshold (default: 0.02)",
    )
    parser.add_argument(
        "--cost", type=float, default=0.003,
        help="Round-trip cost assumption (default: 0.003 = 0.3%%)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Polybot — Historical Backtest")
    print("=" * 60)
    print()

    start_time: float = time.time()

    result: BacktestResult = run_backtest(
        n_windows=args.windows,
        days=args.days,
        min_edge=args.min_edge,
        round_trip_cost=args.cost,
    )

    elapsed: float = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    print_results(result)


if __name__ == "__main__":
    main()
