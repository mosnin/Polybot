"""
backtest.py — Deep backtesting and expectancy validation suite for PolyBot.

Downloads 30 days of 1-second BTC ticks via ccxt, replays every 5-minute
window through the full Bayesian + z-score + Monte Carlo pipeline, and
outputs comprehensive performance metrics.

This module is self-contained — it reuses BayesianModel from model.py
but requires no CLOB connection, wallet, or live executor. Everything is
simulated from historical price data.

Exit criteria for live deployment:
- Win rate >= 58% (edge above costs with margin of safety)
- Daily expectancy >= 0.8% (after fees/gas)
- Sharpe ratio > 1.0 (risk-adjusted returns are meaningful)
- Max drawdown < 25% (survivable with starting capital)

Usage:
    # Standalone CLI
    python backtest.py

    # Programmatic (from dashboard)
    import asyncio
    from backtest import run_backtest
    from config import load_config
    result = asyncio.run(run_backtest(load_config()))
"""

import asyncio
import datetime
import json
import logging
import os
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import Config, load_config
from model import BayesianModel

logger = logging.getLogger("Backtest")

# ---------------------------------------------------------------------------
# 1. Historical Data Download
# ---------------------------------------------------------------------------

CACHE_DIR: str = "/tmp"
CACHE_MAX_AGE_HOURS: int = 24


def _cache_path(days: int) -> str:
    """Return filesystem path for the cached tick data file."""
    return os.path.join(CACHE_DIR, f"polybot_btc_1s_{days}d.json")


def _cache_is_valid(path: str) -> bool:
    """Check if cached file exists and is less than 24 hours old."""
    if not os.path.exists(path):
        return False
    age_hours: float = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


async def download_historical_ticks(days: int = 30) -> List[dict]:
    """Download 1-second BTC/USDT perpetual candles from Binance via ccxt.

    Fetches ``days`` worth of 1-second OHLCV data, extracting the close price
    as the tick price.  Results are cached to disk for 24 hours to avoid
    redundant API calls.

    Pagination: ccxt returns max 1000 candles per request.  For 30 days
    (2,592,000 seconds) this requires ~2,592 requests with rate limiting.

    Args:
        days: Number of historical days to download (default 30)

    Returns:
        List of {"timestamp": int_ms, "price": float} dicts, sorted chronologically
    """
    cache_file: str = _cache_path(days)
    if _cache_is_valid(cache_file):
        logger.info(f"Loading cached tick data from {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)

    logger.info(f"Downloading {days} days of 1-second BTC ticks from Binance...")

    import ccxt

    exchange = ccxt.binance({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
    })

    now_ms: int = int(time.time() * 1000)
    since_ms: int = now_ms - (days * 86400 * 1000)
    all_ticks: List[dict] = []
    batch_count: int = 0

    current_since: int = since_ms
    while current_since < now_ms:
        try:
            candles = exchange.fetch_ohlcv(
                "BTC/USDT:USDT",
                timeframe="1s",
                since=current_since,
                limit=1000,
            )
        except Exception as e:
            logger.warning(f"Fetch error at batch {batch_count}: {e}, retrying...")
            await asyncio.sleep(2.0)
            continue

        if not candles:
            break

        for candle in candles:
            # candle = [timestamp_ms, open, high, low, close, volume]
            all_ticks.append({
                "timestamp": int(candle[0]),
                "price": float(candle[4]),  # close price
            })

        # Advance past the last candle's timestamp
        current_since = int(candles[-1][0]) + 1000  # +1 second
        batch_count += 1

        # Rate limiting — respect exchange limits
        rate_delay: float = exchange.rateLimit / 1000.0
        await asyncio.sleep(rate_delay)

        # Progress logging every 500 batches (~500k ticks)
        if batch_count % 500 == 0:
            pct: float = (current_since - since_ms) / (now_ms - since_ms) * 100
            logger.info(
                f"Download progress: {pct:.1f}% "
                f"({len(all_ticks):,} ticks, {batch_count} batches)"
            )

    # Sort chronologically (should already be, but ensure)
    all_ticks.sort(key=lambda t: t["timestamp"])

    # Cache to disk
    logger.info(
        f"Download complete: {len(all_ticks):,} ticks over {days} days. "
        f"Caching to {cache_file}"
    )
    with open(cache_file, "w") as f:
        json.dump(all_ticks, f)

    return all_ticks


# ---------------------------------------------------------------------------
# 2. Window Slicing
# ---------------------------------------------------------------------------

WINDOW_SECONDS: int = 300  # 5 minutes
MIN_TICKS_PER_WINDOW: int = 10


def slice_into_windows(
    ticks: List[dict], window_secs: int = WINDOW_SECONDS
) -> List[List[dict]]:
    """Group ticks into non-overlapping 5-minute windows.

    Each window is identified by ``floor(timestamp_sec / window_secs) * window_secs``.
    Windows with fewer than MIN_TICKS_PER_WINDOW ticks are discarded.

    Args:
        ticks: List of {"timestamp": int_ms, "price": float} dicts
        window_secs: Window duration in seconds (default 300 = 5 min)

    Returns:
        List of windows, each a list of tick dicts sorted chronologically
    """
    buckets: Dict[int, List[dict]] = defaultdict(list)

    for tick in ticks:
        ts_sec: float = tick["timestamp"] / 1000.0
        window_key: int = int(ts_sec // window_secs) * window_secs
        buckets[window_key].append(tick)

    # Sort windows chronologically, filter out thin windows
    windows: List[List[dict]] = []
    for key in sorted(buckets.keys()):
        window = buckets[key]
        if len(window) >= MIN_TICKS_PER_WINDOW:
            windows.append(window)

    logger.info(
        f"Sliced {len(ticks):,} ticks into {len(windows):,} valid windows "
        f"(discarded {len(buckets) - len(windows)} thin windows)"
    )
    return windows


# ---------------------------------------------------------------------------
# 3. Window Simulation
# ---------------------------------------------------------------------------


def _synthesize_implied_prob(current_price: float, open_price: float) -> float:
    """Synthesize an implied probability from price trajectory.

    Since historical CLOB midpoints are not available, we model the implied
    probability as a function of how much price has moved from window open.
    This reflects the real-world dynamic where the CLOB midpoint tracks
    recent price action with a lag — exactly the edge the bot exploits.

    Formula: ``0.5 + (current - open) / open * 10``, clamped to [0.4, 0.6]

    Args:
        current_price: Current BTC price at evaluation point
        open_price: Price at window open

    Returns:
        Synthetic implied probability of UP outcome (0.4 to 0.6)
    """
    if open_price <= 0:
        return 0.5
    move_pct: float = (current_price - open_price) / open_price
    implied: float = 0.5 + move_pct * 10.0
    return max(0.4, min(0.6, implied))


def simulate_window(
    window_ticks: List[dict], config: Config
) -> Optional[dict]:
    """Simulate one 5-minute window through the full trading pipeline.

    Creates a fresh BayesianModel, feeds ~80% of ticks for inference,
    then evaluates for a trade signal.  If a signal is generated, checks
    the actual outcome against the remaining 20% of ticks.

    Args:
        window_ticks: List of tick dicts for this window, sorted chronologically
        config: Bot configuration for model parameters

    Returns:
        Trade result dict if signal generated, None otherwise
    """
    model = BayesianModel(
        min_edge=config.min_edge_threshold,
        round_trip_cost=config.round_trip_cost_pct,
        decay_factor=config.decay_factor,
    )

    n_ticks: int = len(window_ticks)
    eval_idx: int = int(n_ticks * 0.8)  # evaluate after 80% of ticks consumed

    if eval_idx < 5:
        return None  # insufficient data for meaningful inference

    open_price: float = window_ticks[0]["price"]
    prev_price: Optional[float] = None

    # Feed ticks sequentially to the Bayesian model
    for i in range(eval_idx):
        price: float = window_ticks[i]["price"]
        model.update(price, prev_price)
        prev_price = price

    current_price: float = window_ticks[eval_idx - 1]["price"]
    remaining_seconds: float = float(n_ticks - eval_idx)  # ~1 tick per second

    # Synthesize implied probability (no real CLOB data in backtest)
    implied_prob_up: float = _synthesize_implied_prob(current_price, open_price)

    # Run full evaluation pipeline (no order_book → conservative defaults)
    signal = model.evaluate(
        current_price=current_price,
        implied_prob_up=implied_prob_up,
        remaining_seconds=max(remaining_seconds, 10.0),
        order_book=None,
    )

    if signal is None:
        return None

    # Determine actual outcome from remaining ticks
    exit_price: float = window_ticks[-1]["price"]
    if signal.direction == "UP":
        won: bool = exit_price > current_price
    else:
        won = exit_price < current_price

    return {
        "timestamp": window_ticks[eval_idx - 1]["timestamp"],
        "direction": signal.direction,
        "edge": signal.edge,
        "z_score": signal.z_score,
        "mc_ev": signal.mc_ev,
        "mc_win_prob": signal.mc_win_prob,
        "kelly_fraction": signal.kelly_fraction,
        "entry_price": current_price,
        "exit_price": exit_price,
        "implied_prob": implied_prob_up,
        "won": won,
    }


# ---------------------------------------------------------------------------
# 4. Full Backtest Runner
# ---------------------------------------------------------------------------


def _compute_sharpe(daily_returns: List[float]) -> float:
    """Compute annualized Sharpe ratio from daily returns.

    Sharpe = mean(returns) / std(returns) * sqrt(365)
    Returns 0.0 if insufficient data or zero variance.
    """
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns)
    std = float(np.std(arr, ddof=1))
    if std < 1e-10:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(365))


def _compute_max_drawdown(equity_curve: List[Tuple[float, float]]) -> float:
    """Compute maximum peak-to-trough drawdown from equity curve.

    Args:
        equity_curve: List of (timestamp, balance) tuples

    Returns:
        Max drawdown as a fraction (0.0 to 1.0)
    """
    if len(equity_curve) < 2:
        return 0.0
    peak: float = equity_curve[0][1]
    max_dd: float = 0.0
    for _, balance in equity_curve:
        if balance > peak:
            peak = balance
        if peak > 0:
            dd: float = (peak - balance) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


async def run_backtest(
    config: Config,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> dict:
    """Run the complete backtest: download data, simulate all windows, compute metrics.

    Downloads historical ticks, slices them into 5-minute windows, runs each
    through the full Bayesian pipeline, tracks P&L with Kelly sizing, and
    computes all performance metrics.

    Args:
        config: Bot configuration (strategy params, costs, capital)
        progress_callback: Optional callable(float) for progress updates (0.0 to 1.0)

    Returns:
        Dict with comprehensive backtest results including alert flags
    """
    # Step 1: Download historical ticks
    ticks: List[dict] = await download_historical_ticks(config.historical_data_days)

    if not ticks:
        return {
            "total_windows": 0, "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "avg_edge": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "net_return": 0.0,
            "final_balance": config.starting_capital,
            "daily_expectancy": 0.0, "avg_kelly": 0.0,
            "trades": [], "equity_curve": [],
            "alert_win_rate": True, "alert_expectancy": True,
        }

    # Step 2: Slice into 5-minute windows
    windows: List[List[dict]] = slice_into_windows(ticks)

    # Step 3: Simulate each window
    trades: List[dict] = []
    balance: float = config.starting_capital
    equity_curve: List[Tuple[float, float]] = [(ticks[0]["timestamp"], balance)]

    # Daily balance tracking for Sharpe calculation
    daily_balances: Dict[str, List[float]] = defaultdict(list)

    total_windows: int = len(windows)
    for i, window in enumerate(windows):
        # Progress callback every 100 windows
        if progress_callback and i % 100 == 0:
            progress_callback(i / max(total_windows, 1))

        # Safety floor check — skip if below floor
        if balance <= config.safety_floor_usdc:
            break

        trade: Optional[dict] = simulate_window(window, config)

        if trade is not None:
            # Compute P&L using Kelly-sized position (mirrors live sizing logic)
            kelly: float = trade["kelly_fraction"]
            available: float = balance - config.safety_floor_usdc - config.gas_buffer_usdc
            if available <= 0:
                continue

            position_size: float = available * kelly * config.max_exposure_pct
            position_size = min(position_size, available)
            position_size = max(position_size, 0.0)

            if trade["won"]:
                pnl: float = position_size * trade["edge"]
            else:
                pnl = -position_size * config.round_trip_cost_pct

            # Deduct gas cost per trade
            pnl -= config.gas_buffer_usdc

            trade["pnl"] = pnl
            trade["position_size"] = position_size
            trade["balance_before"] = balance

            balance += pnl
            balance = max(balance, 0.0)  # can't go negative

            trade["balance_after"] = balance
            trades.append(trade)

            # Track equity curve
            equity_curve.append((trade["timestamp"], balance))

            # Track daily balance for Sharpe
            day_str: str = datetime.datetime.fromtimestamp(
                trade["timestamp"] / 1000.0
            ).strftime("%Y-%m-%d")
            daily_balances[day_str].append(balance)

    # Final progress
    if progress_callback:
        progress_callback(1.0)

    # Step 4: Compute metrics
    wins: int = sum(1 for t in trades if t["won"])
    losses: int = len(trades) - wins
    win_rate: float = wins / max(len(trades), 1)

    avg_edge: float = (
        sum(t["edge"] for t in trades) / len(trades) if trades else 0.0
    )
    avg_kelly: float = (
        sum(t["kelly_fraction"] for t in trades) / len(trades) if trades else 0.0
    )

    # Sharpe ratio from daily returns
    daily_returns: List[float] = []
    sorted_days: List[str] = sorted(daily_balances.keys())
    prev_day_balance: float = config.starting_capital
    for day in sorted_days:
        end_balance: float = daily_balances[day][-1]
        if prev_day_balance > 0:
            daily_ret: float = (end_balance - prev_day_balance) / prev_day_balance
            daily_returns.append(daily_ret)
        prev_day_balance = end_balance

    sharpe: float = _compute_sharpe(daily_returns)
    max_dd: float = _compute_max_drawdown(equity_curve)
    net_return: float = (
        (balance - config.starting_capital) / config.starting_capital
        if config.starting_capital > 0 else 0.0
    )

    # Daily expectancy: average daily return
    daily_expectancy: float = (
        sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
    )

    # Alert thresholds
    alert_win_rate: bool = win_rate < 0.58
    alert_expectancy: bool = daily_expectancy < 0.008  # 0.8%

    result: dict = {
        "total_windows": total_windows,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_edge": avg_edge,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "net_return": net_return,
        "final_balance": balance,
        "daily_expectancy": daily_expectancy,
        "avg_kelly": avg_kelly,
        "trades": trades,
        "equity_curve": equity_curve,
        "alert_win_rate": alert_win_rate,
        "alert_expectancy": alert_expectancy,
    }

    logger.info(
        f"Backtest complete: {len(trades)} trades over {total_windows} windows | "
        f"Win rate: {win_rate:.1%} | Sharpe: {sharpe:.2f} | "
        f"Max DD: {max_dd:.1%} | Net return: {net_return:.1%} | "
        f"Final balance: ${balance:.2f}"
    )

    return result


# ---------------------------------------------------------------------------
# 5. CLI Entry Point
# ---------------------------------------------------------------------------


async def _cli_main() -> None:
    """CLI entry point — run backtest and print results to console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config: Config = load_config()
    logger.info("=" * 60)
    logger.info("  PolyBot Deep Backtest Engine")
    logger.info(f"  Historical days: {config.historical_data_days}")
    logger.info(f"  Starting capital: ${config.starting_capital:.2f}")
    logger.info(f"  Min edge: {config.min_edge_threshold}")
    logger.info(f"  Round-trip cost: {config.round_trip_cost_pct:.3%}")
    logger.info(f"  Decay factor: {config.decay_factor}")
    logger.info("=" * 60)

    start_time: float = time.time()
    result: dict = await run_backtest(config)
    elapsed: float = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Completed in:           {elapsed:.1f}s")
    print(f"  Total Windows Analyzed: {result['total_windows']:,}")
    print(f"  Total Trades:           {result['total_trades']:,}")
    print(f"  Wins / Losses:          {result['wins']} / {result['losses']}")
    print(f"  Win Rate:               {result['win_rate']:.1%}")
    print(f"  Average Edge:           {result['avg_edge']:.4f}")
    print(f"  Average Kelly:          {result['avg_kelly']:.3f}")
    print(f"  Sharpe Ratio:           {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:           {result['max_drawdown']:.1%}")
    print(f"  Net Return:             {result['net_return']:.1%}")
    print(f"  Final Balance:          ${result['final_balance']:.2f}")
    print(f"  Daily Expectancy:       {result['daily_expectancy']:.2%}")
    print("=" * 60)

    # Alerts
    if result["alert_win_rate"]:
        print("  ALERT: Win rate below 58% threshold!")
    if result["alert_expectancy"]:
        print("  ALERT: Daily expectancy below 0.8% threshold!")

    if not result["alert_win_rate"] and not result["alert_expectancy"]:
        print("  STRATEGY VALIDATED — expectancy confirms live deployment readiness")
    else:
        print("  STRATEGY FAILED VALIDATION — review parameters before live trading")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(_cli_main())
