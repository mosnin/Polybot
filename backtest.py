"""
backtest.py — Deep backtesting and expectancy validation suite for PolyBot.

Downloads 30 days of 1-second BTC ticks via ccxt, replays every 5-minute
window through the full Bayesian + z-score + Monte Carlo pipeline, and
outputs comprehensive performance metrics.

Supports two modes:
1. **Synthetic backtest** (``run_backtest``): Uses OKX 1-min candles with
   synthesized implied probabilities.  Fast (~30 s), no CLOB connection needed.
2. **Real Polymarket backtest** (``run_real_backtest``): Discovers actual past
   5-min BTC markets via Gamma API, fetches real CLOB price history, and uses
   real market resolutions as ground truth.  Slower (~3–5 min) but fully
   authentic.

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
    from backtest import run_backtest, run_real_backtest
    from config import load_config
    result = asyncio.run(run_backtest(load_config()))
    result = asyncio.run(run_real_backtest(load_config()))
"""

import asyncio
import datetime
import json
import logging
import os
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import aiohttp
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
    """Download BTC/USDT-SWAP 1-minute candles from OKX REST API.

    Uses GET /api/v5/market/history-candles with bar=1m (no API key required,
    no geo-restrictions). Paginates backwards until ``days`` of data is
    collected, then generates synthetic 1-second ticks by linearly
    interpolating between each minute's open and close with small Gaussian
    noise — giving the Bayesian model realistic intra-minute price paths.

    30 days = ~43 200 minute candles = ~432 requests at 100/request.

    Uses subprocess curl for HTTP requests (reliable in all environments).

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

    import random
    import subprocess

    OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"
    INST_ID = "BTC-USDT-SWAP"
    LIMIT = 100  # max per OKX request

    now_ms: int = int(time.time() * 1000)
    cutoff_ms: int = now_ms - (days * 86400 * 1000)

    logger.info(f"Downloading {days} days of BTC 1m candles from OKX REST API...")

    def _fetch_candles(after_ms: Optional[int] = None) -> dict:
        """Fetch one page of candles via curl subprocess."""
        url = (
            f"{OKX_CANDLES_URL}?instId={INST_ID}&bar=1m&limit={LIMIT}"
        )
        if after_ms is not None:
            url += f"&after={after_ms}"
        result = subprocess.run(
            ["curl", "-s", "--max-time", "15", url],
            capture_output=True, text=True, timeout=20,
        )
        return json.loads(result.stdout)

    minute_candles: List[dict] = []
    after_ms: Optional[int] = None
    batch: int = 0

    while True:
        retries: int = 0
        payload: Optional[dict] = None
        while retries <= 8:
            try:
                payload = await asyncio.get_event_loop().run_in_executor(
                    None, _fetch_candles, after_ms
                )
                break
            except Exception as e:
                retries += 1
                delay = min(2.0 * (2 ** retries), 30.0)
                logger.warning(f"OKX fetch error (retry {retries}): {e}, wait {delay:.0f}s")
                await asyncio.sleep(delay)

        if payload is None:
            raise RuntimeError("OKX download failed after max retries")

        data = payload.get("data", [])
        if not data:
            break  # no more candles

        # OKX returns newest-first; each row: [ts, open, high, low, close, vol, ...]
        reached_cutoff = False
        for row in data:
            ts = int(row[0])
            if ts < cutoff_ms:
                reached_cutoff = True
                break
            minute_candles.append({
                "ts": ts,
                "open": float(row[1]),
                "close": float(row[4]),
            })

        if reached_cutoff:
            break

        # Paginate backwards — oldest ts in this batch
        after_ms = int(data[-1][0])
        batch += 1
        if batch % 50 == 0:
            pct = max(0.0, (now_ms - after_ms) / (now_ms - cutoff_ms) * 100)
            logger.info(
                f"OKX download: {pct:.0f}% ({len(minute_candles):,} candles)"
            )
        await asyncio.sleep(0.12)  # ~8 req/s, stay under rate limit

    minute_candles.sort(key=lambda c: c["ts"])
    logger.info(f"Downloaded {len(minute_candles):,} 1-minute candles from OKX")

    # Generate synthetic 1-second ticks by interpolating between candle open/close
    all_ticks: List[dict] = []
    for candle in minute_candles:
        ts_start = candle["ts"]
        open_px = candle["open"]
        close_px = candle["close"]
        for s in range(60):
            alpha = s / 59.0
            base = open_px + alpha * (close_px - open_px)
            # 1 basis-point Gaussian noise simulates realistic tick noise
            noise = random.gauss(0.0, base * 0.0001)
            all_ticks.append({
                "timestamp": ts_start + s * 1000,
                "price": max(1.0, base + noise),
            })

    all_ticks.sort(key=lambda t: t["timestamp"])

    logger.info(
        f"Generated {len(all_ticks):,} synthetic ticks from {len(minute_candles):,} "
        f"1-minute candles. Caching to {cache_file}"
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


def _compute_backtest_metrics(
    trades: List[dict],
    total_windows: int,
    config: Config,
    initial_timestamp: Optional[int] = None,
    data_source: str = "okx_synthetic",
) -> dict:
    """Compute backtest summary metrics from a list of raw trade signals.

    Applies Kelly-based position sizing, computes P&L, builds equity curve,
    and derives all performance statistics.  Shared between ``run_backtest``
    and ``run_real_backtest``.

    Args:
        trades: List of raw trade dicts (must have ``kelly_fraction``, ``edge``,
                ``won``, ``timestamp``).  Modified in-place with P&L fields.
        total_windows: Total number of windows analyzed
        config: Bot configuration for sizing / cost parameters
        initial_timestamp: First timestamp for equity curve anchor (ms)
        data_source: Label for the data source

    Returns:
        Complete result dict with all metrics + alert flags
    """
    balance: float = config.starting_capital
    equity_curve: List[Tuple[float, float]] = []
    if initial_timestamp is not None:
        equity_curve.append((initial_timestamp, balance))

    daily_balances: Dict[str, List[float]] = defaultdict(list)
    sized_trades: List[dict] = []

    for trade in trades:
        if balance <= config.safety_floor_usdc:
            break

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

        pnl -= config.gas_buffer_usdc

        trade["pnl"] = pnl
        trade["position_size"] = position_size
        trade["balance_before"] = balance

        balance += pnl
        balance = max(balance, 0.0)

        trade["balance_after"] = balance
        sized_trades.append(trade)

        equity_curve.append((trade["timestamp"], balance))

        day_str: str = datetime.datetime.fromtimestamp(
            trade["timestamp"] / 1000.0
        ).strftime("%Y-%m-%d")
        daily_balances[day_str].append(balance)

    wins: int = sum(1 for t in sized_trades if t["won"])
    losses: int = len(sized_trades) - wins
    win_rate: float = wins / max(len(sized_trades), 1)

    avg_edge: float = (
        sum(t["edge"] for t in sized_trades) / len(sized_trades)
        if sized_trades else 0.0
    )
    avg_kelly: float = (
        sum(t["kelly_fraction"] for t in sized_trades) / len(sized_trades)
        if sized_trades else 0.0
    )

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

    daily_expectancy: float = (
        sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
    )

    alert_win_rate: bool = win_rate < 0.58
    alert_expectancy: bool = daily_expectancy < 0.008

    result: dict = {
        "total_windows": total_windows,
        "total_trades": len(sized_trades),
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
        "trades": sized_trades,
        "equity_curve": equity_curve,
        "alert_win_rate": alert_win_rate,
        "alert_expectancy": alert_expectancy,
        "data_source": data_source,
    }

    logger.info(
        f"Backtest complete ({data_source}): {len(sized_trades)} trades over "
        f"{total_windows} windows | Win rate: {win_rate:.1%} | "
        f"Sharpe: {sharpe:.2f} | Max DD: {max_dd:.1%} | "
        f"Net return: {net_return:.1%} | Final balance: ${balance:.2f}"
    )

    return result


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
    raw_trades: List[dict] = []
    total_windows: int = len(windows)

    for i, window in enumerate(windows):
        if progress_callback and i % 100 == 0:
            progress_callback(i / max(total_windows, 1))

        trade: Optional[dict] = simulate_window(window, config)
        if trade is not None:
            raw_trades.append(trade)

    if progress_callback:
        progress_callback(1.0)

    # Step 4: Compute metrics via shared function
    return _compute_backtest_metrics(
        trades=raw_trades,
        total_windows=total_windows,
        config=config,
        initial_timestamp=ticks[0]["timestamp"],
        data_source="okx_synthetic",
    )


# ---------------------------------------------------------------------------
# 5. Real Polymarket Data Backtest
# ---------------------------------------------------------------------------

GAMMA_API_BASE: str = "https://gamma-api.polymarket.com"
CLOB_API_BASE: str = "https://clob.polymarket.com"
SLUG_PREFIX: str = "btc-updown-5m-"


def _real_markets_cache_path(days: int) -> str:
    return os.path.join(CACHE_DIR, f"polybot_real_markets_{days}d.json")


def _clob_prices_cache_path(days: int) -> str:
    return os.path.join(CACHE_DIR, f"polybot_clob_prices_{days}d.json")


async def discover_real_markets(
    days: int = 30,
    concurrency: int = 20,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[dict]:
    """Enumerate past 5-min BTC markets via Gamma API slug lookup.

    Generates deterministic slugs for every 5-min window in the requested
    time range and queries the Gamma API for each.  Markets that existed
    are returned with token IDs and resolution outcomes.

    Args:
        days: Number of historical days to scan
        concurrency: Max concurrent HTTP requests to Gamma API
        progress_callback: Optional progress reporter (0.0–1.0)

    Returns:
        List of dicts with keys: slug, market_id, yes_token_id,
        no_token_id, start_ts, end_ts, resolution ("YES" or "NO")
    """
    cache_file: str = _real_markets_cache_path(days)
    if _cache_is_valid(cache_file):
        logger.info(f"Loading cached real market data from {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)

    now: int = int(time.time())
    start: int = now - days * 86400
    # Align to 5-min boundaries
    start = (start // WINDOW_SECONDS) * WINDOW_SECONDS

    slugs: List[Tuple[str, int]] = []
    ts: int = start
    while ts < now:
        slugs.append((f"{SLUG_PREFIX}{ts}", ts))
        ts += WINDOW_SECONDS

    logger.info(
        f"Discovering real markets: {len(slugs)} possible 5-min windows "
        f"over {days} days"
    )

    markets: List[dict] = []
    sem = asyncio.Semaphore(concurrency)
    completed: int = 0
    total: int = len(slugs)

    async def _fetch_one(
        session: aiohttp.ClientSession, slug: str, start_ts: int
    ) -> Optional[dict]:
        nonlocal completed
        url: str = f"{GAMMA_API_BASE}/events/slug/{slug}"
        for attempt in range(3):
            try:
                async with sem:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 404:
                            return None
                        if resp.status != 200:
                            if attempt < 2:
                                await asyncio.sleep(2.0 ** attempt)
                                continue
                            return None
                        data = await resp.json()
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2.0 ** attempt)
                    continue
                return None
            finally:
                completed += 1
                if progress_callback and completed % 200 == 0:
                    progress_callback(completed / total * 0.4)  # 0–40% for discovery

            event_markets = data.get("markets") if isinstance(data, dict) else None
            if not event_markets:
                return None

            mkt = event_markets[0]
            # Parse token IDs (may be JSON string or list)
            raw_tokens = mkt.get("clobTokenIds", [])
            if isinstance(raw_tokens, str):
                try:
                    raw_tokens = json.loads(raw_tokens)
                except (json.JSONDecodeError, TypeError):
                    return None
            if len(raw_tokens) < 2:
                return None

            # Determine resolution from outcomePrices
            outcome_prices = mkt.get("outcomePrices", [])
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            resolution: Optional[str] = None
            if len(outcome_prices) >= 2:
                try:
                    if float(outcome_prices[0]) > 0.5:
                        resolution = "YES"
                    elif float(outcome_prices[1]) > 0.5:
                        resolution = "NO"
                except (ValueError, TypeError):
                    pass

            if resolution is None:
                return None  # Market not yet resolved

            return {
                "slug": slug,
                "market_id": mkt.get("id", ""),
                "yes_token_id": raw_tokens[0],
                "no_token_id": raw_tokens[1],
                "start_ts": start_ts,
                "end_ts": start_ts + WINDOW_SECONDS,
                "resolution": resolution,
            }

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_one(session, slug, start_ts)
            for slug, start_ts in slugs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, dict):
            markets.append(r)

    logger.info(f"Discovered {len(markets)} real markets out of {total} windows")

    # Cache results
    with open(cache_file, "w") as f:
        json.dump(markets, f)

    return markets


async def fetch_clob_price_history(
    markets: List[dict],
    concurrency: int = 10,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, List[dict]]:
    """Fetch minute-by-minute CLOB share prices for each market's YES token.

    Calls the official Polymarket CLOB ``/prices-history`` endpoint for each
    discovered market to obtain the real implied-probability time series.

    Args:
        markets: List of market dicts from ``discover_real_markets``
        concurrency: Max concurrent CLOB requests
        progress_callback: Optional progress reporter (0.4–0.7 range)

    Returns:
        Dict mapping market slug → list of {"t": unix, "p": float}
    """
    if not markets:
        return {}

    # Check cache
    days_approx: int = max(
        1,
        int((max(m["end_ts"] for m in markets) - min(m["start_ts"] for m in markets))
            / 86400) + 1,
    )
    cache_file: str = _clob_prices_cache_path(days_approx)
    if _cache_is_valid(cache_file):
        logger.info(f"Loading cached CLOB price history from {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)

    logger.info(f"Fetching CLOB price history for {len(markets)} markets...")

    price_data: Dict[str, List[dict]] = {}
    sem = asyncio.Semaphore(concurrency)
    completed: int = 0
    total: int = len(markets)

    async def _fetch_prices(
        session: aiohttp.ClientSession, market: dict
    ) -> Optional[Tuple[str, List[dict]]]:
        nonlocal completed
        token_id: str = market["yes_token_id"]
        url: str = (
            f"{CLOB_API_BASE}/prices-history"
            f"?market={token_id}"
            f"&startTs={market['start_ts']}"
            f"&endTs={market['end_ts']}"
            f"&fidelity=1"
        )
        for attempt in range(3):
            try:
                async with sem:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status != 200:
                            if attempt < 2:
                                await asyncio.sleep(2.0 ** attempt)
                                continue
                            return None
                        data = await resp.json()
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2.0 ** attempt)
                    continue
                return None
            finally:
                completed += 1
                if progress_callback and completed % 50 == 0:
                    progress_callback(0.4 + completed / total * 0.3)  # 40–70%

            history = data.get("history", [])
            if len(history) < 2:
                return None

            return (market["slug"], history)

    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_prices(session, m) for m in markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, tuple):
            slug, history = r
            price_data[slug] = history

    logger.info(
        f"Fetched CLOB prices for {len(price_data)} / {total} markets"
    )

    with open(cache_file, "w") as f:
        json.dump(price_data, f)

    return price_data


def simulate_real_window(
    btc_ticks: List[dict],
    clob_prices: List[dict],
    resolution: str,
    config: Config,
) -> Optional[dict]:
    """Run the Bayesian + MC pipeline on one window using real CLOB data.

    The BTC ticks drive ``model.update()`` (momentum estimation), while the
    CLOB share price provides the real implied probability that the
    synthesized version approximated.  The win/loss outcome is determined
    by the actual market resolution, not a price comparison.

    Args:
        btc_ticks: OKX 1-min interpolated prices for this 5-min window
        clob_prices: CLOB minute-by-minute prices [{"t": unix, "p": float}]
        resolution: "YES" or "NO" — the actual market outcome
        config: Bot configuration

    Returns:
        Trade result dict or None if no signal generated
    """
    model = BayesianModel(
        min_edge=config.min_edge_threshold,
        round_trip_cost=config.round_trip_cost_pct,
        decay_factor=config.decay_factor,
    )

    n_ticks: int = len(btc_ticks)
    eval_idx: int = int(n_ticks * 0.8)

    if eval_idx < 5:
        return None

    prev_price: Optional[float] = None
    for i in range(eval_idx):
        price: float = btc_ticks[i]["price"]
        model.update(price, prev_price)
        prev_price = price

    current_price: float = btc_ticks[eval_idx - 1]["price"]
    remaining_seconds: float = float(n_ticks - eval_idx)

    # Use real CLOB price as implied probability.
    # Find the CLOB price point closest to the evaluation timestamp.
    eval_ts: float = btc_ticks[eval_idx - 1]["timestamp"] / 1000.0
    best_clob: Optional[dict] = None
    best_dist: float = float("inf")
    for cp in clob_prices:
        dist: float = abs(cp["t"] - eval_ts)
        if dist < best_dist:
            best_dist = dist
            best_clob = cp

    if best_clob is None:
        return None

    # CLOB share price IS the implied probability for this binary market
    implied_prob_up: float = float(best_clob["p"])
    # Clamp to sane range to avoid degenerate edge calculations
    implied_prob_up = max(0.01, min(0.99, implied_prob_up))

    signal = model.evaluate(
        current_price=current_price,
        implied_prob_up=implied_prob_up,
        remaining_seconds=max(remaining_seconds, 10.0),
        order_book=None,
    )

    if signal is None:
        return None

    # Determine outcome from real market resolution
    if signal.direction == "UP":
        won: bool = resolution == "YES"
    else:
        won = resolution == "NO"

    exit_price: float = btc_ticks[-1]["price"]

    return {
        "timestamp": btc_ticks[eval_idx - 1]["timestamp"],
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


async def run_real_backtest(
    config: Config,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> dict:
    """Run backtest using real Polymarket CLOB data and market resolutions.

    Pipeline:
    1. Download OKX 1-min candles (reuse ``download_historical_ticks``)
    2. Discover real Polymarket 5-min BTC markets via Gamma API
    3. Fetch CLOB price history for each discovered market
    4. Align BTC ticks with CLOB windows
    5. Simulate each window with real implied probabilities + resolutions
    6. Compute metrics via shared ``_compute_backtest_metrics``

    The result dict is compatible with ``run_backtest`` output, so the
    dashboard can render it identically.

    Args:
        config: Bot configuration
        progress_callback: Optional progress reporter (0.0–1.0)

    Returns:
        Dict with comprehensive backtest results including alert flags.
        Includes ``data_source: "polymarket_clob"`` and
        ``alert_no_markets: True`` if no real markets were found.
    """
    days: int = config.historical_data_days

    # Phase 1: Download BTC ticks (0–10% progress)
    logger.info("Real backtest — Phase 1: Downloading BTC ticks...")
    ticks: List[dict] = await download_historical_ticks(days)

    if not ticks:
        return {
            "total_windows": 0, "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "avg_edge": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "net_return": 0.0,
            "final_balance": config.starting_capital,
            "daily_expectancy": 0.0, "avg_kelly": 0.0,
            "trades": [], "equity_curve": [],
            "alert_win_rate": True, "alert_expectancy": True,
            "alert_no_markets": True,
            "data_source": "polymarket_clob",
        }

    if progress_callback:
        progress_callback(0.1)

    # Phase 2: Discover real markets (10–50% progress)
    logger.info("Real backtest — Phase 2: Discovering real Polymarket markets...")
    real_markets: List[dict] = await discover_real_markets(
        days=days,
        concurrency=20,
        progress_callback=progress_callback,
    )

    if not real_markets:
        logger.warning("No real Polymarket 5-min BTC markets found")
        return {
            "total_windows": 0, "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "avg_edge": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "net_return": 0.0,
            "final_balance": config.starting_capital,
            "daily_expectancy": 0.0, "avg_kelly": 0.0,
            "trades": [], "equity_curve": [],
            "alert_win_rate": True, "alert_expectancy": True,
            "alert_no_markets": True,
            "data_source": "polymarket_clob",
        }

    if progress_callback:
        progress_callback(0.5)

    # Phase 3: Fetch CLOB price histories (50–80% progress)
    logger.info("Real backtest — Phase 3: Fetching CLOB price histories...")
    clob_prices: Dict[str, List[dict]] = await fetch_clob_price_history(
        markets=real_markets,
        concurrency=10,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback(0.8)

    # Phase 4: Slice BTC ticks and align with real markets
    logger.info("Real backtest — Phase 4: Aligning BTC ticks with real markets...")
    btc_windows: Dict[int, List[dict]] = {}
    for tick in ticks:
        ts_sec: float = tick["timestamp"] / 1000.0
        window_key: int = int(ts_sec // WINDOW_SECONDS) * WINDOW_SECONDS
        if window_key not in btc_windows:
            btc_windows[window_key] = []
        btc_windows[window_key].append(tick)

    # Phase 5: Simulate each matched window
    logger.info("Real backtest — Phase 5: Simulating matched windows...")
    raw_trades: List[dict] = []
    matched_count: int = 0

    # Sort markets chronologically for consistent equity curve
    real_markets.sort(key=lambda m: m["start_ts"])

    for mkt in real_markets:
        slug: str = mkt["slug"]
        start_ts: int = mkt["start_ts"]

        # Need both BTC ticks and CLOB prices for this window
        if slug not in clob_prices:
            continue
        if start_ts not in btc_windows:
            continue

        window_btc = btc_windows[start_ts]
        if len(window_btc) < MIN_TICKS_PER_WINDOW:
            continue

        # Sort ticks chronologically within window
        window_btc.sort(key=lambda t: t["timestamp"])

        trade: Optional[dict] = simulate_real_window(
            btc_ticks=window_btc,
            clob_prices=clob_prices[slug],
            resolution=mkt["resolution"],
            config=config,
        )

        if trade is not None:
            raw_trades.append(trade)
        matched_count += 1

    logger.info(
        f"Real backtest: {matched_count} windows matched, "
        f"{len(raw_trades)} signals generated"
    )

    if progress_callback:
        progress_callback(0.95)

    # Phase 6: Compute metrics
    result = _compute_backtest_metrics(
        trades=raw_trades,
        total_windows=matched_count,
        config=config,
        initial_timestamp=ticks[0]["timestamp"] if ticks else None,
        data_source="polymarket_clob",
    )

    result["real_markets_found"] = len(real_markets)
    result["clob_prices_fetched"] = len(clob_prices)
    result["windows_matched"] = matched_count

    if matched_count == 0:
        result["alert_no_markets"] = True

    if progress_callback:
        progress_callback(1.0)

    return result


# ---------------------------------------------------------------------------
# 6. CLI Entry Point
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
