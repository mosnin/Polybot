"""
data_feed.py — Async price streaming and Polymarket market discovery.

Two independent producers feed shared asyncio.Queue instances:
1. BybitWebSocket — sub-20ms BTC/USDT perpetual ticks via ccxt.pro
2. GammaMarketFinder — discovers active 5-minute BTC Up/Down token IDs

The winning edge starts here: low-latency Bybit price data lets us detect
momentum shifts before they're reflected in Polymarket CLOB prices. The
deterministic slug generation for Gamma API eliminates discovery lag.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
import ccxt.pro as ccxtpro


@dataclass
class PriceTick:
    """Single price observation from Bybit perpetual stream.

    Fields capture the full L1 snapshot needed for model updates:
    - last_price: most recent trade price (primary signal)
    - best_bid/ask: top-of-book for spread analysis
    - bid/ask_volume: order flow imbalance detection
    """

    timestamp: float  # unix ms from exchange
    last_price: float
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float


@dataclass
class MarketWindow:
    """Active 5-minute BTC Up/Down market on Polymarket.

    Each window has two tokens:
    - yes_token_id (UP): pays $1 if BTC finishes higher than open
    - no_token_id (DOWN): pays $1 if BTC finishes lower than open

    The start/end timestamps define the exact resolution window.
    """

    market_id: str
    slug: str
    question: str
    yes_token_id: str  # UP token
    no_token_id: str  # DOWN token
    start_timestamp: float  # when window opened (unix seconds)
    end_timestamp: float  # when window resolves (unix seconds)


class BybitWebSocket:
    """Async Bybit perpetual WebSocket feed via ccxt.pro.

    Subscribes to BTC/USDT:USDT perpetual swap for two reasons:
    1. Perpetual has tighter spreads and more volume than spot
    2. Perpetual price leads spot by ~100ms on average, giving us
       an information advantage over Polymarket participants using spot feeds

    The watch_ticker loop pushes PriceTick objects to a shared queue
    consumed by the main trading loop.
    """

    # USDT-margined perpetual swap — highest liquidity BTC instrument on Bybit
    SYMBOL: str = "BTC/USDT:USDT"
    MAX_RETRIES: int = 10
    BASE_DELAY: float = 1.0  # seconds for exponential backoff

    def __init__(self, tick_queue: asyncio.Queue) -> None:
        """Initialize with reference to shared tick queue.

        Args:
            tick_queue: asyncio.Queue that receives PriceTick objects.
                        Consumed by bot.py's trade loop.
        """
        self.tick_queue: asyncio.Queue = tick_queue
        self.exchange: Optional[ccxtpro.bybit] = None
        self._running: bool = False
        self.logger: logging.Logger = logging.getLogger("BybitWS")

    async def connect(self) -> None:
        """Initialize the ccxt.pro exchange instance.

        Uses defaultType=swap to route all calls to the futures API.
        enableRateLimit prevents hitting Bybit's WebSocket message limits.
        """
        self.exchange = ccxtpro.bybit(
            {
                "options": {"defaultType": "swap"},
                "enableRateLimit": True,
            }
        )
        self._running = True
        self.logger.info(f"Bybit WS initialized for {self.SYMBOL}")

    async def _watch_with_retry(self, watch_coro_factory, label: str):
        """Generic retry wrapper with exponential backoff for WS methods.

        ccxt.pro handles reconnection internally, but network partitions
        or exchange maintenance can cause prolonged failures. This wrapper
        ensures graceful degradation rather than crashing the bot.

        Args:
            watch_coro_factory: Callable returning an awaitable (e.g. watch_ticker)
            label: Human-readable name for log messages

        Returns:
            The data from the successful watch call, or None if max retries hit.
        """
        retries: int = 0
        while self._running:
            try:
                data = await watch_coro_factory()
                retries = 0  # reset on success — connection is healthy
                return data
            except Exception as e:
                retries += 1
                if retries > self.MAX_RETRIES:
                    self.logger.error(
                        f"{label}: max retries ({self.MAX_RETRIES}) exceeded, giving up"
                    )
                    raise
                # Exponential backoff: 2s, 4s, 8s... capped at 60s
                delay: float = min(self.BASE_DELAY * (2**retries), 60.0)
                self.logger.warning(
                    f"{label} error (retry {retries}/{self.MAX_RETRIES}) "
                    f"in {delay:.0f}s: {e}"
                )
                await asyncio.sleep(delay)
        return None

    async def run_ticker_loop(self) -> None:
        """Infinite loop consuming Bybit ticker WebSocket messages.

        Each iteration blocks on watch_ticker until a new message arrives
        (~50-200ms intervals). The PriceTick is immediately pushed to the
        shared queue for the trading loop to consume.

        This is the primary data source — every model update starts with
        a tick from this loop.
        """
        self.logger.info("Ticker loop started")
        while self._running:
            ticker = await self._watch_with_retry(
                lambda: self.exchange.watch_ticker(self.SYMBOL), "watch_ticker"
            )
            if ticker:
                tick = PriceTick(
                    timestamp=ticker["timestamp"],
                    last_price=ticker["last"],
                    best_bid=ticker["bid"],
                    best_ask=ticker["ask"],
                    bid_volume=ticker.get("bidVolume", 0.0),
                    ask_volume=ticker.get("askVolume", 0.0),
                )
                await self.tick_queue.put(tick)

    async def run_orderbook_loop(self) -> None:
        """Secondary feed for L2 orderbook depth data.

        Runs alongside the ticker loop to provide depth-weighted midpoint
        calculations. The orderbook data enriches volatility estimation
        by revealing large resting orders that may act as support/resistance.

        Currently consumed for monitoring; future enhancement could feed
        volume-weighted mid into the Bayesian model.
        """
        self.logger.info("Orderbook loop started")
        while self._running:
            await self._watch_with_retry(
                lambda: self.exchange.watch_order_book(self.SYMBOL),
                "watch_order_book",
            )

    async def close(self) -> None:
        """Gracefully shut down the WebSocket connection."""
        self._running = False
        if self.exchange:
            await self.exchange.close()
            self.logger.info("Bybit WS closed")


class GammaMarketFinder:
    """Discovers active 5-minute BTC Up/Down markets on Polymarket.

    The winning edge in market discovery: instead of polling all active markets
    and filtering (slow, subject to indexing lag), we deterministically compute
    the exact market slug from the current timestamp. Polymarket 5-min BTC
    markets follow the pattern: btc-updown-5m-{floor(unix_time / 300) * 300}

    This lets us query the exact market endpoint, reducing discovery latency
    from seconds to milliseconds.
    """

    GAMMA_BASE: str = "https://gamma-api.polymarket.com"
    POLL_INTERVAL: int = 60  # seconds between discovery checks
    WINDOW_SECONDS: int = 300  # 5 minutes

    def __init__(self, market_queue: asyncio.Queue) -> None:
        """Initialize with reference to shared market queue.

        Args:
            market_queue: asyncio.Queue that receives MarketWindow objects.
                          Consumed by bot.py's market discovery loop.
        """
        self.market_queue: asyncio.Queue = market_queue
        self.current_market: Optional[MarketWindow] = None
        self._running: bool = False
        self._token_cache: dict[str, MarketWindow] = {}  # slug -> MarketWindow
        self.logger: logging.Logger = logging.getLogger("GammaFinder")

    def _generate_slug(self) -> str:
        """Deterministically compute the current 5-min market slug.

        The slug encodes the window start time as a unix timestamp floored
        to the nearest 300-second boundary. This is how Polymarket indexes
        their rolling 5-minute BTC markets.

        Returns:
            Slug string like 'btc-updown-5m-1710500400'
        """
        now: float = time.time()
        # Floor to nearest 5-minute boundary
        rounded: int = int((now // self.WINDOW_SECONDS) * self.WINDOW_SECONDS)
        return f"btc-updown-5m-{rounded}"

    async def _fetch_market(
        self, session: aiohttp.ClientSession, slug: str
    ) -> Optional[MarketWindow]:
        """Query Gamma API for a specific market by slug.

        Uses the events/slug endpoint for direct lookup (O(1) vs O(n) filtering).
        Includes 3-retry exponential backoff for transient API failures.

        Args:
            session: Reusable aiohttp session for connection pooling
            slug: Market slug like 'btc-updown-5m-1710500400'

        Returns:
            MarketWindow if found and valid, None otherwise
        """
        # Check cache first — avoids redundant API calls for same window
        if slug in self._token_cache:
            return self._token_cache[slug]

        url: str = f"{self.GAMMA_BASE}/events/slug/{slug}"
        retries: int = 3

        for attempt in range(retries):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 404:
                        # Market not yet indexed — normal for very new windows
                        self.logger.debug(f"Market {slug} not found (404)")
                        return None
                    resp.raise_for_status()
                    data = await resp.json()

                    # Events endpoint returns object with 'markets' array
                    markets = data.get("markets", [])
                    if not markets:
                        self.logger.debug(f"No markets in event {slug}")
                        return None

                    market = markets[0]

                    # Parse token IDs — stored as JSON string or list
                    token_ids_raw = market.get("clobTokenIds", "[]")
                    if isinstance(token_ids_raw, str):
                        token_ids = json.loads(token_ids_raw)
                    else:
                        token_ids = token_ids_raw

                    if len(token_ids) < 2:
                        self.logger.warning(
                            f"Market {slug} has <2 tokens: {token_ids}"
                        )
                        return None

                    # Extract window timestamps from slug
                    rounded_ts: int = int(slug.split("-")[-1])

                    window = MarketWindow(
                        market_id=market.get("id", ""),
                        slug=slug,
                        question=market.get("question", ""),
                        yes_token_id=token_ids[0],  # First token = YES/UP
                        no_token_id=token_ids[1],  # Second token = NO/DOWN
                        start_timestamp=float(rounded_ts),
                        end_timestamp=float(rounded_ts + self.WINDOW_SECONDS),
                    )

                    # Cache to avoid re-fetching same window
                    self._token_cache[slug] = window
                    return window

            except (aiohttp.ClientError, json.JSONDecodeError, KeyError, ValueError) as e:
                delay: float = 2.0**attempt
                self.logger.warning(
                    f"Gamma fetch attempt {attempt + 1}/{retries} "
                    f"failed: {e}, retry in {delay:.0f}s"
                )
                await asyncio.sleep(delay)

        return None

    async def run(self) -> None:
        """Main discovery loop — polls for new market windows.

        On each iteration:
        1. Compute current window slug from timestamp
        2. Fetch market data (from cache or API)
        3. If new window detected, push to market_queue

        The 60-second poll interval is conservative. Near window boundaries,
        the bot's main loop handles the transition by checking remaining time.
        """
        self._running = True
        async with aiohttp.ClientSession() as session:
            while self._running:
                slug: str = self._generate_slug()
                market: Optional[MarketWindow] = await self._fetch_market(
                    session, slug
                )

                if market and (
                    self.current_market is None
                    or market.slug != self.current_market.slug
                ):
                    self.current_market = market
                    await self.market_queue.put(market)
                    self.logger.info(
                        f"New market window: {slug} "
                        f"UP={market.yes_token_id[:12]}... "
                        f"DOWN={market.no_token_id[:12]}..."
                    )

                await asyncio.sleep(self.POLL_INTERVAL)

        # Clean up cache to prevent memory leak across long sessions
        self._token_cache.clear()

    async def close(self) -> None:
        """Signal the discovery loop to stop."""
        self._running = False
        self.logger.info("GammaMarketFinder stopped")
