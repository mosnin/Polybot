"""
data_feed.py — Async price streaming and Polymarket market discovery.

Two independent producers feed shared asyncio.Queue instances:
1. OKXWebSocket  — real-time BTC/USDT-SWAP ticks via OKX WS v5 public feed
2. GammaMarketFinder — discovers active 5-minute BTC Up/Down token IDs

OKX public WebSocket requires no API key and has no geo-restrictions.
Endpoint: wss://ws.okx.com:8443/ws/v5/public
Channel:  tickers / BTC-USDT-SWAP
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp


@dataclass
class PriceTick:
    """Single price observation from the OKX WebSocket feed.

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


class OKXWebSocket:
    """Async OKX public WebSocket feed for BTC-USDT-SWAP tickers.

    Connects to wss://ws.okx.com:8443/ws/v5/public — no API key required,
    no geo-restrictions. Subscribes to the 'tickers' channel for
    BTC-USDT-SWAP perpetual swap.

    OKX WS v5 protocol:
    - Subscribe: send JSON {"op":"subscribe","args":[{"channel":"tickers","instId":"BTC-USDT-SWAP"}]}
    - Push: {"arg":{...},"data":[{"last":"..","bidPx":"..","askPx":"..","bidSz":"..","askSz":"..","ts":".."}]}
    - Heartbeat: send text "ping" every 25s, server responds "pong"
    """

    WS_URL: str = "wss://ws.okx.com:8443/ws/v5/public"
    INST_ID: str = "BTC-USDT-SWAP"
    PING_INTERVAL: float = 25.0   # seconds between heartbeat pings
    MAX_RETRIES: int = 10

    def __init__(self, tick_queue: asyncio.Queue) -> None:
        self.tick_queue: asyncio.Queue = tick_queue
        self._session: Optional[aiohttp.ClientSession] = None
        self._running: bool = False
        self.logger: logging.Logger = logging.getLogger("OKXWS")

    async def connect(self) -> None:
        """Open the aiohttp session (actual WS connect happens in run_ticker_loop)."""
        self._session = aiohttp.ClientSession()
        self._running = True
        self.logger.info(f"OKX WS ready — will connect to {self.WS_URL}")

    async def _ping_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send 'ping' every 25 seconds to keep the connection alive."""
        while self._running and not ws.closed:
            await asyncio.sleep(self.PING_INTERVAL)
            try:
                await ws.send_str("ping")
            except Exception:
                break

    async def run_ticker_loop(self) -> None:
        """Connect, subscribe, and stream ticks. Reconnects on any error."""
        retries: int = 0
        while self._running:
            try:
                async with self._session.ws_connect(
                    self.WS_URL,
                    heartbeat=None,          # we handle ping manually
                    receive_timeout=60.0,    # 60s read timeout
                ) as ws:
                    # Subscribe to BTC-USDT-SWAP tickers
                    await ws.send_str(json.dumps({
                        "op": "subscribe",
                        "args": [{"channel": "tickers", "instId": self.INST_ID}],
                    }))
                    self.logger.info(f"OKX WS connected — subscribed to {self.INST_ID} tickers")
                    retries = 0  # reset on successful connect

                    # Start heartbeat task
                    ping_task = asyncio.create_task(self._ping_loop(ws))

                    try:
                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                raw = msg.data
                                if raw == "pong":
                                    continue  # heartbeat response, ignore
                                try:
                                    payload = json.loads(raw)
                                except json.JSONDecodeError:
                                    continue

                                # Skip subscribe ack / event messages
                                if "data" not in payload:
                                    continue

                                for item in payload["data"]:
                                    try:
                                        tick = PriceTick(
                                            timestamp=float(item["ts"]),
                                            last_price=float(item["last"]),
                                            best_bid=float(item["bidPx"]),
                                            best_ask=float(item["askPx"]),
                                            bid_volume=float(item.get("bidSz", 0.0)),
                                            ask_volume=float(item.get("askSz", 0.0)),
                                        )
                                        await self.tick_queue.put(tick)
                                    except (KeyError, ValueError):
                                        continue

                            elif msg.type in (
                                aiohttp.WSMsgType.ERROR,
                                aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.CLOSED,
                            ):
                                self.logger.warning(f"OKX WS closed/error: {msg.type}")
                                break
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass

            except Exception as e:
                if not self._running:
                    break
                retries += 1
                delay: float = min(1.0 * (2 ** retries), 60.0)
                self.logger.warning(
                    f"OKX WS error (retry {retries}/{self.MAX_RETRIES}) "
                    f"reconnecting in {delay:.0f}s: {e}"
                )
                if retries > self.MAX_RETRIES:
                    self.logger.error("OKX WS: max retries exceeded")
                    raise
                await asyncio.sleep(delay)

    async def close(self) -> None:
        """Gracefully stop the WebSocket loop."""
        self._running = False
        if self._session:
            await self._session.close()
        self.logger.info("OKX WS closed")


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
