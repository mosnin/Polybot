"""
executor.py — Order execution layer wrapping py-clob-client.

Handles all interaction with the Polymarket CLOB:
- Placing maker limit orders sized by Kelly criterion
- Balance tracking via web3
- Stale order cancellation for capital efficiency

The winning edge in execution: pricing orders at midpoint - 0.01 targets
maker classification. Polymarket rewards passive liquidity providers with
rebates, effectively reducing our round-trip cost below the 0.3% worst case.
Over hundreds of trades, this hidden edge compounds significantly.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from web3 import Web3

# Side constants from py-clob-client
BUY: str = "BUY"
SELL: str = "SELL"


@dataclass
class OrderResult:
    """Record of a placed order for tracking and lifecycle management.

    Fields:
        order_id: CLOB-assigned order identifier for cancellation
        side: BUY or SELL
        price: Limit price (0.00 to 1.00)
        size: Number of conditional tokens
        token_id: Which market token this order is for
        timestamp: Unix time when order was posted
        status: Current lifecycle state
    """

    order_id: str
    side: str
    price: float
    size: float
    token_id: str
    timestamp: float
    status: str  # "posted", "filled", "cancelled", "error"


class OrderExecutor:
    """Manages order placement and lifecycle on Polymarket CLOB.

    Design principles:
    1. Maker-only: price inside the book for passive fills + rebates
    2. Kelly-sized: position sizes are mathematically optimal
    3. Capital-safe: never risks below safety floor
    4. Self-cleaning: stale orders cancelled every 10 seconds
    """

    def __init__(
        self,
        private_key: str,
        alchemy_rpc_url: str,
        clob_host: str,
        chain_id: int,
        signature_type: int,
        gas_buffer: float,
        safety_floor: float,
    ) -> None:
        """Initialize CLOB client and web3 provider.

        The ClobClient handles EIP-712 order signing internally.
        set_api_creds() derives L2 API credentials from the private key
        for authenticated endpoints (order placement, cancellation).

        Args:
            private_key: Polygon wallet private key (0x-prefixed)
            alchemy_rpc_url: Alchemy RPC for balance queries
            clob_host: Polymarket CLOB API host
            chain_id: 137 for Polygon mainnet
            signature_type: 0 for EOA wallets
            gas_buffer: USDC reserved for gas per trade
            safety_floor: Minimum USDC balance to maintain
        """
        self.client: ClobClient = ClobClient(
            host=clob_host,
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
        )
        # Derive L2 API credentials for authenticated trading endpoints.
        # This uses EIP-712 signing to prove key ownership without exposing it.
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

        # Web3 for on-chain balance queries (faster than CLOB API for reads)
        self.w3: Web3 = Web3(Web3.HTTPProvider(alchemy_rpc_url))

        self.gas_buffer: float = gas_buffer
        self.safety_floor: float = safety_floor
        self.active_orders: List[OrderResult] = []
        self.logger: logging.Logger = logging.getLogger("Executor")

    def _call_with_retry(
        self, func, *args, max_retries: int = 3, base_delay: float = 0.5
    ):
        """Execute a sync function with exponential backoff on failure.

        Used for read-only CLOB API calls (balance, midpoint, order book).
        NOT used for order placement — a failed order should not retry
        to avoid double-ordering.

        Args:
            func: Callable to execute
            *args: Arguments to pass to func
            max_retries: Maximum retry attempts
            base_delay: Initial delay in seconds (doubles each retry)

        Returns:
            Return value of func(*args)

        Raises:
            Last exception if all retries exhausted
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Retry {attempt+1}/{max_retries} for {func.__name__}: {e} "
                        f"(next in {delay:.1f}s)"
                    )
                    time.sleep(delay)
        raise last_error

    def get_current_balance(self) -> float:
        """Fetch USDC balance from the CLOB client with retry.

        client.get_balance() returns balance in wei (USDC has 6 decimals
        on Polygon), so we divide by 1e6 to get human-readable USDC.

        Returns:
            Current USDC balance as float
        """
        balance_wei: float = float(self._call_with_retry(self.client.get_balance))
        return balance_wei / 1e6

    def get_midpoint(self, token_id: str) -> float:
        """Get CLOB midpoint price for a conditional token with retry.

        The midpoint = (best_bid + best_ask) / 2. For binary markets,
        this represents the market's implied probability of that outcome.

        Args:
            token_id: Polymarket conditional token identifier

        Returns:
            Midpoint price as float (0.00 to 1.00)
        """
        return float(self._call_with_retry(self.client.get_midpoint, token_id))

    def get_order_book(self, token_id: str) -> dict:
        """Get full L2 order book for a conditional token with retry.

        Returns dict with 'bids' and 'asks' arrays, each containing
        [price, size] entries sorted by price.

        Args:
            token_id: Polymarket conditional token identifier

        Returns:
            Dict with 'bids' and 'asks' arrays
        """
        return self._call_with_retry(self.client.get_order_book, token_id)

    def _compute_order_size(
        self,
        balance: float,
        kelly_fraction: float,
        price: float,
        max_exposure_pct: float = 0.15,
    ) -> Optional[float]:
        """Compute order size in conditional tokens — full Kelly flow-through.

        Explosive mode: no 5% floor clamp. Full Kelly fraction flows through
        to maximize compounding speed. Only capped at max_exposure_pct (15%).

        Args:
            balance: Current USDC balance (live from get_current_balance)
            kelly_fraction: Optimal bet fraction from model (0 to 1.0)
            price: Order price per token
            max_exposure_pct: Maximum exposure as fraction of balance (default 0.15)

        Returns:
            Order size in tokens, or None if insufficient balance
        """
        available: float = balance - self.safety_floor - self.gas_buffer
        if available <= 0:
            self.logger.warning(
                f"Below safety floor: balance={balance:.2f}, "
                f"floor={self.safety_floor:.2f}"
            )
            return None

        # Full Kelly-sized dollar risk — no down-scaling
        dollar_risk: float = available * kelly_fraction

        # Cap at max_exposure_pct of total balance
        max_risk: float = balance * max_exposure_pct
        dollar_risk = min(dollar_risk, max_risk)

        # Never exceed available capital
        dollar_risk = min(dollar_risk, available)

        if dollar_risk < 0.01:
            return None  # sub-cent position not worth the gas

        # Convert to token quantity
        size: float = dollar_risk / price
        return round(size, 2)  # CLOB accepts 2 decimal places for size

    def place_maker_limit(
        self,
        token_id: str,
        direction: str,
        kelly_fraction: float,
        midpoint: float,
        max_exposure_pct: float = 0.10,
        balance_cap: Optional[float] = None,
        reservation_price: Optional[float] = None,
    ) -> Optional[OrderResult]:
        """Place a maker limit order using Stoikov reservation price or naive offset.

        When reservation_price is provided (from Stoikov engine), uses it directly
        as the limit price for optimal inventory-aware quoting. Falls back to
        midpoint - 0.01 when Stoikov is unavailable.

        Args:
            token_id: Which conditional token to buy
            direction: "UP" or "DOWN" (for logging)
            kelly_fraction: Position size fraction from model
            midpoint: Current CLOB midpoint price
            max_exposure_pct: Maximum exposure cap, adjustable via dashboard
            balance_cap: If set, cap effective balance for sizing (pre-compounding)
            reservation_price: Optional Stoikov reservation price for the order

        Returns:
            OrderResult if successfully posted, None on failure
        """
        balance: float = self.get_current_balance()
        if balance_cap is not None:
            balance = min(balance, balance_cap)

        # Stoikov reservation price takes priority over naive offset
        if reservation_price is not None:
            price: float = round(reservation_price, 2)
        else:
            # Fallback: one tick below midpoint (maker side)
            price = round(midpoint - 0.01, 2)

        # Sanity check: price must be in valid range for binary tokens
        if price <= 0.0 or price >= 1.0:
            price = round(midpoint, 2)
            price = max(0.01, min(price, 0.99))

        size: Optional[float] = self._compute_order_size(
            balance, kelly_fraction, price, max_exposure_pct
        )
        if size is None:
            self.logger.warning(
                f"Insufficient balance for {direction} order: "
                f"balance={balance:.2f}"
            )
            return None

        try:
            # Build the order using py-clob-client's OrderArgs
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY,  # always buying the directional token
            )

            # Sign the order (EIP-712 signature)
            signed_order = self.client.create_order(order_args)

            # Post as GTC (Good-Til-Cancelled) — stays on book until
            # filled or we cancel it via cancel_stale_orders
            response = self.client.post_order(signed_order, OrderType.GTC)

            result = OrderResult(
                order_id=response.get("orderID", response.get("id", "unknown")),
                side=BUY,
                price=price,
                size=size,
                token_id=token_id,
                timestamp=time.time(),
                status="posted",
            )
            self.active_orders.append(result)
            self.logger.info(
                f"Order posted: {direction} {size:.2f} tokens @ {price:.2f} "
                f"(id={result.order_id})"
            )
            return result

        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            return None

    def place_batch_maker_limits(
        self,
        token_id: str,
        direction: str,
        kelly_fraction: float,
        midpoint: float,
        max_exposure_pct: float = 0.10,
        levels: int = 3,
        balance_cap: Optional[float] = None,
    ) -> list:
        """Place multiple maker limit orders at different price levels.

        During low-volatility regimes, a single order at midpoint-0.01 may
        not fill because the book is tight and there's little taker flow.
        Spreading the position across multiple price levels:
        - Increases overall fill probability
        - Captures maker rebates on each level
        - Reduces per-order gas cost relative to total filled volume

        The total position size is the same as a single order — it's just
        distributed across levels. Each level gets an equal share.

        Price levels:
        - Level 0: midpoint - 0.01 (best fill probability, worst price)
        - Level 1: midpoint - 0.02 (moderate fill, better price)
        - Level 2: midpoint - 0.03 (lowest fill, best price for us)

        Args:
            token_id: Which conditional token to buy
            direction: "UP" or "DOWN" (for logging)
            kelly_fraction: Position size fraction from model
            midpoint: Current CLOB midpoint price
            max_exposure_pct: Maximum exposure cap
            levels: Number of price levels to spread across (default 3)
            balance_cap: If set, cap effective balance for sizing

        Returns:
            List of OrderResult for successfully posted orders
        """
        balance: float = self.get_current_balance()
        if balance_cap is not None:
            balance = min(balance, balance_cap)

        # Compute total size at the best price level, then split
        best_price: float = round(midpoint - 0.01, 2)
        if best_price <= 0.0 or best_price >= 1.0:
            best_price = round(midpoint, 2)
            best_price = max(0.01, min(best_price, 0.99))

        total_size: Optional[float] = self._compute_order_size(
            balance, kelly_fraction, best_price, max_exposure_pct
        )
        if total_size is None:
            self.logger.warning(
                f"Batch order skipped: insufficient balance for {direction}"
            )
            return []

        per_level_size: float = round(total_size / levels, 2)
        if per_level_size < 0.01:
            return []

        results: list = []
        for i in range(levels):
            price: float = round(midpoint - 0.01 * (i + 1), 2)
            if price <= 0.0 or price >= 1.0:
                continue

            try:
                order_args = OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=per_level_size,
                    side=BUY,
                )
                signed_order = self.client.create_order(order_args)
                response = self.client.post_order(signed_order, OrderType.GTC)

                result = OrderResult(
                    order_id=response.get("orderID", response.get("id", "unknown")),
                    side=BUY,
                    price=price,
                    size=per_level_size,
                    token_id=token_id,
                    timestamp=time.time(),
                    status="posted",
                )
                self.active_orders.append(result)
                results.append(result)
                self.logger.info(
                    f"Batch order L{i}: {direction} {per_level_size:.2f} @ {price:.2f}"
                )
            except Exception as e:
                self.logger.error(f"Batch order L{i} failed: {e}")

        return results

    def place_rebate_optimized_orders(
        self,
        token_id: str,
        direction: str,
        kelly_fraction: float,
        midpoint: float,
        rebate_config: dict,
        max_exposure_pct: float = 0.10,
        balance_cap: Optional[float] = None,
    ) -> list:
        """Place orders using rebate-optimized pricing from performance.rebate_optimizer.

        When CLOB dollar depth exceeds $50k, batches 3 orders at midpoint - 0.005
        (half the normal offset) to maximize maker rebate capture. The tighter
        offset keeps orders closer to mid for higher fill probability while
        still qualifying as passive (maker) orders.

        Falls back to standard single order placement when depth is insufficient.

        Args:
            token_id: Which conditional token to buy
            direction: "UP" or "DOWN" (for logging)
            kelly_fraction: Position size fraction from model
            midpoint: Current CLOB midpoint price
            rebate_config: Dict from rebate_optimizer() with keys:
                use_rebate_batch, price_offset, levels, dollar_depth
            max_exposure_pct: Maximum exposure cap
            balance_cap: If set, cap effective balance for sizing

        Returns:
            List of OrderResult for successfully posted orders
        """
        if not rebate_config.get("use_rebate_batch", False):
            # Not enough depth for rebate batching — use standard single order
            result = self.place_maker_limit(
                token_id, direction, kelly_fraction, midpoint,
                max_exposure_pct, balance_cap,
            )
            return [result] if result else []

        balance: float = self.get_current_balance()
        if balance_cap is not None:
            balance = min(balance, balance_cap)

        price_offset: float = rebate_config.get("price_offset", 0.005)
        levels: int = rebate_config.get("levels", 3)

        best_price: float = round(midpoint - price_offset, 2)
        if best_price <= 0.0 or best_price >= 1.0:
            best_price = round(midpoint, 2)
            best_price = max(0.01, min(best_price, 0.99))

        total_size: Optional[float] = self._compute_order_size(
            balance, kelly_fraction, best_price, max_exposure_pct
        )
        if total_size is None:
            self.logger.warning(
                f"Rebate batch skipped: insufficient balance for {direction}"
            )
            return []

        per_level_size: float = round(total_size / levels, 2)
        if per_level_size < 0.01:
            return []

        results: list = []
        for i in range(levels):
            price: float = round(midpoint - price_offset * (i + 1), 2)
            if price <= 0.0 or price >= 1.0:
                continue

            try:
                order_args = OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=per_level_size,
                    side=BUY,
                )
                signed_order = self.client.create_order(order_args)
                response = self.client.post_order(signed_order, OrderType.GTC)

                result = OrderResult(
                    order_id=response.get("orderID", response.get("id", "unknown")),
                    side=BUY,
                    price=price,
                    size=per_level_size,
                    token_id=token_id,
                    timestamp=time.time(),
                    status="posted",
                )
                self.active_orders.append(result)
                results.append(result)
                self.logger.info(
                    f"Rebate batch L{i}: {direction} {per_level_size:.2f} "
                    f"@ {price:.3f} (depth=${rebate_config.get('dollar_depth', 0):.0f})"
                )
            except Exception as e:
                self.logger.error(f"Rebate batch L{i} failed: {e}")

        return results

    def cancel_stale_orders(self, max_age_seconds: float = 10.0) -> None:
        """Cancel orders older than max_age_seconds.

        Unfilled orders lock up capital. In a fast-moving 5-minute market,
        an order that hasn't filled within 10 seconds is likely at a stale
        price and should be cancelled to free capital for new signals.

        This is called every 10 seconds by the bot's stale order loop.

        Args:
            max_age_seconds: Cancel orders older than this (default 10s)
        """
        now: float = time.time()
        stale: List[OrderResult] = [
            o
            for o in self.active_orders
            if now - o.timestamp > max_age_seconds and o.status == "posted"
        ]

        for order in stale:
            try:
                self.client.cancel(order.order_id)
                order.status = "cancelled"
                self.logger.info(
                    f"Cancelled stale order {order.order_id} "
                    f"(age={now - order.timestamp:.1f}s)"
                )
            except Exception as e:
                self.logger.warning(
                    f"Cancel failed for {order.order_id}: {e}"
                )

        # Purge completed/cancelled orders from active list
        self.active_orders = [
            o for o in self.active_orders if o.status == "posted"
        ]

    def cancel_all(self) -> None:
        """Cancel all open orders — called on shutdown or hard stop.

        Uses the CLOB bulk cancel endpoint for efficiency rather than
        iterating individual cancels.
        """
        try:
            self.client.cancel_all()
            for o in self.active_orders:
                o.status = "cancelled"
            self.active_orders.clear()
            self.logger.info("All orders cancelled")
        except Exception as e:
            self.logger.error(f"cancel_all failed: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID.

        Used by MM mode to cancel the losing side of a spread-locked
        position while keeping the winning side active.

        Args:
            order_id: CLOB-assigned order identifier

        Returns:
            True if cancel succeeded, False on failure
        """
        try:
            self.client.cancel(order_id)
            for o in self.active_orders:
                if o.order_id == order_id:
                    o.status = "cancelled"
            return True
        except Exception as e:
            self.logger.warning(f"Cancel order {order_id[:12]}... failed: {e}")
            return False

    def place_mm_batch(
        self,
        yes_token_id: str,
        no_token_id: str,
        batch_levels: dict,
        balance: float,
        mm_exposure_pct: float = 0.15,
        batch_size: int = 5,
    ) -> Tuple[List[OrderResult], List[OrderResult]]:
        """Place batch maker limit BUYs on both YES and NO tokens at staggered levels.

        Explosive MM: up to batch_size orders per side at staggered price levels,
        sized to mm_exposure_pct (15%) of balance per side, split across levels.

        Args:
            yes_token_id: YES outcome token ID
            no_token_id: NO outcome token ID
            batch_levels: Dict with 'yes' and 'no' lists of price levels
            balance: Current USDC balance
            mm_exposure_pct: Fraction of balance per side (default 15%)
            batch_size: Number of orders per side (default 5)

        Returns:
            Tuple of (yes_orders, no_orders) lists
        """
        total_orders: int = batch_size * 2
        available: float = balance - self.safety_floor - self.gas_buffer * total_orders
        if available <= 0:
            self.logger.warning(
                f"MM batch: below safety floor: balance={balance:.2f}"
            )
            return ([], [])

        dollar_per_side: float = available * mm_exposure_pct
        dollar_per_side = min(dollar_per_side, available / 2)  # never exceed half
        dollar_per_order: float = dollar_per_side / batch_size

        yes_prices: list = batch_levels.get("yes", [])[:batch_size]
        no_prices: list = batch_levels.get("no", [])[:batch_size]

        yes_orders: List[OrderResult] = []
        no_orders: List[OrderResult] = []

        # Place YES side orders at staggered levels
        for i, price in enumerate(yes_prices):
            if price <= 0:
                continue
            size: float = round(dollar_per_order / price, 2)
            if size < 0.01:
                continue
            try:
                args = OrderArgs(
                    token_id=yes_token_id, price=price,
                    size=size, side=BUY,
                )
                signed = self.client.create_order(args)
                resp = self.client.post_order(signed, OrderType.GTC)
                order = OrderResult(
                    order_id=resp.get("orderID", ""),
                    side=BUY, price=price, size=size,
                    token_id=yes_token_id, timestamp=time.time(), status="posted",
                )
                self.active_orders.append(order)
                yes_orders.append(order)
                self.logger.info(
                    f"MM YES batch [{i+1}/{batch_size}]: {size:.2f} @ {price:.3f}"
                )
            except Exception as e:
                self.logger.error(f"MM YES batch [{i+1}] failed: {e}")

        # Place NO side orders at staggered levels
        for i, price in enumerate(no_prices):
            if price <= 0:
                continue
            size = round(dollar_per_order / price, 2)
            if size < 0.01:
                continue
            try:
                args = OrderArgs(
                    token_id=no_token_id, price=price,
                    size=size, side=BUY,
                )
                signed = self.client.create_order(args)
                resp = self.client.post_order(signed, OrderType.GTC)
                order = OrderResult(
                    order_id=resp.get("orderID", ""),
                    side=BUY, price=price, size=size,
                    token_id=no_token_id, timestamp=time.time(), status="posted",
                )
                self.active_orders.append(order)
                no_orders.append(order)
                self.logger.info(
                    f"MM NO batch [{i+1}/{batch_size}]: {size:.2f} @ {price:.3f}"
                )
            except Exception as e:
                self.logger.error(f"MM NO batch [{i+1}] failed: {e}")

        return (yes_orders, no_orders)
