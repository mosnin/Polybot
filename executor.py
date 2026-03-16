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
from typing import List, Optional

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

    def get_current_balance(self) -> float:
        """Fetch USDC balance from the CLOB client.

        client.get_balance() returns balance in wei (USDC has 6 decimals
        on Polygon), so we divide by 1e6 to get human-readable USDC.

        Returns:
            Current USDC balance as float
        """
        balance_wei: float = float(self.client.get_balance())
        return balance_wei / 1e6

    def get_midpoint(self, token_id: str) -> float:
        """Get CLOB midpoint price for a conditional token.

        The midpoint = (best_bid + best_ask) / 2. For binary markets,
        this represents the market's implied probability of that outcome.

        Args:
            token_id: Polymarket conditional token identifier

        Returns:
            Midpoint price as float (0.00 to 1.00)
        """
        return float(self.client.get_midpoint(token_id))

    def get_order_book(self, token_id: str) -> dict:
        """Get full L2 order book for a conditional token.

        Returns dict with 'bids' and 'asks' arrays, each containing
        [price, size] entries sorted by price.

        Args:
            token_id: Polymarket conditional token identifier

        Returns:
            Dict with 'bids' and 'asks' arrays
        """
        return self.client.get_order_book(token_id)

    def _compute_order_size(
        self,
        balance: float,
        kelly_fraction: float,
        price: float,
        max_exposure_pct: float = 0.10,
    ) -> Optional[float]:
        """Compute order size in conditional tokens with all safety guards.

        Sizing pipeline:
        1. Available capital = balance - safety_floor - gas_buffer
           (never touch the safety floor — it's our survival guarantee)
        2. Dollar risk = available * kelly_fraction
           (Kelly tells us the optimal fraction of bankroll to risk)
        3. Clamp to 5%-max_exposure_pct of total balance
           (hard limits prevent model errors from oversizing)
        4. Size in tokens = dollar_risk / price
           (convert dollar amount to token quantity)

        The clamping is a second safety net beyond Kelly.
        Even if the model outputs a high Kelly fraction due to a
        spurious edge, the clamp prevents catastrophic position sizes.

        Args:
            balance: Current USDC balance (live from get_current_balance)
            kelly_fraction: Optimal bet fraction from model (0 to 0.25)
            price: Order price per token
            max_exposure_pct: Maximum exposure as fraction of balance (0.05-0.10),
                              adjustable via dashboard slider

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

        # Kelly-sized dollar risk
        dollar_risk: float = available * kelly_fraction

        # Hard clamp to 5%-max_exposure_pct of total balance regardless of Kelly output
        min_risk: float = balance * 0.05
        max_risk: float = balance * max_exposure_pct
        dollar_risk = max(min(dollar_risk, max_risk), min_risk)

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
    ) -> Optional[OrderResult]:
        """Place a maker limit order targeting passive fills and rebates.

        Pricing strategy: order at midpoint - 0.01 (one tick below mid).
        This places the order on the passive (maker) side of the book:
        - Order sits on the book waiting for a taker to cross
        - When filled, we pay maker fees (often negative = rebate)
        - The 0.01 offset is the minimum tick size on Polymarket

        Why maker > taker:
        - Taker fee: ~0.15% → costs eat into edge
        - Maker fee: ~0% or negative → preserves/enhances edge
        - Over 1000+ trades, this difference is the margin between
          profit and loss for small-edge strategies

        We always BUY the directional token (UP token if direction="UP",
        DOWN token if direction="DOWN"). The bot.py caller selects the
        correct token_id based on direction.

        Args:
            token_id: Which conditional token to buy
            direction: "UP" or "DOWN" (for logging)
            kelly_fraction: Position size fraction from model
            midpoint: Current CLOB midpoint price
            max_exposure_pct: Maximum exposure cap, adjustable via dashboard

        Returns:
            OrderResult if successfully posted, None on failure
        """
        balance: float = self.get_current_balance()

        # Price one tick below midpoint → sits on the passive (maker) side.
        # This means we might not get filled immediately, but when we do,
        # we capture the maker rebate.
        price: float = round(midpoint - 0.01, 2)

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
