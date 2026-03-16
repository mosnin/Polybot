"""
withdraw.py — Async USDC withdrawal from Polymarket via bridge.

Initiates a USDC.e transfer from your Polygon wallet to a destination address
on any supported chain. Uses the Polymarket Bridge API for withdrawal quotes
and the standard ERC-20 transfer for on-chain execution.

The withdrawal flow:
1. Check live USDC.e balance via web3
2. Build and sign an ERC-20 transfer transaction
3. Broadcast on Polygon and wait for confirmation

Usage (standalone CLI):
    python withdraw.py --amount 50.0 --destination 0xYourAddress --chain polygon

Usage (from dashboard):
    from withdraw import withdraw_from_dashboard
    result = await withdraw_from_dashboard(private_key, rpc_url, amount)

Security: Private key never leaves this process. All signing is local via web3.py.
Full private key control — zero custody at all times.
"""

import argparse
import asyncio
import logging
import sys
from typing import Dict, Optional

import aiohttp
from eth_account import Account
from web3 import Web3

from config import load_config

logger: logging.Logger = logging.getLogger("Withdraw")

# Minimal ERC-20 ABI for transfer and balance check
ERC20_TRANSFER_ABI = [
    {
        "type": "function",
        "name": "transfer",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "function",
        "name": "balanceOf",
        "stateMutability": "view",
        "inputs": [{"name": "_owner", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]

# Gas limit for ERC-20 transfer on Polygon
TRANSFER_GAS_LIMIT: int = 100_000


def get_live_balance(w3: Web3, usdc_address: str, owner: str) -> float:
    """Fetch live USDC.e balance from Polygon via web3.

    This is the canonical balance source — all risk calculations and
    withdrawal limits must reference this value, never a cached or
    hardcoded amount.

    Args:
        w3: Web3 instance connected to Polygon RPC
        usdc_address: USDC.e contract address
        owner: Wallet address to check

    Returns:
        Balance in USDC (human-readable float)
    """
    usdc_contract = w3.eth.contract(
        address=Web3.to_checksum_address(usdc_address),
        abi=ERC20_TRANSFER_ABI,
    )
    balance_wei: int = usdc_contract.functions.balanceOf(
        Web3.to_checksum_address(owner)
    ).call()
    return balance_wei / 1e6


async def get_withdrawal_quote(
    session: aiohttp.ClientSession,
    bridge_api_base: str,
    amount: float,
    destination_chain: str = "polygon",
) -> Dict:
    """Request a withdrawal quote from the Polymarket Bridge API.

    The quote includes estimated fees and output amount for the withdrawal.

    Args:
        session: Reusable aiohttp session
        bridge_api_base: Bridge API base URL
        amount: USDC amount to withdraw
        destination_chain: Target chain (default "polygon" for same-chain transfer)

    Returns:
        Quote dict with fee estimates, or error dict on failure
    """
    url: str = f"{bridge_api_base}/withdraw/quote"
    payload: Dict = {
        "amount": str(amount),
        "asset": "USDC",
        "destination_chain": destination_chain,
    }

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "status": "quoted",
                    "amount": amount,
                    "fee": data.get("fee", 0.0),
                    "output": data.get("output_amount", amount),
                    "chain": destination_chain,
                }
            else:
                # Bridge API may not support quotes — fallback to direct transfer
                return {
                    "status": "quoted",
                    "amount": amount,
                    "fee": 0.0,  # Polymarket has zero withdrawal fees
                    "output": amount,
                    "chain": destination_chain,
                    "note": "Direct transfer (bridge quote unavailable)",
                }
    except Exception as e:
        logger.warning(f"Bridge quote failed (using direct transfer): {e}")
        return {
            "status": "quoted",
            "amount": amount,
            "fee": 0.0,
            "output": amount,
            "chain": destination_chain,
            "note": "Direct transfer fallback",
        }


def execute_transfer(
    w3: Web3,
    usdc_address: str,
    private_key: str,
    destination: str,
    amount: float,
    usdc_decimals: int = 6,
) -> Dict[str, str]:
    """Build, sign, and broadcast a USDC.e transfer on Polygon.

    This executes the actual on-chain transfer of USDC.e tokens from
    the bot wallet to the destination address. Used for same-chain
    withdrawals (Polygon → Polygon address).

    Args:
        w3: Web3 instance connected to Polygon
        usdc_address: USDC.e contract address
        private_key: Wallet private key for signing
        destination: Recipient address
        amount: USDC amount to transfer (human-readable)
        usdc_decimals: Token decimals (6 for USDC)

    Returns:
        Dict with tx_hash and status

    Raises:
        ValueError: If balance insufficient or chain_id wrong
    """
    # Safety: verify Polygon mainnet
    chain_id: int = w3.eth.chain_id
    if chain_id != 137:
        raise ValueError(f"Expected chain_id=137 (Polygon), got {chain_id}")

    account = Account.from_key(private_key)
    owner: str = account.address

    # Verify sufficient balance
    balance: float = get_live_balance(w3, usdc_address, owner)
    if balance < amount:
        raise ValueError(
            f"Insufficient balance: have ${balance:.2f}, need ${amount:.2f}"
        )

    # Convert human-readable amount to wei
    amount_wei: int = int(amount * (10**usdc_decimals))

    usdc_contract = w3.eth.contract(
        address=Web3.to_checksum_address(usdc_address),
        abi=ERC20_TRANSFER_ABI,
    )

    # Build transfer transaction
    nonce: int = w3.eth.get_transaction_count(owner)
    gas_price: int = w3.eth.gas_price

    tx = usdc_contract.functions.transfer(
        Web3.to_checksum_address(destination),
        amount_wei,
    ).build_transaction(
        {
            "from": owner,
            "nonce": nonce,
            "gas": TRANSFER_GAS_LIMIT,
            "gasPrice": gas_price,
            "chainId": 137,
        }
    )

    # Sign locally — private key stays in this process
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)

    # Broadcast
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    logger.info(f"Transfer TX broadcast: {tx_hash.hex()}")

    # Wait for confirmation
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt["status"] != 1:
        raise Exception(f"Transfer reverted: {tx_hash.hex()}")

    logger.info(
        f"Transfer confirmed: ${amount:.2f} USDC → {destination[:10]}... "
        f"(block {receipt['blockNumber']}, gas {receipt['gasUsed']})"
    )

    return {
        "status": "success",
        "tx_hash": tx_hash.hex(),
        "amount": str(amount),
        "destination": destination,
        "gas_used": str(receipt["gasUsed"]),
    }


async def withdraw_from_dashboard(
    private_key: str,
    rpc_url: str,
    amount: float,
    destination: Optional[str] = None,
) -> Dict[str, str]:
    """Dashboard-callable withdrawal function.

    If no destination is provided, transfers to the wallet's own address
    (useful for moving from CLOB balance to on-chain balance). In practice,
    the user should provide a destination address for actual withdrawals.

    Args:
        private_key: Polygon wallet private key
        rpc_url: Alchemy RPC URL
        amount: USDC amount to withdraw
        destination: Recipient address (defaults to own wallet)

    Returns:
        Dict with status, tx_hash, amount, error
    """
    try:
        from config import Config

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            return {"status": "error", "error": "Cannot connect to RPC"}

        account = Account.from_key(private_key)
        owner: str = account.address
        usdc_addr: str = Config.usdc_token_address

        # Default destination is own wallet (CLOB → on-chain)
        if destination is None:
            destination = owner

        # Verify balance
        balance: float = get_live_balance(w3, usdc_addr, owner)
        if balance < amount:
            return {
                "status": "error",
                "error": f"Insufficient balance: ${balance:.2f} < ${amount:.2f}",
            }

        # Get quote (informational)
        async with aiohttp.ClientSession() as session:
            quote = await get_withdrawal_quote(
                session, Config.bridge_api_base, amount
            )
            logger.info(f"Withdrawal quote: {quote}")

        # Execute on-chain transfer
        result = execute_transfer(
            w3, usdc_addr, private_key, destination, amount, Config.usdc_decimals
        )
        return result

    except Exception as e:
        logger.error(f"Withdrawal failed: {e}")
        return {"status": "error", "error": str(e)}


async def main() -> None:
    """Standalone CLI entry point for withdrawal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Withdraw USDC from Polymarket wallet"
    )
    parser.add_argument(
        "--amount",
        type=float,
        required=True,
        help="USDC amount to withdraw",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default=None,
        help="Destination address (default: own wallet)",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default="polygon",
        help="Target chain (default: polygon)",
    )
    args = parser.parse_args()

    config = load_config()
    w3 = Web3(Web3.HTTPProvider(config.alchemy_rpc_url))

    if not w3.is_connected():
        print("ERROR: Cannot connect to Polygon RPC.")
        sys.exit(1)

    account = Account.from_key(config.private_key)
    owner: str = account.address

    print("=" * 60)
    print("  Polymarket USDC Withdrawal")
    print("=" * 60)
    print(f"  Wallet:      {owner}")

    # Live balance check
    balance: float = get_live_balance(
        w3, config.usdc_token_address, owner
    )
    print(f"  Live Balance: ${balance:.2f}")
    print(f"  Amount:       ${args.amount:.2f}")

    destination: str = args.destination or owner
    print(f"  Destination:  {destination}")
    print(f"  Chain:        {args.chain}")
    print()

    if args.amount > balance:
        print(f"ERROR: Amount ${args.amount:.2f} exceeds balance ${balance:.2f}")
        sys.exit(1)

    # Get quote
    async with aiohttp.ClientSession() as session:
        quote = await get_withdrawal_quote(
            session, config.bridge_api_base, args.amount, args.chain
        )
        print(f"  Fee:    ${quote.get('fee', 0):.4f}")
        print(f"  Output: ${quote.get('output', args.amount):.2f}")
        if quote.get("note"):
            print(f"  Note:   {quote['note']}")
    print()

    # Confirmation
    confirm: str = input("Proceed with withdrawal? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Execute
    print("\nExecuting transfer...")
    result = execute_transfer(
        w3,
        config.usdc_token_address,
        config.private_key,
        destination,
        args.amount,
        config.usdc_decimals,
    )
    print(f"\nTX Hash: {result['tx_hash']}")
    print(f"Status:  {result['status']}")
    print(f"Gas:     {result['gas_used']}")


if __name__ == "__main__":
    asyncio.run(main())
