"""
approve_usdc.py — One-time USDC.e approval for Polymarket CTF Exchange contracts.

Before the bot can place orders, the wallet must approve Polymarket's exchange
contracts to spend USDC.e on its behalf. This script sets infinite allowance
(2^256 - 1) on both the standard CTF Exchange and the Neg Risk CTF Exchange.

This is a one-time operation per wallet. Once approved, the allowance persists
until explicitly revoked. The script checks existing allowance before approving
to avoid unnecessary gas spend.

Usage (standalone):
    python approve_usdc.py

Usage (from dashboard):
    from approve_usdc import approve_from_dashboard
    result = approve_from_dashboard(private_key, rpc_url)

Security: The private key never leaves this process. All signing happens locally
via web3.py. Zero custody — you control your keys at all times.
"""

import logging
import sys
from typing import Dict, Optional

from eth_account import Account
from web3 import Web3

from config import load_config

logger: logging.Logger = logging.getLogger("ApproveUSDC")

# Infinite allowance — standard practice for DeFi approvals.
# This avoids needing to re-approve after each trade.
INFINITE_ALLOWANCE: int = 2**256 - 1

# Threshold below which we consider allowance "not set" and need to approve.
# Using 2^128 as threshold — anything above this is effectively infinite.
ALLOWANCE_THRESHOLD: int = 2**128

# Gas limit for ERC-20 approve transaction on Polygon.
# approve() is a simple storage write, typically uses ~46k gas.
# 150k provides generous headroom for network congestion.
APPROVE_GAS_LIMIT: int = 150_000

# Minimal ERC-20 ABI — only the functions we need for approval
ERC20_APPROVE_ABI = [
    {
        "type": "function",
        "name": "approve",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "function",
        "name": "allowance",
        "stateMutability": "view",
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "type": "function",
        "name": "balanceOf",
        "stateMutability": "view",
        "inputs": [{"name": "_owner", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


def check_allowance(
    usdc_contract, owner: str, spender: str
) -> int:
    """Read current USDC.e allowance for a spender.

    Args:
        usdc_contract: Web3 contract instance for USDC.e
        owner: Wallet address that owns the tokens
        spender: Contract address that would spend the tokens

    Returns:
        Current allowance in wei (USDC has 6 decimals)
    """
    return usdc_contract.functions.allowance(
        Web3.to_checksum_address(owner),
        Web3.to_checksum_address(spender),
    ).call()


def get_usdc_balance(usdc_contract, owner: str) -> float:
    """Read current USDC.e balance for a wallet.

    Args:
        usdc_contract: Web3 contract instance for USDC.e
        owner: Wallet address

    Returns:
        Balance in USDC (human-readable, 6 decimal conversion)
    """
    balance_wei: int = usdc_contract.functions.balanceOf(
        Web3.to_checksum_address(owner)
    ).call()
    return balance_wei / 1e6


def approve_spender(
    w3: Web3,
    usdc_contract,
    private_key: str,
    spender: str,
    gas_limit: int = APPROVE_GAS_LIMIT,
) -> str:
    """Build, sign, and broadcast an ERC-20 approve transaction.

    Sets infinite allowance (2^256 - 1) for the spender on USDC.e.
    This allows the Polymarket exchange to escrow USDC when placing orders.

    Args:
        w3: Web3 instance connected to Polygon RPC
        usdc_contract: Web3 contract instance for USDC.e
        private_key: Wallet private key for signing (0x-prefixed)
        spender: Exchange contract address to approve
        gas_limit: Maximum gas units (default 150,000)

    Returns:
        Transaction hash as hex string

    Raises:
        Exception: If transaction fails or times out
    """
    account = Account.from_key(private_key)
    owner: str = account.address

    # Safety: verify we're on Polygon mainnet
    chain_id: int = w3.eth.chain_id
    if chain_id != 137:
        raise ValueError(
            f"Expected Polygon mainnet (chain_id=137), got {chain_id}. "
            "Check your ALCHEMY_RPC_URL."
        )

    # Build the approve transaction
    nonce: int = w3.eth.get_transaction_count(owner)
    gas_price: int = w3.eth.gas_price

    tx = usdc_contract.functions.approve(
        Web3.to_checksum_address(spender),
        INFINITE_ALLOWANCE,
    ).build_transaction(
        {
            "from": owner,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": gas_price,
            "chainId": 137,
        }
    )

    # Sign locally — private key never leaves this process
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)

    # Broadcast
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    logger.info(f"Approve TX broadcast: {tx_hash.hex()}")

    # Wait for confirmation (timeout 120s)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt["status"] != 1:
        raise Exception(
            f"Approve transaction reverted: {tx_hash.hex()}"
        )

    logger.info(
        f"Approve confirmed in block {receipt['blockNumber']} "
        f"(gas used: {receipt['gasUsed']})"
    )
    return tx_hash.hex()


def approve_from_dashboard(private_key: str, rpc_url: str) -> Dict[str, str]:
    """Dashboard-callable approval function.

    Checks and approves both CTF Exchange contracts in one call.
    Returns a result dict for the dashboard to display.

    Args:
        private_key: Polygon wallet private key
        rpc_url: Alchemy RPC URL for Polygon

    Returns:
        Dict with keys: status, ctf_tx, neg_risk_tx, error
    """
    try:
        from config import Config

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            return {"status": "error", "error": "Cannot connect to RPC"}

        account = Account.from_key(private_key)
        owner: str = account.address

        # Load contract addresses from Config defaults
        usdc_addr: str = Config.usdc_token_address
        ctf_addr: str = Config.ctf_exchange_address
        neg_risk_addr: str = Config.neg_risk_ctf_exchange_address

        usdc_contract = w3.eth.contract(
            address=Web3.to_checksum_address(usdc_addr),
            abi=ERC20_APPROVE_ABI,
        )

        result: Dict[str, str] = {
            "status": "already_approved",
            "ctf_tx": "",
            "neg_risk_tx": "",
        }

        # Check and approve CTF Exchange
        ctf_allowance: int = check_allowance(usdc_contract, owner, ctf_addr)
        if ctf_allowance < ALLOWANCE_THRESHOLD:
            tx_hash = approve_spender(w3, usdc_contract, private_key, ctf_addr)
            result["ctf_tx"] = tx_hash
            result["status"] = "success"
            logger.info(f"CTF Exchange approved: {tx_hash}")

        # Check and approve Neg Risk CTF Exchange
        neg_allowance: int = check_allowance(usdc_contract, owner, neg_risk_addr)
        if neg_allowance < ALLOWANCE_THRESHOLD:
            tx_hash = approve_spender(
                w3, usdc_contract, private_key, neg_risk_addr
            )
            result["neg_risk_tx"] = tx_hash
            result["status"] = "success"
            logger.info(f"Neg Risk CTF Exchange approved: {tx_hash}")

        return result

    except Exception as e:
        logger.error(f"Approval failed: {e}")
        return {"status": "error", "error": str(e)}


def main() -> None:
    """Standalone CLI entry point for USDC approval.

    Loads config from .env, checks existing allowances, and approves
    both exchange contracts if needed. Includes confirmation prompt.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s",
    )

    print("=" * 60)
    print("  Polymarket USDC.e Approval Script")
    print("=" * 60)

    config = load_config()
    w3 = Web3(Web3.HTTPProvider(config.alchemy_rpc_url))

    if not w3.is_connected():
        print("ERROR: Cannot connect to Polygon RPC. Check ALCHEMY_RPC_URL.")
        sys.exit(1)

    chain_id: int = w3.eth.chain_id
    print(f"  Chain ID: {chain_id} ({'Polygon' if chain_id == 137 else 'UNKNOWN'})")

    account = Account.from_key(config.private_key)
    owner: str = account.address
    print(f"  Wallet:   {owner}")

    usdc_contract = w3.eth.contract(
        address=Web3.to_checksum_address(config.usdc_token_address),
        abi=ERC20_APPROVE_ABI,
    )

    balance: float = get_usdc_balance(usdc_contract, owner)
    print(f"  USDC.e:   ${balance:.2f}")
    print()

    # Check CTF Exchange allowance
    ctf_allowance: int = check_allowance(
        usdc_contract, owner, config.ctf_exchange_address
    )
    ctf_needs_approval: bool = ctf_allowance < ALLOWANCE_THRESHOLD
    print(
        f"  CTF Exchange ({config.ctf_exchange_address[:10]}...): "
        f"{'NEEDS APPROVAL' if ctf_needs_approval else 'Already approved'}"
    )

    # Check Neg Risk CTF Exchange allowance
    neg_allowance: int = check_allowance(
        usdc_contract, owner, config.neg_risk_ctf_exchange_address
    )
    neg_needs_approval: bool = neg_allowance < ALLOWANCE_THRESHOLD
    print(
        f"  Neg Risk CTF ({config.neg_risk_ctf_exchange_address[:10]}...): "
        f"{'NEEDS APPROVAL' if neg_needs_approval else 'Already approved'}"
    )
    print()

    if not ctf_needs_approval and not neg_needs_approval:
        print("Both contracts already approved. No action needed.")
        return

    # Confirmation prompt
    print("This will broadcast approval transaction(s) on Polygon mainnet.")
    print(f"Estimated gas cost: ~$0.01-0.05 per approval")
    confirm: str = input("Proceed? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Approve
    if ctf_needs_approval:
        print("\nApproving CTF Exchange...")
        tx = approve_spender(
            w3, usdc_contract, config.private_key, config.ctf_exchange_address
        )
        print(f"  TX Hash: {tx}")

    if neg_needs_approval:
        print("\nApproving Neg Risk CTF Exchange...")
        tx = approve_spender(
            w3, usdc_contract, config.private_key,
            config.neg_risk_ctf_exchange_address,
        )
        print(f"  TX Hash: {tx}")

    print("\nAll approvals complete. You can now run the bot.")


if __name__ == "__main__":
    main()
