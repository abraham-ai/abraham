import json
import time
from typing import Optional, Tuple, Dict, Any

from web3 import Web3
from web3.exceptions import ContractLogicError, TimeExhausted
from eth_account import Account


import ipfs

from config import (
    logger,
    BASE_SEPOLIA_RPC,
    PRIVATE_KEY,
    CHAIN_ID,
    CONTRACT_ADDRESS_AUCTION,
    CONTRACT_ABI_AUCTION,
)

# ---------- Errors ----------

class BlockchainError(Exception):
    pass


# ---------- Helpers ----------

def _fmt_gwei(wei: int) -> str:
    return f"{wei / 1_000_000_000:.3f} gwei"


def _explorer_url(tx_hash_hex: str) -> str:
    # Base Sepolia explorer
    return f"https://sepolia.basescan.org/tx/{tx_hash_hex}"


def make_w3() -> Web3:
    w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise BlockchainError("Web3 connection failed")
    return w3


def resolve_nonce(w3: Web3, address: str, provided_nonce: Optional[int]) -> int:
    pending_nonce = w3.eth.get_transaction_count(address, "pending")
    if provided_nonce is None:
        return pending_nonce
    if provided_nonce > pending_nonce:
        logger.warning(f"Provided nonce {provided_nonce} > pending {pending_nonce}; tx will queue until gaps fill.")
    elif provided_nonce < pending_nonce:
        logger.warning(f"Provided nonce {provided_nonce} < pending {pending_nonce}; using {pending_nonce}.")
        return pending_nonce
    return provided_nonce


def suggest_fees(
    w3: Web3,
    urgency: str = "fast",
    min_priority_gwei: float = 1.0,
) -> Dict[str, int]:
    """
    EIP-1559 fee suggestion. Works on Base Sepolia where max_priority_fee may be 0.
    Returns either {'maxFeePerGas', 'maxPriorityFeePerGas} or {'gasPrice'} (legacy).
    """
    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas")
    if base_fee is None:
        # Legacy (no EIP-1559). Small buffer + floor.
        gp = max(int(w3.eth.gas_price * 1.2), w3.to_wei(min_priority_gwei * 2, "gwei"))
        return {"gasPrice": gp}

    mult = {
        "slow": 1.2,
        "normal": 1.5,
        "fast": 2.0,
        "urgent": 3.0,
    }.get(urgency, 2.0)

    # Priority fee may be 0 on some RPCs; clamp to a sane minimum.
    try:
        prio = int(w3.eth.max_priority_fee)
    except Exception:
        prio = 0
    prio = max(prio, w3.to_wei(min_priority_gwei, "gwei"))

    # Compute max fee AFTER priority is finalized.
    max_fee = int(base_fee * mult) + prio

    # Final safety: never let maxFee < priority (violates EIP‑1559).
    if max_fee < prio:
        max_fee = prio

    return {"maxFeePerGas": max_fee, "maxPriorityFeePerGas": prio}


def simulate_call(
    contract_function,
    from_address: str,
    value: int = 0,
) -> None:
    """
    Run a dry call to surface reverts pre-broadcast.
    """
    try:
        # Call with only essentials; don't pass gas/fees to avoid masking errors.
        contract_function.call({"from": from_address, "value": value})
    except ContractLogicError as e:
        # Most useful error for devs
        raise BlockchainError(f"Simulation reverted: {e}") from e
    except ValueError as e:
        # JSON-RPC error shape commonly in e.args[0]
        msg = e.args[0].get("message") if e.args and isinstance(e.args[0], dict) else str(e)
        raise BlockchainError(f"Simulation failed: {msg}") from e
    except Exception as e:
        raise BlockchainError(f"Simulation failed: {e}") from e


def estimate_gas(
    contract_function,
    from_address: str,
    nonce: int,
    value: int = 0,
    gas_limit_cap: int = 1_200_000,
) -> int:
    """
    Try to get an accurate estimate. Add a small buffer and cap. If estimate fails,
    let caller decide a default fallback.
    """
    est = contract_function.estimate_gas({"from": from_address, "nonce": nonce, "value": value})
    gas_limit = int(est * 1.20)  # 20% buffer
    gas_limit = min(gas_limit, gas_limit_cap)
    return gas_limit


def wait_for_confirmations(w3, tx_hash, confirmations=3, inclusion_timeout_s=120, conf_timeout_s=180, poll_interval=1.0):
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=inclusion_timeout_s)
    if receipt.status != 1:
        raise BlockchainError(f"Transaction reverted in block {receipt.blockNumber}: {tx_hash.hex()}")
    target = receipt.blockNumber + confirmations
    t0 = time.time()
    while w3.eth.block_number < target:
        if time.time() - t0 > conf_timeout_s:
            raise BlockchainError(f"Timed out waiting for {confirmations} confirmations: {tx_hash.hex()}")
        time.sleep(poll_interval)
    return receipt


# ---------- Core: safe_send ----------

def safe_send(
    w3: Web3,
    contract_function,
    owner_account,
    *,
    op_name: str = "TX",
    nonce: Optional[int] = None,
    value: int = 0,
    default_gas_limit: int = 250_000,
    gas_limit_cap: int = 1_200_000,
    confirmations: int = 3,
    timeout_s: int = 180,
    poll_interval: float = 1.0,
    urgency: str = "fast",
    speed_up_on_timeout: bool = True,
    speed_up_bump: float = 1.125,  # 12.5% bump when replacing
) -> Tuple[bytes, Any]:
    """
    1) simulate, 2) estimate gas, 3) build+sign+send (EIP-1559 if available), 4) wait N confirmations.
    On inclusion timeout, optionally re-broadcast with higher fee using the same nonce.
    """
    addr = owner_account.address

    # ---- 1) simulate
    simulate_call(contract_function, addr, value=value)

    # ---- 2) nonce + gas + fees
    use_nonce = resolve_nonce(w3, addr, nonce)
    try:
        gas_limit = estimate_gas(
            contract_function, addr, use_nonce, value=value, gas_limit_cap=gas_limit_cap
        )
    except Exception as e:
        logger.warning(f"{op_name}: gas estimate failed: {e}; using default={default_gas_limit}")
        gas_limit = default_gas_limit

    fee_params = suggest_fees(w3, urgency=urgency)
    
    # Guard against RPC oddities: ensure maxFeePerGas >= maxPriorityFeePerGas
    if "maxFeePerGas" in fee_params and "maxPriorityFeePerGas" in fee_params:
        if fee_params["maxFeePerGas"] < fee_params["maxPriorityFeePerGas"]:
            latest_base = w3.eth.get_block("latest").get("baseFeePerGas", 0)
            # Lift maxFee to be at least prio + base fee (or prio + 0.1 gwei if base fee is tiny)
            fee_params["maxFeePerGas"] = fee_params["maxPriorityFeePerGas"] + max(
                latest_base, w3.to_wei(0.1, "gwei")
            )

    # ---- 3) build + sign + send
    common_fields = {
        "chainId": CHAIN_ID or w3.eth.chain_id,
        "from": addr,
        "nonce": use_nonce,
        "gas": gas_limit,
        "value": value,
    }
    tx_fields = {**common_fields, **fee_params}
    if "maxFeePerGas" in fee_params and "maxPriorityFeePerGas" in fee_params:
        tx_fields["type"] = 2  # explicit EIP-1559

    tx = contract_function.build_transaction(tx_fields)
    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)

    logger.info(
        f"→ {op_name} nonce={use_nonce} gas={gas_limit} "
        + (
            f"maxFee={_fmt_gwei(fee_params['maxFeePerGas'])} prio={_fmt_gwei(fee_params['maxPriorityFeePerGas'])}"
            if "maxFeePerGas" in fee_params
            else f"gasPrice={_fmt_gwei(fee_params['gasPrice'])}"
        )
    )

    try:
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    except ValueError as e:
        msg = e.args[0].get("message") if e.args and isinstance(e.args[0], dict) else str(e)
        if "replacement transaction underpriced" in msg.lower():
            raise BlockchainError(f"{op_name}: replacement underpriced") from e
        if "nonce too low" in msg.lower():
            raise BlockchainError(f"{op_name}: nonce too low (another tx likely mined)") from e
        if "insufficient funds" in msg.lower():
            raise BlockchainError(f"{op_name}: insufficient funds for gas") from e
        raise BlockchainError(f"{op_name}: broadcast error: {msg}") from e

    tx_hex = tx_hash.hex()
    logger.info(f"TX: {tx_hex}")
    logger.info(f"Explorer: {_explorer_url(tx_hex)}")

    # ---- 4) wait for confirmations, possibly speed up if stuck
    try:
        receipt = wait_for_confirmations(
            w3=w3, tx_hash=tx_hash, confirmations=confirmations, conf_timeout_s=timeout_s, inclusion_timeout_s=timeout_s, poll_interval=poll_interval
        )

        eff = receipt.get("effectiveGasPrice")
        logger.info(
            f"✅ {op_name} confirmed: {tx_hex} | block={receipt.blockNumber} gasUsed={receipt.gasUsed}"
            + (f" effGasPrice={_fmt_gwei(eff)}" if eff is not None else "")
        )

        return tx_hash, receipt

    except BlockchainError as first_err:
        # Inclusion/confirmation timeout. Optionally replace with higher tip and try once or twice.
        if not speed_up_on_timeout:
            raise

        logger.warning(f"{op_name}: {first_err}. Attempting speed-up replacement (same nonce).")

        # Fetch current head fees and bump
        attempts = 0
        last_err = first_err
        while attempts < 2:  # keep minimal (1–2 attempts)
            attempts += 1
            bump_fees = suggest_fees(w3, urgency="urgent")
            if "maxFeePerGas" in bump_fees:
                bump_fees["maxFeePerGas"] = int(max(bump_fees["maxFeePerGas"], fee_params.get("maxFeePerGas", 0)) * speed_up_bump)
                bump_fees["maxPriorityFeePerGas"] = int(max(bump_fees["maxPriorityFeePerGas"], fee_params.get("maxPriorityFeePerGas", 0)) * speed_up_bump)
                tx_fields = {**common_fields, **bump_fees, "type": 2}
            else:
                bump_fees["gasPrice"] = int(max(bump_fees["gasPrice"], fee_params.get("gasPrice", 0)) * speed_up_bump)
                tx_fields = {**common_fields, **bump_fees}

            tx_replacement = contract_function.build_transaction(tx_fields)
            signed_replacement = w3.eth.account.sign_transaction(tx_replacement, private_key=PRIVATE_KEY)

            try:
                tx_hash = w3.eth.send_raw_transaction(signed_replacement.raw_transaction)
                tx_hex = tx_hash.hex()
                logger.info(
                    f"↻ Speed-up attempt {attempts}: {_fmt_gwei(bump_fees.get('maxFeePerGas', bump_fees.get('gasPrice', 0)))} → {tx_hex}  {_explorer_url(tx_hex)}"
                )
                receipt = wait_for_confirmations(
                    w3=w3,
                    tx_hash=tx_hash,
                    confirmations=confirmations,
                    conf_timeout_s=timeout_s,
                    inclusion_timeout_s=timeout_s,
                    poll_interval=poll_interval,
                )
                logger.info(
                    f"✅ {op_name} confirmed after speed-up: {tx_hex} | block={receipt.blockNumber}"
                )
                return tx_hash, receipt

            except ValueError as e:
                msg = e.args[0].get("message") if e.args and isinstance(e.args[0], dict) else str(e)
                last_err = BlockchainError(f"Speed-up broadcast failed: {msg}")
                logger.warning(str(last_err))
                # Continue loop to try second bump
            except BlockchainError as e:
                last_err = e
                logger.warning(str(e))
                # Continue loop to try once more

        # If we got here, both attempts failed
        raise last_err




def get_creation():
    jpg1_path = "Eden_creation_gene3_A-group-of-humans-and-cute-cyborgs-living-vanlife-in-the-desert,_66e7305c058bec36f6c4c306_0.png"

    ipfs_image = ipfs.pin(jpg1_path)
    image_hash = ipfs_image.split("/")[-1]
    
    json_data = {
        "description": "here is a description",
        "external_url": "https://abraham.ai",
        "image": f"ipfs://{image_hash}",
        "name": "Abraham Says hello",
        "attributes": [
            {
                "trait_type": "Medium",
                "value": "Digital Genesis"
            },
            {
                "trait_type": "Palette",
                "value": "Quantum Purple"
            },
            {
                "trait_type": "Origin",
                "value": "Collective Imagination"
            },
            {
                "trait_type": "Autonomy Level",
                "value": "Emergent"
            }
        ]
    }

    ipfs_url = ipfs.pin(json_data)
    ipfs_hash = ipfs_url.split("/")[-1]

    return ipfs_hash



# ---------- Test/main ----------

def main():
    # Load ABI
    with open(CONTRACT_ABI_AUCTION, "r") as f:
        contract_abi = json.load(f)

    w3 = make_w3()
    owner = Account.from_key(PRIVATE_KEY)

    contract = w3.eth.contract(address=CONTRACT_ADDRESS_AUCTION, abi=contract_abi)

    # Get the creation
    ipfs_hash = get_creation()

    # Your test function call
    contract_function = contract.functions.setTokenURI(
        0,
        f"ipfs://{ipfs_hash}"
    )

    # contract_function = contract.functions.startGenesisAuction()

    # Send with slightly aggressive fees and 3 confirmations
    try:
        safe_send(
            w3,
            contract_function,
            owner,
            op_name="SET_TOKEN_URI",
            nonce=None,               # or set an explicit nonce to pin
            value=0,                  # non-payable
        )

    except BlockchainError as e:
        logger.error(f"❌ SET_TOKEN_URI failed: {e}")
        raise


if __name__ == "__main__":
    main()
