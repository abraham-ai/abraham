import json
import time
import random
import modal
import requests
from web3 import Web3
from eth_account import Account
from typing import Any, Dict, List, Tuple, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    logger, 
    IPFS_PREFIX, 
    IPFS_BASE_URL, 
    PINATA_JWT,
    BASE_SEPOLIA_RPC,
    PRIVATE_KEY,
    CONTRACT_ADDRESS,
    CONTRACT_ABI,
    CHAIN_ID,
    GAS_LIMIT_CREATE_SESSION,
    GAS_LIMIT_UPDATE_SESSION,
    SUBGRAPH_URL,
    ABRAHAM_ADDRESS
)

nonce_dict = modal.Dict.from_name("nonce_dict", create_if_missing=True)

class BlockchainError(Exception):
    """Error during blockchain operations."""
    pass


def check_pending_transactions(w3, account_address: str) -> int:
    """Check for pending transactions and return the count."""
    try:
        pending_count = w3.eth.get_transaction_count(account_address, 'pending')
        confirmed_count = w3.eth.get_transaction_count(account_address, 'latest')
        
        pending_txs = pending_count - confirmed_count
        if pending_txs > 0:
            logger.info(f"Found {pending_txs} pending transactions (confirmed: {confirmed_count}, pending: {pending_count})")
        
        return pending_txs
    except Exception as e:
        logger.error(f"Error checking pending transactions: {e}")
        return 0


def allocate_nonces(w3, account_address: str, count: int) -> List[int]:
    """Pre-allocate consecutive nonces for parallel transaction execution."""
    max_retries = 3
    
    for retry in range(max_retries):
        try:
            pending_txs = check_pending_transactions(w3, account_address)
            if pending_txs > 5:
                logger.warning(f"⚠️ {pending_txs} pending transactions detected")
            
            # Get current blockchain nonce and allocate from there
            blockchain_nonce = w3.eth.get_transaction_count(account_address, 'pending')
            nonce_counter_key = f"nonce_counter_{account_address}"
            highest_attempted = nonce_dict.get(nonce_counter_key, blockchain_nonce - 1)
            
            start_nonce = max(blockchain_nonce, highest_attempted + 1)
            allocated_nonces = list(range(start_nonce, start_nonce + count))
            
            # Update global counter
            nonce_dict[nonce_counter_key] = allocated_nonces[-1]
            
            logger.info(f"Allocated {count} nonces: {allocated_nonces[0]}-{allocated_nonces[-1]} (blockchain: {blockchain_nonce})")
            return allocated_nonces
            
        except Exception as e:
            logger.error(f"Nonce allocation failed (attempt {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                time.sleep(0.5 + random.uniform(0, 0.5))
            else:
                raise BlockchainError(f"Failed to allocate nonces: {e}")


def get_next_nonce(w3, account_address: str) -> int:
    """Get a single nonce for immediate use."""
    return allocate_nonces(w3, account_address, 1)[0]


def calculate_gas_price(w3) -> int:
    """Calculate optimal gas price with minimum threshold and buffer."""
    suggested_gas_price = w3.eth.gas_price
    min_gas_price = w3.to_wei(0.1, 'gwei')  # Minimum for Base Sepolia
    
    # Use at least minimum, prefer higher if suggested, add 20% buffer
    gas_price = max(suggested_gas_price, min_gas_price)
    gas_price = int(gas_price * 1.2)
    
    logger.debug(f"Gas price: suggested={suggested_gas_price/10**9:.4f} gwei, using={gas_price/10**9:.4f} gwei")
    return gas_price


def wait_for_transaction_confirmation(w3, tx_hash, timeout: int = 120):
    """
    Robust transaction confirmation with better timeout handling.
    
    Returns receipt or raises BlockchainError with detailed status.
    """
    logger.info(f"Waiting for confirmation: {tx_hash.hex()} (timeout: {timeout}s)")
    
    try:
        # Try to get receipt with timeout
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
        
        if receipt.status == 1:
            logger.info(f"✅ Transaction confirmed: {tx_hash.hex()}")
            return receipt
        else:
            logger.error(f"❌ Transaction reverted: {tx_hash.hex()}")
            logger.error(f"Gas used: {receipt.gasUsed}, Block: {receipt.blockNumber}")
            raise BlockchainError(f"Transaction reverted: {tx_hash.hex()}")
            
    except Exception as e:
        if "timeout" in str(e).lower() or "not in the chain" in str(e).lower():
            logger.warning(f"⏱️ Transaction timeout after {timeout}s, checking status...")
            
            # Check if transaction was actually mined
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                if receipt.status == 1:
                    logger.info(f"✅ Transaction was mined successfully: {tx_hash.hex()}")
                    return receipt
                else:
                    logger.error(f"❌ Transaction was mined but failed: {tx_hash.hex()}")
                    raise BlockchainError(f"Transaction failed: {tx_hash.hex()}")
                    
            except Exception:
                # Transaction not found, check if still pending
                try:
                    tx = w3.eth.get_transaction(tx_hash)
                    logger.warning(f"⏳ Transaction still pending, gas: {tx.gasPrice/10**9:.4f} gwei")
                    raise BlockchainError(f"Transaction timeout: {tx_hash.hex()}")
                except Exception:
                    logger.error(f"❌ Transaction not found: {tx_hash.hex()}")
                    raise BlockchainError(f"Transaction disappeared: {tx_hash.hex()}")
        else:
            raise BlockchainError(f"Transaction confirmation failed: {str(e)}")


def send_blockchain_transaction(
    w3, 
    contract_function, 
    owner_account, 
    nonce: int, 
    default_gas_limit: int,
    operation_name: str
):
    """
    Common transaction sending logic with gas estimation and error handling.
    
    Returns: (tx_hash, receipt)
    """
    try:
        # Estimate gas
        try:
            estimated_gas = contract_function.estimate_gas({
                "from": owner_account.address,
                "nonce": nonce
            })
            gas_limit = int(estimated_gas * 1.2)  # 20% buffer
            logger.debug(f"Estimated gas: {estimated_gas}, using: {gas_limit}")
        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}, using default: {default_gas_limit}")
            gas_limit = default_gas_limit
        
        # Build and sign transaction
        gas_price = calculate_gas_price(w3)
        tx_data = contract_function.build_transaction({
            "chainId": CHAIN_ID,
            "from": owner_account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": gas_price,
        })
        
        signed_tx = w3.eth.account.sign_transaction(tx_data, private_key=PRIVATE_KEY)
        
        # Send transaction
        logger.info(f"Sending {operation_name} transaction (nonce: {nonce}, gas: {gas_limit}, price: {gas_price/10**9:.4f} gwei)")
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        logger.info(f"Transaction sent: {tx_hash.hex()}")
        
        # Wait for confirmation
        receipt = wait_for_transaction_confirmation(w3, tx_hash)
        
        return tx_hash, receipt
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Categorize common errors
        if "already known" in error_msg.lower():
            logger.error(f"❌ {operation_name} failed: Nonce collision (transaction already known)")
        elif "underpriced" in error_msg.lower():
            logger.error(f"❌ {operation_name} failed: Gas price too low")
        elif "insufficient funds" in error_msg.lower():
            logger.error(f"❌ {operation_name} failed: Insufficient funds")
        elif "nonce too low" in error_msg.lower():
            logger.error(f"❌ {operation_name} failed: Nonce too low")
        else:
            logger.error(f"❌ {operation_name} failed: {error_type} - {error_msg}")
        
        raise BlockchainError(f"{operation_name} failed: {error_msg}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((BlockchainError, requests.RequestException)),
    before_sleep=lambda retry_state: logger.info(f"Retrying IPFS upload (attempt {retry_state.attempt_number}/3)...")
)
def upload_to_ipfs(data: str) -> str:
    """Upload data to Pinata and return IPFS URL with retry logic."""
    try:
        if not PINATA_JWT:
            raise BlockchainError("PINATA_JWT not configured")
            
        headers = {"Authorization": f"Bearer {PINATA_JWT}"}
        url = f"{IPFS_BASE_URL}/pinning/pinFileToIPFS"

        # Download and upload file
        if isinstance(data, str) and data.startswith(('http://', 'https://')):
            response = requests.get(data, timeout=60)
            if response.status_code != 200:
                raise BlockchainError(f"Failed to download file: {response.status_code}")
            files = {"file": ("file", response.content)}
        else:
            raise ValueError("Data must be a URL")

        logger.info("Uploading to IPFS...")
        response = requests.post(url, files=files, headers=headers, timeout=60)
        if response.status_code != 200:
            raise BlockchainError(f"IPFS upload failed: {response.status_code}")
        
        ipfs_hash = response.json()["IpfsHash"]
        logger.info(f"Uploaded to IPFS: {ipfs_hash}")
        return f"{IPFS_PREFIX}{ipfs_hash}"
        
    except requests.RequestException as e:
        raise BlockchainError(f"Network error during IPFS upload: {e}")
    except Exception as e:
        raise BlockchainError(f"IPFS upload error: {e}")


def get_contract_data(session_ids: List[str] = None) -> Dict[str, Any]:
    """Get contract data from subgraph, optionally filtered by session IDs."""
    try:
        if not SUBGRAPH_URL:
            raise BlockchainError("SUBGRAPH_URL not configured")

        # Build query based on whether we're filtering by IDs
        if session_ids is None:
            var_decl = ""
            where_arg = ""
            variables = {"firstCreations": 500, "firstMsgs": 200}
        else:
            var_decl = "$ids: [ID!], "
            where_arg = "where: { id_in: $ids }"
            variables = {"firstCreations": 500, "firstMsgs": 200, "ids": session_ids}

        query = f'''
        query Timeline({var_decl}$firstCreations: Int!, $firstMsgs: Int!) {{
            creations(
                first: $firstCreations
                orderBy: lastActivityAt
                orderDirection: desc
                {where_arg}
            ) {{
                id
                firstMessageAt
                lastActivityAt
                closed
                messages(first: $firstMsgs, orderBy: timestamp, orderDirection: asc) {{
                    uuid
                    author
                    content
                    media
                    praiseCount
                    timestamp
                    praises {{
                        praiser
                        timestamp
                    }}
                }}
            }}
        }}'''

        response = requests.post(
            SUBGRAPH_URL,
            json={"query": query, "variables": variables},
            timeout=60,
        )
        
        if response.status_code != 200:
            raise BlockchainError(f"Subgraph request failed: {response.status_code}")

        data = response.json()
        if "errors" in data:
            raise BlockchainError(f"Subgraph query errors: {data['errors']}")

        creations = {
            c["id"]: {
                "messages": c["messages"],
                "closed": c["closed"],
                "firstMessageAt": c["firstMessageAt"],
                "lastActivityAt": c["lastActivityAt"],
            }
            for c in data["data"]["creations"]
        }

        if session_ids:
            missing = [sid for sid in session_ids if sid not in creations]
            if missing:
                raise BlockchainError(f"Session IDs not found: {', '.join(missing)}")
            return {sid: creations[sid] for sid in session_ids}

        return creations
        
    except requests.RequestException as e:
        raise BlockchainError(f"Network error accessing subgraph: {e}")
    except Exception as e:
        raise BlockchainError(f"Error retrieving contract data: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(BlockchainError),
    before_sleep=lambda retry_state: logger.info(f"Retrying create_session_on_chain (attempt {retry_state.attempt_number}/3)...")
)
def create_session_on_chain(
    session_id: str, 
    message_id: str, 
    content: str,
    ipfs_url: str,
    nonce: Optional[int] = None,
) -> Any:
    """Create session on blockchain with retry logic and validation."""
    try:
        logger.info(f"Creating session on chain: {session_id[:8]}...")

        # Setup Web3 and contract
        if not all([BASE_SEPOLIA_RPC, PRIVATE_KEY, CONTRACT_ADDRESS, CONTRACT_ABI]):
            raise BlockchainError("Missing blockchain configuration")

        with open(CONTRACT_ABI, "r") as f:
            contract_abi = json.load(f)

        w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
        if not w3.is_connected():
            raise BlockchainError("Web3 connection failed")

        owner_account = Account.from_key(PRIVATE_KEY)
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)
        
        # Handle nonce - validate if provided, get fresh if needed
        if nonce is None:
            nonce = get_next_nonce(w3, owner_account.address)
        else:
            current_nonce = w3.eth.get_transaction_count(owner_account.address, 'pending')
            if nonce < current_nonce:
                logger.warning(f"Nonce {nonce} stale (current: {current_nonce}), getting fresh nonce")
                nonce = get_next_nonce(w3, owner_account.address)
        
        # Build contract function call
        contract_function = contract.functions.createSession(
            sessionId=str(session_id),
            firstMessageId=str(message_id),
            content=content,
            media=str(ipfs_url)
        )
        
        # Send transaction using common logic
        tx_hash, receipt = send_blockchain_transaction(
            w3, contract_function, owner_account, nonce, 
            GAS_LIMIT_CREATE_SESSION, "CREATE_SESSION"
        )
        
        logger.info(f"✅ Session created on chain: {session_id[:8]}...")
        return receipt
        
    except Exception as e:
        raise BlockchainError(f"Failed to create session on chain: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(BlockchainError),
    before_sleep=lambda retry_state: logger.info(f"Retrying update_session_on_chain (attempt {retry_state.attempt_number}/3)...")
)
def update_session_on_chain(
    session_id: str, 
    message_id: str, 
    ipfs_url: str, 
    content: str, 
    closed: bool = False,
    nonce: Optional[int] = None,
) -> Tuple[Any, Any]:
    """Update session on blockchain with retry logic and validation."""
    try:        
        logger.info(f"Updating session on chain: {session_id[:8]}... (closed: {closed})")
        
        # Setup Web3 and contract
        if not all([BASE_SEPOLIA_RPC, PRIVATE_KEY, CONTRACT_ADDRESS, CONTRACT_ABI]):
            raise BlockchainError("Missing blockchain configuration")
        
        with open(CONTRACT_ABI, "r") as f:
            contract_abi = json.load(f)
        
        w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
        if not w3.is_connected():
            raise BlockchainError("Web3 connection failed")

        owner_account = Account.from_key(PRIVATE_KEY)
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)

        # Handle nonce - validate if provided, get fresh if needed
        if nonce is None:
            nonce = get_next_nonce(w3, owner_account.address)
        else:
            current_nonce = w3.eth.get_transaction_count(owner_account.address, 'pending')
            if nonce < current_nonce:
                logger.warning(f"Nonce {nonce} stale (current: {current_nonce}), getting fresh nonce")
                nonce = get_next_nonce(w3, owner_account.address)

        # Validate parameters
        if not session_id or not message_id:
            raise BlockchainError(f"Invalid parameters: session_id={session_id}, message_id={message_id}")
        
        content = str(content or "")
        ipfs_url = str(ipfs_url or "")

        # Build contract function call
        contract_function = contract.functions.abrahamUpdate(
            str(session_id),
            str(message_id),
            content,
            ipfs_url,
            closed
        )
        
        # Send transaction using common logic
        tx_hash, receipt = send_blockchain_transaction(
            w3, contract_function, owner_account, nonce, 
            GAS_LIMIT_UPDATE_SESSION, "UPDATE_SESSION"
        )

        logger.info(f"✅ Session updated on chain: {session_id[:8]}...")
        return tx_hash, receipt
        
    except Exception as e:
        raise BlockchainError(f"Failed to update session on chain: {e}")


async def get_new_blessings(
    session_id: str, 
    last_abraham_message_id: Optional[str] = None
) -> List[dict]:
    """Get new blessings that came after Abraham's last response."""
    logger.debug(f"Fetching blessings for session {session_id}")
    
    try:
        session_data = get_contract_data([session_id])
        messages = session_data[session_id]["messages"]
        
        if not messages:
            return []
        
        # Sort by timestamp
        messages = sorted(messages, key=lambda x: int(x.get('timestamp', 0)))
        
        # Find Abraham's most recent message
        abraham_last_index = -1
        if last_abraham_message_id:
            for i, msg in enumerate(messages):
                if msg['uuid'] == last_abraham_message_id:
                    abraham_last_index = i
                    break
        
        if abraham_last_index == -1:
            # Fallback: find by author
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('author', '').lower() == ABRAHAM_ADDRESS.lower():
                    abraham_last_index = i
                    break
        
        # Get user messages after Abraham's last message
        new_blessings = []
        start_index = abraham_last_index + 1 if abraham_last_index >= 0 else 0
        
        for msg in messages[start_index:]:
            if msg.get('author', '').lower() != ABRAHAM_ADDRESS.lower():
                new_blessings.append(msg)
        
        logger.debug(f"Found {len(new_blessings)} new blessings")
        return new_blessings
        
    except Exception as e:
        logger.error(f"Error getting new blessings for session {session_id}: {str(e)}")
        return []


def cancel_pending_transactions() -> int:
    """Cancel all pending transactions by sending replacement self-transfers."""
    try:
        w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
        if not w3.is_connected():
            raise BlockchainError("Web3 connection failed")
        
        owner_account = Account.from_key(PRIVATE_KEY)
        confirmed_nonce = w3.eth.get_transaction_count(owner_account.address, 'latest')
        pending_nonce = w3.eth.get_transaction_count(owner_account.address, 'pending')
        
        pending_count = pending_nonce - confirmed_nonce
        if pending_count == 0:
            logger.info("No pending transactions to cancel")
            return 0
        
        logger.warning(f"Canceling {pending_count} pending transactions")
        
        # Use higher gas price for replacement
        current_gas_price = w3.eth.gas_price
        replacement_gas_price = max(int(current_gas_price * 2), w3.to_wei(0.5, 'gwei'))
        
        canceled_count = 0
        for nonce in range(confirmed_nonce, pending_nonce):
            try:
                # Self-transfer with 0 value to cancel
                cancel_tx = {
                    'nonce': nonce,
                    'to': owner_account.address,
                    'value': 0,
                    'gas': 21000,
                    'gasPrice': replacement_gas_price,
                    'chainId': CHAIN_ID
                }
                
                signed_tx = w3.eth.account.sign_transaction(cancel_tx, private_key=PRIVATE_KEY)
                tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                
                logger.info(f"Canceled nonce {nonce}: {tx_hash.hex()}")
                canceled_count += 1
                
            except Exception as e:
                logger.error(f"Failed to cancel nonce {nonce}: {str(e)}")
        
        if canceled_count > 0:
            logger.info("Waiting 10s for cancellations to process...")
            time.sleep(10)
        
        logger.info(f"Canceled {canceled_count}/{pending_count} transactions")
        return canceled_count
        
    except Exception as e:
        logger.error(f"Failed to cancel pending transactions: {e}")
        raise


def reset_nonce_tracking():
    """Reset nonce tracking to start fresh from blockchain state."""
    try:
        w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
        if not w3.is_connected():
            raise BlockchainError("Web3 connection failed")
        
        owner_account = Account.from_key(PRIVATE_KEY)
        blockchain_nonce = w3.eth.get_transaction_count(owner_account.address, 'latest')
        pending_nonce = w3.eth.get_transaction_count(owner_account.address, 'pending')
        
        logger.info(f"Resetting nonce tracking (confirmed: {blockchain_nonce}, pending: {pending_nonce})")
        
        # Clear nonce counter and old reservations
        nonce_counter_key = f"nonce_counter_{owner_account.address}"
        if nonce_counter_key in nonce_dict:
            del nonce_dict[nonce_counter_key]
        
        # Clean up old batch keys
        keys_to_delete = [k for k in nonce_dict.keys() if k.startswith('batch_')]
        for key in keys_to_delete:
            del nonce_dict[key]
        
        logger.info(f"✅ Nonce tracking reset, next nonce: {pending_nonce}")
        return pending_nonce
        
    except Exception as e:
        logger.error(f"Failed to reset nonce tracking: {e}")
        raise


def clear_pending_and_reset(auto_cancel_threshold: int = 5) -> int:
    """Clear pending transactions and reset nonce tracking."""
    try:
        w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
        if not w3.is_connected():
            raise BlockchainError("Web3 connection failed")
        
        owner_account = Account.from_key(PRIVATE_KEY)
        pending_txs = check_pending_transactions(w3, owner_account.address)
        
        canceled_count = 0
        if pending_txs > auto_cancel_threshold:
            logger.warning(f"Auto-canceling {pending_txs} pending transactions")
            canceled_count = cancel_pending_transactions()
        
        reset_nonce_tracking()
        return canceled_count
        
    except Exception as e:
        logger.error(f"Failed to clear pending and reset: {e}")
        raise


def prepare_blockchain_operations(count: int, auto_clear_pending: bool = True) -> List[int]:
    """Pre-allocate nonces for parallel blockchain operations."""
    try:
        w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
        if not w3.is_connected():
            raise BlockchainError("Web3 connection failed")
        
        owner_account = Account.from_key(PRIVATE_KEY)
        pending_txs = check_pending_transactions(w3, owner_account.address)
        
        if auto_clear_pending and pending_txs > 5:
            logger.warning(f"Auto-clearing {pending_txs} pending transactions")
            clear_pending_and_reset()
        elif pending_txs > 10:
            logger.warning(f"Warning: {pending_txs} pending transactions detected")
        
        return allocate_nonces(w3, owner_account.address, count)
        
    except Exception as e:
        raise BlockchainError(f"Failed to prepare blockchain operations: {e}")