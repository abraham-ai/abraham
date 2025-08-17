from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Optional

from chain import (
    safe_send,
    BlockchainError,
    load_contract,
)

from config import (
    logger,
    CONTRACT_ADDRESS_AUCTION,
    CONTRACT_ABI_AUCTION,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(BlockchainError),
    before_sleep=lambda retry_state: logger.info(f"Retrying set_token (attempt {retry_state.attempt_number}/3)...")
)
def set_token(
    ipfs_hash: str,
    token_id: Optional[int] = None, 
):
    """Update session on blockchain with retry logic and validation."""
    try:
        w3, owner, contract = load_contract(
            CONTRACT_ADDRESS_AUCTION,
            CONTRACT_ABI_AUCTION
        )

        if token_id is None:
            current = contract.functions.getCurrentAuction().call()
            token_id = current[0]

        logger.info(f"Setting up token {token_id}...")

        contract_function = contract.functions.setTokenURI(
            token_id,
            f"ipfs://{ipfs_hash}"
        )

        safe_send(w3, contract_function, owner, op_name="SET_TOKEN_URI")

    except Exception as e:
        logger.error(f"❌ Error in set_token: {str(e)}")
        raise


def start_genesis_auction():
    """Start the genesis auction."""
    try:        
        logger.info(f"Starting genesis auction...")

        w3, owner, contract = load_contract(
            CONTRACT_ADDRESS_AUCTION,
            CONTRACT_ABI_AUCTION
        )

        contract_function = contract.functions.startGenesisAuction()
        safe_send(w3, contract_function, owner, op_name="START_GENESIS_AUCTION")

    except Exception as e:
        logger.error(f"❌ Error in set_token: {str(e)}")
        raise
