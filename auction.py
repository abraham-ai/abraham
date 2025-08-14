import io
import json
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Assumes you already have:
# - PINATA_JWT
# - IPFS_BASE_URL = "https://api.pinata.cloud"
# - IPFS_PREFIX = "ipfs://"
# - logger
# - BlockchainError (your custom exception)

import chain
from chain import *
import config
import ipfs


from config import (
    logger,
    BASE_SEPOLIA_RPC,
    PRIVATE_KEY,
    CONTRACT_ADDRESS_AUCTION,
    CONTRACT_ABI_AUCTION,
    GAS_LIMIT_UPDATE_SESSION,
)



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(BlockchainError),
    before_sleep=lambda retry_state: logger.info(f"Retrying setup_auction (attempt {retry_state.attempt_number}/3)...")
)
def setup_auction(
    ipfs_hash: str, 
    nonce: Optional[int] = None
) -> Tuple[Any, Any]:
    """Update session on blockchain with retry logic and validation."""
    try:        
        logger.info(f"Setting up auction: {ipfs_hash}...")
        
        # Setup Web3 and contract
        if not all([BASE_SEPOLIA_RPC, PRIVATE_KEY, CONTRACT_ADDRESS_AUCTION, CONTRACT_ABI_AUCTION]):
            raise BlockchainError("Missing blockchain configuration")
        
        with open(CONTRACT_ABI_AUCTION, "r") as f:
            contract_abi = json.load(f)
        
        w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
        if not w3.is_connected():
            raise BlockchainError("Web3 connection failed")

        owner_account = Account.from_key(PRIVATE_KEY)
        contract = w3.eth.contract(address=CONTRACT_ADDRESS_AUCTION, abi=contract_abi)

        # Diagnostics: contract state and permissions
        try:
            contract_owner = contract.functions.owner().call()
            print("CONTRACT OWNER", contract_owner)
            genesis_started = contract.functions.genesisStarted().call()
            print("GENESIS STARTED", genesis_started)
            next_seeded = contract.functions.isNextTokenUriSeeded().call()
            print("NEXT SEEDED", next_seeded)
            current_view = contract.functions.getCurrentAuctionView().call()
            print("CURRENT VIEW", current_view)
            token_id = int(current_view[1])
            print("TOKEN ID", token_id)

        

            logger.info(
                f"owner={contract_owner}, sender={owner_account.address}, genesisStarted={genesis_started}, "
                f"isNextTokenUriSeeded={next_seeded}, currentAuctionId={current_view[0]}, tokenId={token_id}, "
                f"isAuctionActive={current_view[8]}, hasStarted={current_view[9]}, hasEnded={current_view[10]}"
            )
            try:
                owner_of_0 = contract.functions.ownerOf(0).call()
                logger.info(f"Token 0 exists, owner: {owner_of_0}")
            except Exception as e:
                logger.info(f"ownerOf(0) reverted (likely non-existent): {e}")
            if contract_owner.lower() != owner_account.address.lower():
                logger.warning("Caller is not the contract owner; transaction will likely revert.")
        except Exception as e:
            logger.warning(f"Diagnostics failed: {e}")
            token_id = 0

        # Handle nonce - validate if provided, get fresh if needed
        if nonce is None:
            nonce = get_next_nonce(w3, owner_account.address)
        else:
            current_nonce = w3.eth.get_transaction_count(owner_account.address, 'pending')
            if nonce < current_nonce:
                logger.warning(f"Nonce {nonce} stale (current: {current_nonce}), getting fresh nonce")
                nonce = get_next_nonce(w3, owner_account.address)

        logger.info(f"Using tokenId={token_id} for setTokenURI")
        token_id = 2
        # raise Exception("stop")
        # Build contract function call
        print(token_id)
        print(f"ipfs://{ipfs_hash}")
        contract_function = contract.functions.setTokenURI(
            token_id,
            f"ipfs://{ipfs_hash}"
        )

        # Preflight: try static call to capture explicit revert reason/data if any
        try:
            contract_function.call({"from": owner_account.address})
        except Exception as e:
            logger.error(f"Preflight call revert: {e}")
        
        # Send transaction using common logic
        tx_hash, receipt = send_blockchain_transaction(
            w3, contract_function, owner_account, nonce, 
            GAS_LIMIT_UPDATE_SESSION, "UPDATE_SESSION"
        )

        logger.info(f"✅ Auction set on chain: {ipfs_hash}...")
        return tx_hash, receipt
        
    except Exception as e:
        raise BlockchainError(f"Failed to update session on chain: {e}")




def test():
    
    jpg1_path =  "/Users/gene/Downloads/5fe70de27258d5b838132ae72f063611e3661d41aa93a5db545f289103800a38.png"
    jpg1_path = "/Users/gene/Downloads/Eden_creation_geneAbstract-fractal-mathematical-landscape-pure-geometric-forms-M6895045459a243560b66c5bf.png"
    jpg1_path = "/Users/gene/Mars/Mars2/article_2024/Eden_creation_gene3_A-group-of-humans-and-cute-cyborgs-living-vanlife-in-the-desert,_66e7305c058bec36f6c4c306_0.png"

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
    print(ipfs_url)

    ipfs_hash = ipfs_url.split("/")[-1]
    print(ipfs_hash)

    tx_hash, receipt = setup_auction(ipfs_hash)
    print(tx_hash)
    print(receipt)

    # read the receipt
    print(receipt)
    print(receipt.status)
    print(receipt.transactionHash)
    print(receipt.transactionIndex)
    print(receipt.blockHash)
    print(receipt.blockNumber)

    # find errror if there is one
    if receipt.status != 1:
        print(receipt.logs)
        print(receipt.logs[0].args)
        print(receipt.logs[0].args.tokenId)
        print(receipt.logs[0].args.tokenURI)
        print(receipt.logs[0].args.owner)


def start_auction():

    logger.info(f"Genesis start")
    
    # Setup Web3 and contract
    if not all([BASE_SEPOLIA_RPC, PRIVATE_KEY, CONTRACT_ADDRESS_AUCTION, CONTRACT_ABI_AUCTION]):
        raise BlockchainError("Missing blockchain configuration")
    
    with open(CONTRACT_ABI_AUCTION, "r") as f:
        contract_abi = json.load(f)
    
    w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
    if not w3.is_connected():
        raise BlockchainError("Web3 connection failed")

    owner_account = Account.from_key(PRIVATE_KEY)
    contract = w3.eth.contract(address=CONTRACT_ADDRESS_AUCTION, abi=contract_abi)
    
    nonce = get_next_nonce(w3, owner_account.address)
    contract_function = contract.functions.startGenesisAuction()

    # Preflight: try static call to capture explicit revert reason/data if any
    try:
        contract_function.call({"from": owner_account.address})
    except Exception as e:
        logger.error(f"Preflight call revert: {e}")
    
    # Send transaction using common logic
    tx_hash, receipt = send_blockchain_transaction(
        w3, contract_function, owner_account, nonce, 
        GAS_LIMIT_UPDATE_SESSION, "UPDATE_SESSION"
    )

    logger.info(f"✅ Genesis auction started...")
    return tx_hash, receipt

if __name__ == "__main__":
    test()
    # start_auction()
