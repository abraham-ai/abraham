import ipfs
import json
from eth_account import Account
from web3 import Web3

from chain import (
    safe_send,
    BlockchainError,
    make_w3,
)

from tournament import (
    create_session,
    update_session,
    create_session_batch,
    update_session_batch,
)

from config import (
    logger,
    PRIVATE_KEY,
    CONTRACT_ABI_AUCTION,
    CONTRACT_ADDRESS_AUCTION,
    CONTRACT_ABI_TOURNAMENT,
    CONTRACT_ADDRESS_TOURNAMENT,
)


# ---------- Tests ----------
def test_auction():

    # Setup Metadata
    ipfs_image = ipfs.pin("sample.jpg")
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
            }
        ]
    }

    ipfs_url = ipfs.pin(json_data)
    ipfs_hash = ipfs_url.split("/")[-1]

    # Load ABI
    with open(CONTRACT_ABI_AUCTION, "r") as f:
        contract_abi = json.load(f)

    w3 = make_w3()
    owner = Account.from_key(PRIVATE_KEY)

    contract = w3.eth.contract(address=CONTRACT_ADDRESS_AUCTION, abi=contract_abi)

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


def test_tournament():
    import uuid
    session_id = str(uuid.uuid4())
    
    # Setup Metadata
    ipfs_image = ipfs.pin("sample.jpg")
    image_hash = ipfs_image.split("/")[-1]
    
    # Create Session
    create_session(
        session_id=session_id,
        first_message_id="123", 
        content="This is a test creation from Abraham's tests",
        media=f"https://gateway.pinata.cloud/ipfs/{image_hash}"
    )

    # Update Session
    update_session(
        session_id=session_id,
        message_id="345",
        content="This is a test update from Abraham's tests",
        media=f"https://gateway.pinata.cloud/ipfs/{image_hash}",
        closed=False
    )

    # Close Session
    update_session(
        session_id=session_id,
        message_id="567",
        content="Abraham closes the session",
        media="",
        closed=True
    )


    # Test batch create and update
    sessions_to_create = [
        {
            "session_id": str(uuid.uuid4()),
            "message_id": "123",
            "content": "This is a test creation from Abraham's tests #1",
            "media": f"https://gateway.pinata.cloud/ipfs/{image_hash}"
        },
        {
            "session_id": str(uuid.uuid4()),
            "message_id": "345",
            "content": "This is a test update from Abraham's tests #2",
            "media": f"https://gateway.pinata.cloud/ipfs/{image_hash}",
        }
    ]

    create_session_batch(sessions_to_create)

    sessions_to_update = [
        {
            "session_id": sessions_to_create[0]["session_id"],
            "message_id": "456",
            "content": "This is a test update from Abraham's tests #3. It should be open.",
            "media": f"https://gateway.pinata.cloud/ipfs/{image_hash}",
            "closed": False
        },
        {
            "session_id": sessions_to_create[1]["session_id"],
            "message_id": "567",
            "content": "This is a test update from Abraham's tests #4. It should close.",
            "media": f"https://gateway.pinata.cloud/ipfs/{image_hash}",
            "closed": True
        }
    ]

    update_session_batch(sessions_to_update)


if __name__ == "__main__":
    test_tournament()
    

