import ipfs
import uuid
from datetime import datetime

from tournament import (
    create_session,
    update_session,
    create_session_batch,
    update_session_batch,
)

from auction import (
    set_token,
    #start_genesis_auction,
)


# ---------- Tests ----------
def test_auction():
    # Setup Metadata
    ipfs_image = ipfs.pin("sample.jpg")
    image_hash = ipfs_image.split("/")[-1]
    
    json_data = {
        "description": "Here is a test description",
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

    set_token(0, ipfs_hash)


def test_tournament():
    session_id = str(uuid.uuid4())
    
    # Setup Metadata
    cid = ipfs.pin("sample.jpg")
    
    # Create Session
    create_session(
        session_id=session_id,
        message_id="123", 
        created_at=datetime.now(),
        content="This is a test creation from Abraham's tests",
        media=f"https://gateway.pinata.cloud/ipfs/{cid}"
    )

    # Update Session
    update_session(
        session_id=session_id,
        message_id="345",
        created_at=datetime.now(),
        content="This is a test update from Abraham's tests",
        media=f"https://gateway.pinata.cloud/ipfs/{cid}",
        closed=False
    )

    # Close Session
    update_session(
        session_id=session_id,
        message_id="567",
        created_at=datetime.now(),
        content="Abraham closes the session",
        media="",
        closed=True
    )

    # Test batch create and update
    sessions_to_create = [
        {
            "session_id": str(uuid.uuid4()),
            "message_id": "123",
            "created_at": datetime.now(),
            "content": "This is a test creation from Abraham's tests #1",
            "media": f"https://gateway.pinata.cloud/ipfs/{cid}"
        },
        {
            "session_id": str(uuid.uuid4()),
            "message_id": "345",
            "created_at": datetime.now(),
            "content": "This is a test update from Abraham's tests #2",
            "media": f"https://gateway.pinata.cloud/ipfs/{cid}",
        }
    ]
    
    create_session_batch(sessions_to_create)

    sessions_to_update = [
        {
            "session_id": sessions_to_create[0]["session_id"],
            "message_id": "456",
            "created_at": datetime.now(),
            "content": "This is a test update from Abraham's tests #3. It should be open.",
            "media": f"https://gateway.pinata.cloud/ipfs/{cid}",
            "closed": False
        },
        {
            "session_id": sessions_to_create[1]["session_id"],
            "message_id": "567",
            "created_at": datetime.now(),
            "content": "This is a test update from Abraham's tests #4. It should close.",
            "media": f"https://gateway.pinata.cloud/ipfs/{cid}",
            "closed": True
        }
    ]

    update_session_batch(sessions_to_update)


if __name__ == "__main__":
    test_tournament()
    