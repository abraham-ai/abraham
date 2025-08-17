import requests
import pytz
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

import ipfs

from chain import (
    safe_send,
    BlockchainError,
    load_contract,
)

from config import (
    logger, 
    CONTRACT_ADDRESS_TOURNAMENT,
    CONTRACT_ABI_TOURNAMENT,
    SUBGRAPH_URL,
    ABRAHAM_ADDRESS,
    IPFS_PREFIX,
)


def is_server_error(exception):
    """Check if the exception is a server-side error that should be retried."""
    if isinstance(exception, requests.RequestException):
        # Network errors (timeouts, connection errors)
        if isinstance(exception, (requests.Timeout, requests.ConnectionError)):
            return True
        # HTTP 5xx errors
        if hasattr(exception, 'response') and exception.response is not None:
            return 500 <= exception.response.status_code < 600
    return False

@retry(
    retry=retry_if_exception(is_server_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=lambda retry_state: logger.warning(f"Retrying get_tournament_data due to server error (attempt {retry_state.attempt_number})")
)
def get_tournament_data(
    session_ids: Optional[List[str]] = None, 
    date_filter: Optional[datetime] = None,
    closed_filter: Optional[bool] = None
) -> Dict[str, Any]:
    """Get tournament contract data from subgraph, optionally filtered by session IDs, date, and closed status.
    
    Args:
        session_ids: Optional list of session IDs to filter by
        date_filter: Optional datetime object to filter by date (sessions from that day)
        closed_filter: Optional boolean to filter by closed status (True=closed, False=open)
    """
    try:
        # Handle date filtering
        date_where_clause = ""
        if date_filter:            
            start_timestamp = int(datetime(date_filter.year, date_filter.month, date_filter.day).timestamp())
            end_timestamp = start_timestamp + 86400  # Add 24 hours            
            date_where_clause = f"firstMessageAt_gte: {start_timestamp}, firstMessageAt_lt: {end_timestamp}"

        # Handle closed filtering
        closed_where_clause = ""
        if closed_filter is not None:
            closed_where_clause = f"closed: {str(closed_filter).lower()}"

        # Build where clause by combining all filters
        where_conditions = []
        if session_ids is not None:
            where_conditions.append("id_in: $ids")
        if date_where_clause:
            where_conditions.append(date_where_clause)
        if closed_where_clause:
            where_conditions.append(closed_where_clause)
        
        # Build query variables and where clause
        if session_ids is not None:
            var_decl = "$ids: [ID!], "
            variables = {"firstCreations": 500, "firstMsgs": 200, "ids": session_ids}
        else:
            var_decl = ""
            variables = {"firstCreations": 500, "firstMsgs": 200}
        
        if where_conditions:
            where_arg = f"where: {{ {', '.join(where_conditions)} }}"
        else:
            where_arg = ""

        query = f'''
        query Timeline({var_decl}$firstCreations: Int!, $firstMsgs: Int!) {{
            creations(
                first: $firstCreations
                orderBy: lastActivityAt
                orderDirection: desc
                {where_arg}
            ) {{
                id
                closed
                ethSpent
                firstMessageAt
                lastActivityAt
                messages(first: $firstMsgs, orderBy: timestamp, orderDirection: asc) {{
                    uuid
                    author
                    cid
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
            timeout=60
        )
        
        if response.status_code != 200:
            # For 5xx errors, raise requests.HTTPError to trigger retry
            if 500 <= response.status_code < 600:
                response.raise_for_status()
            # For other errors (4xx, etc), don't retry
            raise Exception(f"Subgraph request failed: {response.status_code}")

        data = response.json()
        if "errors" in data:
            raise Exception(f"Subgraph query errors: {data['errors']}")

        creations = {
            c["id"]: {
                "messages": c["messages"],
                "closed": c["closed"],
                "firstMessageAt": c["firstMessageAt"], 
                "lastActivityAt": c["lastActivityAt"]
            }
            for c in data["data"]["creations"]
        }

        if session_ids:
            missing = [sid for sid in session_ids if sid not in creations]
            if missing:
                raise Exception(f"Session IDs not found: {', '.join(missing)}")
            return {sid: creations[sid] for sid in session_ids}

        return creations
        
    except requests.RequestException as e:
        logger.error(f"Network error accessing subgraph: {e}")
        raise Exception(f"Network error accessing subgraph: {e}")

    except Exception as e:
        logger.error(f"Error retrieving contract data: {e}")
        raise Exception(f"Error retrieving contract data: {e}")



async def get_new_blessings(
    session_id: str, 
    last_abraham_message_id: Optional[str] = None
) -> List[dict]:
    """Get new blessings that came after Abraham's last response."""
    logger.debug(f"Fetching blessings for session {session_id}")
    
    try:
        session_data = get_tournament_data(session_ids=[session_id])
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
                ipfs_url = f"{IPFS_PREFIX}{msg['cid']}"
                response = requests.get(ipfs_url)
                json_data = response.json()
                msg["content"] = json_data["content"]
                new_blessings.append(msg)
        
        logger.debug(f"Found {len(new_blessings)} new blessings")
        return new_blessings
        
    except Exception as e:
        logger.error(f"Error getting new blessings for session {session_id}: {str(e)}")
        return []


def create_session(
    session_id: str,
    message_id: str,
    created_at: datetime,
    content: str,
    media: str
):
    """
    Create a new session.

    Args:
        session - dictionary with the following keys:
            - session_id - unique id for the session
            - message_id - message id of abraham's first message
            - content - message content
            - media - optional ipfs url
    """
    try:
        w3, owner, contract = load_contract(
            address=CONTRACT_ADDRESS_TOURNAMENT,
            abi_path=CONTRACT_ABI_TOURNAMENT
        )

        json_data = {
            "sessionId": session_id,
            "messageId": message_id,
            "createdAt": int(created_at.timestamp()),
            "author": ABRAHAM_ADDRESS,
            "kind": "owner",
            "content": content
        }

        if media:
            media_cid = ipfs.pin(media)
            json_data["media"] = [{
                "type": "image",
                "mime": "image/webp",
                "src": f"https://gateway.pinata.cloud/ipfs/{media_cid}"
            }]

        cid = ipfs.pin(json_data)

        contract_function = contract.functions.createSession(
            sessionId=session_id,
            firstMessageId=message_id,
            cid=cid
        )

        safe_send(
            w3,
            contract_function,
            owner,
            op_name="CREATE_SESSION",
            nonce=None,               # or set an explicit nonce to pin
            value=0,                  # non-payable
        )

    except BlockchainError as e:
        logger.error(f"❌ CREATE_SESSION failed: {e}")
        raise


def create_session_batch(
    sessions: List[Dict[str, Any]]
):
    """
    Create multiple new sessions.

    Args:
        sessions - list of session data. Each session is a dictionary with the following keys:
            - session_id - unique id for the session
            - message_id - message id of abraham's first message
            - content - message content
            - media - optional ipfs url
    """
    try:
        w3, owner, contract = load_contract(
            address=CONTRACT_ADDRESS_TOURNAMENT,
            abi_path=CONTRACT_ABI_TOURNAMENT
        )

        for session in sessions:
            json_data = {
                "sessionId": session["session_id"],
                "messageId": session["message_id"],
                "createdAt": int(session["created_at"].timestamp()),
                "author": ABRAHAM_ADDRESS,
                "kind": "owner",
                "content": session["content"]
            }
            if session.get("media"):
                media_cid = ipfs.pin(session["media"])
                json_data["media"] = [{
                    "type": "image",
                    "mime": "image/webp",
                    "src": f"https://gateway.pinata.cloud/ipfs/{media_cid}"
                }]
            session["cid"] = ipfs.pin(json_data)

        session_data = [
            {
                "sessionId": session["session_id"],
                "firstMessageId": session["message_id"],
                "cid": session["cid"]
            }
            for session in sessions
        ]

        contract_function = contract.functions.abrahamBatchCreate(
            session_data
        )

        safe_send(
            w3,
            contract_function,
            owner,
            op_name="ABRAHAM_BATCH_CREATE",
            nonce=None,               # or set an explicit nonce to pin
            value=0,                  # non-payable
        )

    except BlockchainError as e:
        logger.error(f"❌ ABRAHAM_BATCH_CREATE failed: {e}")
        raise


def update_session(
    session_id: str,
    message_id: str,
    created_at: datetime,
    content: str,
    media: str,
    closed: bool
):
    """
    Update a session with a new message.

    Args:
        session_id
        message_id - message id of abraham's last message
        content - message content
        media - optional ipfs url
        closed (bool) - to close the session
    """
    try:
        w3, owner, contract = load_contract(
            address=CONTRACT_ADDRESS_TOURNAMENT,
            abi_path=CONTRACT_ABI_TOURNAMENT
        )

        json_data = {
            "sessionId": session_id,
            "messageId": message_id,
            "createdAt": int(created_at.timestamp()),
            "author": ABRAHAM_ADDRESS,
            "kind": "owner",
            "content": content,
        }

        if media:
            media_cid = ipfs.pin(media)
            json_data["media"] = [{
                "type": "image",
                "mime": "image/webp",
                "src": f"https://gateway.pinata.cloud/ipfs/{media_cid}"
            }]

        cid = ipfs.pin(json_data)

        contract_function = contract.functions.abrahamUpdate(
            sessionId=session_id,
            messageId=message_id, 
            cid=cid,
            closed=closed
        )

        safe_send(
            w3,
            contract_function,
            owner,
            op_name="ABRAHAM_UPDATE",
            nonce=None,               # or set an explicit nonce to pin
            value=0,                  # non-payable
        )

    except BlockchainError as e:
        logger.error(f"❌ ABRAHAM_UPDATE failed: {e}")
        raise


def update_session_batch(
    sessions: List[Dict[str, Any]]
):
    """
    Update multiple sessions with new messages.

    Args:
        sessions - list of session data. Each session is a dictionary with the following keys:
            - session_id - unique id for the session
            - message_id - message id of abraham's last message
            - content - message content
            - media - optional ipfs url
            - closed (bool) - to close the session
    """
    try:
        w3, owner, contract = load_contract(
            address=CONTRACT_ADDRESS_TOURNAMENT,
            abi_path=CONTRACT_ABI_TOURNAMENT
        )

        for session in sessions:
            json_data = {
                "sessionId": session["session_id"],
                "messageId": session["message_id"],
                "createdAt": int(session["created_at"].timestamp()),
                "author": ABRAHAM_ADDRESS,
                "kind": "owner",
                "content": session["content"]
            }
            if session.get("result_url"):
                media_cid = ipfs.pin(session["result_url"])
                json_data["media"] = [{
                    "type": "image",
                    "mime": "image/webp",
                    "src": f"https://gateway.pinata.cloud/ipfs/{media_cid}"
                }]
            session["cid"] = ipfs.pin(json_data)

        session_data = [
            {
                "sessionId": session["session_id"],
                "messageId": session["message_id"],
                "cid": session["cid"],
                "closed": session["closed"]
            }
            for session in sessions
        ]

        contract_function = contract.functions.abrahamBatchUpdateAcrossSessions(
            session_data
        )

        safe_send(
            w3,
            contract_function,
            owner,
            op_name="ABRAHAM_UPDATE",
            nonce=None,               # or set an explicit nonce to pin
            value=0,                  # non-payable
        )

    except BlockchainError as e:
        logger.error(f"❌ ABRAHAM_UPDATE failed: {e}")
        raise


def close_all_open_sessions():
    """Close open sessions from the last day"""

    today = datetime.now(pytz.timezone('US/Eastern'))
    active_sessions = get_tournament_data(date_filter=today, closed_filter=False)

    if not active_sessions:
        logger.info("✅ No sessions found to close")
        return

    update_session_batch([
        {
            "session_id": session_id,
            "message_id": "999999",
            "created_at": today,
            "content": "Closed by Abraham!",
            "media": "",
            "closed": True
        } 
        for session_id in active_sessions.keys()
    ])    

    logger.info(f"Closed {len(active_sessions)} active sessions")
    
