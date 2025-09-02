"""

# To run this FastAPI server, use the following uvicorn command:
# uvicorn farcaster1:app --host 0.0.0.0 --port 8000 --reload


"""
from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException, Request, status
import os
import httpx
import json
import hmac
import json
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal
from bson import ObjectId
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse
from jinja2 import Template
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.agent import Agent
from eve.agent.session.session_llm import async_prompt
from eve.agent.session.session_prompts import system_template
from eve.agent.session.models import (
    ChatMessage, ChatMessageRequestInput, LLMConfig, 
    LLMContext, PromptSessionContext, Session
)
from eve.agent.session.session import (
    add_user_message, 
    async_prompt_session, 
    build_llm_context
)
from eve.mongo import Collection, Document
from eve.s3 import upload_file_from_url
from eve.user import User
from eve.tool import Tool
from eve.mongo import MongoDocumentNotFound
from eden import get_current_timestamp
from eve.agent.session.models import UpdateType


user = User.load("gene3")
abraham = Agent.load("abraham")
farcaster_tool = Tool.load("farcaster_cast")


# ---- Config ----
TARGET_FID = int(os.getenv("ABRAHAM_FARCASTER_ID"))
NEYNAR_WEBHOOK_SECRET = os.getenv("NEYNAR_WEBHOOK_SECRET", "")
NEYNAR_API_KEY = os.environ["NEYNAR_API_KEY"]
HDRS = {"accept": "application/json", "api_key": NEYNAR_API_KEY}

# ---- App / Logger ----
app = FastAPI(title="Neynar Webhook Listener")
logger = logging.getLogger("uvicorn.error")

# ---- Helpers ----
def verify_neynar_signature(raw_body: bytes, signature: str) -> bool:
    """
    Neynar signs the raw body with HMAC-SHA512, hex-encoded, in 'X-Neynar-Signature'.
    """
    mac = hmac.new(
        key=NEYNAR_WEBHOOK_SECRET.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha512,
    )
    expected = mac.hexdigest()
    return hmac.compare_digest(expected, signature)


def _extract_embed_urls(embeds: Any) -> List[str]:
    """
    Embeds can be a list of strings (URLs) or objects with a 'url' key.
    """
    urls = []
    urls: List[str] = []
    if isinstance(embeds, list):
        for e in embeds:
            if isinstance(e, str):
                urls.append(e)
            elif isinstance(e, dict):
                u = e.get("url") or e.get("uri") or e.get("href")
                if u:
                    urls.append(u)
    return urls


@Collection("farcaster_events")
class FarcasterEvent(Document):
    cast_hash: str
    event: Dict[str, Any]
    status: Literal["running", "completed", "failed"]
    error: Optional[str] = None
    session_id: Optional[ObjectId] = None
    message_id: Optional[ObjectId] = None
    reply_cast: Optional[Dict[str, Any]] = None
    reply_fid: Optional[int] = None


def upload_to_s3(media_urls: List[str]) -> List[str]:
    uploaded_urls = []
    for media_url in media_urls:
        uploaded_url, _ = upload_file_from_url(media_url)
        uploaded_urls.append(uploaded_url)
    return uploaded_urls


def _split_media(urls: List[str]) -> Dict[str, List[str]]:
    media_exts = (
        ".jpg", ".jpeg", ".png", ".gif", ".webp",
        ".mp4", ".mov", ".webm", ".avi", ".mkv",
        ".mp3", ".wav", ".ogg",
    )
    media, other = [], []
    for u in urls:
        # strip query when checking extension
        cleaned = u.split("?", 1)[0].lower()
        (media if cleaned.endswith(media_exts) or "imagedelivery.net" in u else other).append(u)
    return {"media_urls": media, "other_urls": other}


def normalize_cast_event(evt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Neynar's cast webhook into a compact, convenient shape.
    Based on example payload in Neynar docs for 'cast.created'. 
    """
    cast = evt.get("data") or {}
    author = cast.get("author") or {}
    parent_author = (cast.get("parent_author") or {}) or {}

    # Embeds/media
    embed_urls = _extract_embed_urls(cast.get("embeds"))
    split = _split_media(embed_urls)

    # Mentions (profiles Neynar already resolved for you)
    mentioned_profiles = cast.get("mentioned_profiles") or []
    mentioned_fids = [
        p.get("fid")
        for p in mentioned_profiles
        if isinstance(p, dict) and p.get("fid") is not None
    ]

    normalized: Dict[str, Any] = {
        "event_type": evt.get("type"),            # e.g., "cast.created"
        "event_created_at": evt.get("created_at"),# unix seconds
        "cast": {
            "hash": cast.get("hash"),
            "thread_hash": cast.get("thread_hash"),
            "parent_hash": cast.get("parent_hash"),
            "parent_url": cast.get("parent_url"),
            "root_parent_url": cast.get("root_parent_url"),
            "timestamp": cast.get("timestamp"),   # ISO string from hubs
            "text": cast.get("text"),
            "embeds": embed_urls,
            **split,                               # media_urls, other_urls
            "replies_count": (cast.get("replies") or {}).get("count"),
            "reactions": {
                "likes_count": len((cast.get("reactions") or {}).get("likes") or []),
                "recasts_count": len((cast.get("reactions") or {}).get("recasts") or []),
            },
            "mentioned_fids": mentioned_fids,
            "mentioned_profiles": mentioned_profiles,
        },
        "author": {
            "fid": author.get("fid"),
            "username": author.get("username"),
            "display_name": author.get("display_name"),
            "pfp_url": author.get("pfp_url"),
            "custody_address": author.get("custody_address"),
            # Legacy field: array of eth addresses (older Neynar responses)
            "verifications": author.get("verifications") or [],
            # Newer field: object with eth/sol arrays (+ primary)
            "verified_addresses": author.get("verified_addresses") or None,
            "follower_count": author.get("follower_count"),
            "following_count": author.get("following_count"),
            "active_status": author.get("active_status"),
            "profile": author.get("profile"),
        },
        "parent_author_fid": parent_author.get("fid"),
    }

    # Unify all linked wallets into `author.wallets`
    wallets: List[str] = []
    # custody address
    if normalized["author"].get("custody_address"):
        wallets.append(normalized["author"]["custody_address"])
    # legacy `verifications` (eth addrs)
    for addr in normalized["author"].get("verifications") or []:
        wallets.append(addr)
    # newer `verified_addresses`
    va = normalized["author"].get("verified_addresses")
    if isinstance(va, dict):
        for addr in (va.get("eth_addresses") or []):
            wallets.append(addr)
        for addr in (va.get("sol_addresses") or []):
            wallets.append(addr)  # sol addresses are base58, don't lowercase
        primary = va.get("primary") or {}
        if primary.get("eth_address"):
            wallets.append(primary["eth_address"])
        if primary.get("sol_address"):
            wallets.append(primary["sol_address"])

    # normalize to unique, case-insensitive for hex addresses
    try:
        hexish = {w.lower(): w for w in wallets if w.startswith("0x")}
        non_hex = [w for w in wallets if not w.startswith("0x")]
        unified = sorted(list(hexish.values())) + sorted(set(non_hex))
    except Exception:
        unified = sorted(set(wallets))

    normalized["author"]["wallets"] = unified
    return normalized


async def process_cast(cast: Dict[str, Any]):
    author = cast.get("author") or {}
    author_fid = author.get("fid")
    author_username = author.get("username")
    text = cast.get("text") or ""
    embed_urls = _extract_embed_urls(cast.get("embeds"))
    split = _split_media(embed_urls)
    media_urls = split.get("media_urls") or []
    return author_fid, author_username, text, media_urls


async def fetch_cast_ancestry(cast_hash: str, include_self: bool = True):
    params = {
        "identifier": cast_hash,
        "type": "hash",
        "reply_depth": 0,  # only ancestors, no children
        "include_chronological_parent_casts": "true",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://api.neynar.com/v2/farcaster/cast/conversation",
            headers=HDRS, params=params
        )
    response.raise_for_status()
    data = response.json()

    convo = data.get("conversation", {})
    ancestors = (convo.get("chronological_parent_casts")
                 or convo.get("ancestors")
                 or [])

    if include_self:
        chain = ancestors + [convo["cast"]]
        return chain
    else:
        return ancestors


def is_mention(event: Dict[str, Any], target_fid: int) -> bool:
    is_mention = target_fid in (event["cast"].get("mentioned_fids") or [])
    return is_mention


def is_reply(event: Dict[str, Any], target_fid: int) -> bool:
    is_reply = event.get("parent_author_fid") == target_fid
    return is_reply


async def process_event_pipeline(event: Dict[str, Any]):
    try:
        cast_hash = event["cast"]["hash"]
        event_doc = FarcasterEvent(
            cast_hash=cast_hash,
            event=event,
            status="running",
        )
        event_doc.save()
        session, new_messages = await handle_farcaster(event)
        compact_message = await compact_messages(session, new_messages)

        args = {
            "agent_id": str(abraham.id),
            "text": compact_message.content,
            "embeds": compact_message.media_urls or [],
        }
        parent_hash = event["cast"]["hash"]
        parent_fid  = event["author"]["fid"]
        if parent_hash and parent_fid:
            args.update({"parent_hash": parent_hash, "parent_fid": parent_fid})

        result = await farcaster_tool.async_run(args)
        
        if "output" in result:
            event_doc.update(
                status="completed",
                session_id=session.id,
                message_id=new_messages[0].id,
                reply_cast=result.get("output")[0],
                reply_fid=TARGET_FID,
            )
        else:
            event_doc.update(
                status="failed",
                session_id=session.id,
                message_id=new_messages[0].id,
                error=str(result.get("error")),
            )

    except Exception as e:
        event_doc.update(
            status="failed",
            error=str(e),
        )


async def handle_farcaster(
    event: Dict[str, Any],
) -> Session:
    """Create a session to generate artwork with specified model."""

    cast = event["cast"]
    cast_hash = cast["hash"]
    thread_hash = cast["thread_hash"]
    content = cast["text"] or ""
    author = event["author"]
    author_username = author["username"]
    author_fid = author["fid"]

    # Get or create user
    user = User.from_farcaster(author_fid, author_username)
    
    # get or setup session
    request = PromptSessionRequest(
        user_id=str(user.id),
        actor_agent_ids=[str(abraham.id)],
        message=ChatMessageRequestInput(
            content=content,
            sender_name=author_username,
        ),
    )

    # grab attachment urls
    embed_urls = _extract_embed_urls(cast.get("embeds"))
    split = _split_media(embed_urls)
    media_urls = split.get("media_urls") or []
    media_urls = upload_to_s3(media_urls)
    if media_urls:
        request.message.attachments = media_urls

    session_key = f"farcaster16-{cast_hash}"

    # attempt to get session by session_key
    try:
        session = Session.load(session_key=session_key)
        request.session_id = str(session.id)

    # if no session found, create a new one
    except MongoDocumentNotFound as e:
        background_tasks = BackgroundTasks()
        request.creation_args = SessionCreationArgs(
            owner_id=str(user.id),
            agents=[str(abraham.id)],
            title=f"Farcaster test 55",
            session_key=session_key,
        )
        session = setup_session(
            background_tasks, 
            request.session_id,
            request.user_id,
            request
        )

        # if the cast is not the original, get the previous casts and add them to the session
        if thread_hash != cast_hash:
            prev_casts = await fetch_cast_ancestry(cast_hash, include_self=False)
            for cast in prev_casts:
                author_fid, author_username, text, media_urls = await process_cast(cast)
                media_urls = upload_to_s3(media_urls)
                if author_fid == TARGET_FID:
                    role = "assistant"
                    cast_user = abraham
                else:
                    role = "user"
                    cast_user = User.from_farcaster(author_fid, author_username)
                message = ChatMessage(
                    session=session.id,
                    role=role,
                    content=text,
                    sender=cast_user.id,
                    attachments=media_urls,
                )
                message.save()
                session.messages.append(message.id)
            session.save()


    # Create context with selected model
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=request.message,
        llm_config=LLMConfig(model="claude-sonnet-4-20250514")
    )

    # Add user message to session
    add_user_message(session, context)
    
    # Build LLM context
    context = await build_llm_context(
        session, 
        abraham, 
        context, 
        trace_id=str(uuid.uuid4()), 
    )

    new_messages = []
    
    # Execute the prompt session
    async for update in async_prompt_session(
        session, context, abraham
    ):
        if update.type == UpdateType.ASSISTANT_MESSAGE:
            new_messages.append(update.message)

    return session, new_messages


class CompactMessage(BaseModel):
    """Condense your last messages and any tool results into a single message with resulting media attachments."""
    
    content: Optional[str] = Field(description="Content of the message. Cannot exceed 320 characters.")
    media_urls: Optional[List[str]] = Field(description="URLs of any images or videos to attach")


CAST_TEMPLATE = """# Compact your last messages into a single message

Given your last messages (starting from the message which begins with "{{message_ref}}"), extract from it the following:

media_urls: any successfully produced images, videos, or other media produced by your tool calls
content: a single message, not exceeding 320 characters, which swaps these original messages with a new one which summarizes or restates the original messages as though the whole sequence 

This new message is meant to abstract your original messages -- which may include various thinking, statements of intent (e.g. "Let me ..."), tool calls and results, error handling and retries, or tool sequences -- into a single summarial message in the same tone of your original messages which gives the high level of what you did."""


async def compact_messages(
    session: Any,
    new_messages: List[ChatMessage],
) -> CompactMessage:
    """Extract a compact message from the last assistant messages."""
    
    # Prepare system message
    system_message = system_template.render(
        name=abraham.name,
        current_date_time=get_current_timestamp(),
        description=abraham.description,
        persona=abraham.persona,
        tools=None
    )
    
    # Get session messages
    messages = [ChatMessage.from_mongo(m) for m in session.messages]
    
    # An instruction to compact the messages
    message_ref = new_messages[0].content[:25]
    instruction = Template(CAST_TEMPLATE).render(message_ref=message_ref)

    # Build validation context
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message), 
            *messages,
            ChatMessage(role="system", content=instruction)
        ],
        config=LLMConfig(
            model="claude-sonnet-4-20250514",
            response_format=CompactMessage
        )
    )
    
    # Get compact result
    response = await async_prompt(context)

    result = CompactMessage(**json.loads(response.content)) 

    return result


@app.post("/neynar/webhook")
async def neynar_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_neynar_signature: Optional[str] = Header(None),
):
    # --- Signature check ---
    if not x_neynar_signature:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-Neynar-Signature header")
    raw = await request.body()
    if not verify_neynar_signature(raw, x_neynar_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    # --- Parse raw and build a dedupe key before any heavy work ---
    try:
        evt_raw = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return JSONResponse({"status": "error", "message": "Invalid JSON body"}, status_code=200)

    # If not cast.created, just ack and stop early
    if evt_raw.get("type") != "cast.created":
        return JSONResponse({"status": "skip.type"}, status_code=200)

    # Normalize (cheap) and do quick routing checks
    event = normalize_cast_event(evt_raw)

    parent_fid = event["author"]["fid"]
    if parent_fid == TARGET_FID:
        return JSONResponse({"status": "skip.self"}, status_code=200)

    if not (is_reply(event, TARGET_FID) or is_mention(event, TARGET_FID)):
        return JSONResponse({"status": "skip.not_relevant"}, status_code=200)

    # get idem_key, save to database
    cast_hash = event["cast"]["hash"]

    if FarcasterEvent.find_one({"cast_hash": cast_hash}):
        return JSONResponse({"status": "duplicate_ignored"}, status_code=200)

    # --- Offload heavy work; ACK immediately ---
    # Todo: turn this into a remote job
    background_tasks.add_task(process_event_pipeline, event)
    return JSONResponse({"status": "accepted"}, status_code=200)


@app.get("/healthz")
def healthz():
    return {"ok": True}


async def test():
    from farcaster_examples import reply_to_root, reply_to_inner_abraham_message, reply_to_inner_abraham_message_with_media, root_message    
    event = reply_to_inner_abraham_message_with_media
    
    event = normalize_cast_event(event)
    
    parent_hash = event["cast"]["hash"]
    parent_fid = event["author"]["fid"]

    session, new_messages = await handle_farcaster(event)
    compact_message = await compact_messages(session, new_messages)
    
    args = {
        "agent_id": str(abraham.id),
        "text": compact_message.content,
        "embeds": compact_message.media_urls or [],
    }
    if parent_hash and parent_fid:
        args.update({
            "parent_hash": parent_hash,
            "parent_fid": parent_fid
        })
        
    await farcaster_tool.async_run(args)



# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(test())
