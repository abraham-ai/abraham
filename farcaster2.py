"""
Single-file FastAPI + Modal wrapper.

Deploy:
    modal deploy farcaster1.py
    modal run farcaster1.py::show_url

Local dev:
    uvicorn farcaster1:fastapi_app --host 0.0.0.0 --port 8000 --reload
"""

from dotenv import load_dotenv
load_dotenv()

# ---- stdlib / typing ----
import os
import json
import hmac
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

# ---- third-party runtime ----
import modal  # needed both for wrapper and for in-webhook spawn()
import httpx
from fastapi import FastAPI, Header, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from bson import ObjectId

# ---- your libs (eve stack) ----
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.agent import Agent
from eve.agent.session.models import (
    ChatMessage, ChatMessageRequestInput, LLMConfig, Channel,
    LLMContext, PromptSessionContext, Session, UpdateType, SessionUpdateConfig
)
from eve.agent.session.session import (
    add_chat_message, 
    async_prompt_session, 
    build_llm_context
)
from eve.mongo import Collection, Document, MongoDocumentNotFound
from eve.s3 import upload_file_from_url
from eve.user import User
from eve.tool import Tool
from eden import get_current_timestamp
from eden_utils.compact_messages import compact_messages, CompactMessage

# ======================================================================================
#                               FastAPI service (ASGI)
# ======================================================================================

# Load long-lived objects (if these do heavy I/O in your local env, you can lazy-load them)
abraham = Agent.load("abraham")
farcaster_tool = Tool.load("farcaster_cast")

# ---- Config ----
TARGET_FID = int(os.getenv("ABRAHAM_FARCASTER_ID", "0"))
NEYNAR_WEBHOOK_SECRET = os.getenv("NEYNAR_WEBHOOK_SECRET", "")
NEYNAR_API_KEY = os.environ["NEYNAR_API_KEY"]
HDRS = {"accept": "application/json", "api_key": NEYNAR_API_KEY}

# ---- App / Logger ----
fastapi_app = FastAPI(title="Neynar Webhook Listener")
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
    event: Optional[Dict[str, Any]] = None
    status: Literal["running", "completed", "failed"]
    error: Optional[str] = None
    session_id: Optional[ObjectId] = None
    message_id: Optional[ObjectId] = None
    reply_cast: Optional[Dict[str, Any]] = None
    reply_fid: Optional[int] = None


# @Collection("abraham_creations")
# class AbrahamCreation(Document):
#     session_id: ObjectId
#     day: str
#     cast_hash: str
#     title: str
#     description: str
#     status: Literal["active", "closed"]


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



def normalize_reaction_event(evt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Neynar's reaction webhook into a compact, convenient shape.
    """

    """
    {
    "created_at": 1757778861,
    "type": "reaction.deleted",
    "data": {
        "object": "reaction",
        "event_timestamp": "2025-09-13T15:54:21.976Z",
        "timestamp": "2025-09-13T15:54:20.000Z",
        "reaction_type": 1,
        "target": {
            "object": "cast_dehydrated",
            "hash": "0x226dec859e82d2e0225a7547650f3bd158673599",
            "author": {
                "object": "user_dehydrated",
                "fid": 884285
            },
            "parent_hash": "0x45d8c9caf3ff485ffaf9174cbfb8300920b89766",
            "parent_url": null,
            "app": {
                "object": "user_dehydrated",
                "fid": 9152
            }
        },
        "user": {
            "object": "user_dehydrated",
            "fid": 360240,
            "username": "genekogan",
            "score": 0.53
        },
        "deprecation_notice": "the `cast` field is deprecated and will be removed in the future. Please use the `target` field instead.",
        "cast": {
            "object": "cast_dehydrated",
            "hash": "0x226dec859e82d2e0225a7547650f3bd158673599",
            "author": {
                "object": "user_dehydrated",
                "fid": 884285
            },
            "parent_hash": "0x45d8c9caf3ff485ffaf9174cbfb8300920b89766",
            "parent_url": null,
            "app": {
                "object": "user_dehydrated",
                "fid": 9152
            }
        }
    }
}
"""


    type_event = evt.get("type")
    reaction = evt.get("data") or {}
    target = reaction.get("target") or {}
    



    return reaction

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
    if normalized["author"].get("custody_address"):
        wallets.append(normalized["author"]["custody_address"])
    for addr in normalized["author"].get("verifications") or []:
        wallets.append(addr)
    va = normalized["author"].get("verified_addresses")
    if isinstance(va, dict):
        for addr in (va.get("eth_addresses") or []):
            wallets.append(addr)
        for addr in (va.get("sol_addresses") or []):
            wallets.append(addr)  # base58, do not lowercase
        primary = va.get("primary") or {}
        if primary.get("eth_address"):
            wallets.append(primary["eth_address"])
        if primary.get("sol_address"):
            wallets.append(primary["sol_address"])

    try:
        hexish = {w.lower(): w for w in wallets if isinstance(w, str) and w.startswith("0x")}
        non_hex = [w for w in wallets if not (isinstance(w, str) and w.startswith("0x"))]
        unified = sorted(list(hexish.values())) + sorted(set(non_hex))
    except Exception:
        unified = sorted(set(wallets))

    normalized["author"]["wallets"] = unified
    return normalized


async def unpack_cast(cast: Dict[str, Any]):
    cast_hash = cast.get("hash")
    author = cast.get("author") or {}
    author_fid = author.get("fid")
    author_username = author.get("username")
    text = cast.get("text") or ""
    embed_urls = _extract_embed_urls(cast.get("embeds"))
    split = _split_media(embed_urls)
    media_urls = split.get("media_urls") or []
    timestamp = cast.get("timestamp") # "timestamp": "2025-08-30T04:55:41.000Z",
    return cast_hash, author_fid, author_username, text, media_urls, timestamp


async def fetch_cast_ancestry(cast_hash: str, include_self: bool = True):
    params = {
        "identifier": cast_hash,
        "type": "hash",
        "reply_depth": 0,
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
    return target_fid in (event["cast"].get("mentioned_fids") or [])


def is_reply(event: Dict[str, Any], target_fid: int) -> bool:
    return event.get("parent_author_fid") == target_fid


async def process_cast(event: Dict[str, Any]):
    event_doc: Optional[FarcasterEvent] = None
    try:
        try:
            cast_hash = event["cast"]["hash"]
        except Exception as e:
            logger.error(f"❌ Error getting cast hash: {str(e)}")
            logger.error(f"❌ Event: {json.dumps(event, indent=4)}")
            return

        event_doc = FarcasterEvent(
            cast_hash=cast_hash,
            event=event,
            status="running",
        )
        event_doc.save()

        session, new_messages = await handle_farcaster(event)

        event_doc.update(
            status="completed",
            session_id=session.id,
            message_id=new_messages[0].id,
            # reply_cast=None, #result.get("output")[0],
            reply_fid=TARGET_FID,
        )
        # else:
        #     event_doc.update(
        #         status="failed",
        #         session_id=session.id,
        #         message_id=new_messages[0].id,
        #         error=str(result.get("error")),
        #     )

    except Exception as e:
        if event_doc is not None:
            event_doc.update(status="failed", error=str(e))
        else:
            logger.exception("Failed before event_doc creation: %s", e)


async def induct_user(user: User, author: Dict[str, Any]):
    # update user metadata
    pfp = author.get("pfp_url")
    if pfp and pfp != user.userImage:
        try:
            pfp_url, _ = upload_file_from_url(pfp)
            user.update(userImage=pfp_url.split("/")[-1])
        except Exception as e:
            logger.error(f"❌ Error uploading pfp {pfp} for user {str(user.id)}: {str(e)}")
            return


async def handle_farcaster(event: Dict[str, Any]) -> Session:
    """Create a session to generate artwork with specified model."""
    cast = event["cast"]
    cast_hash = cast["hash"]
    thread_hash = cast.get("thread_hash")
    parent_hash = cast["parent_hash"]
    content = cast.get("text") or ""
    author = event["author"]
    author_username = author["username"]
    author_fid = author["fid"]

    # Get or create user
    user = User.from_farcaster(author_fid, author_username)
    await induct_user(user, author)

    # get or setup session
    request = PromptSessionRequest(
        user_id=str(user.id),
        actor_agent_ids=[str(abraham.id)],
        message=ChatMessageRequestInput(
            content=content,
            sender_name=author_username,
        ),
    )

    # attachments
    embed_urls = _extract_embed_urls(cast.get("embeds"))
    split = _split_media(embed_urls)
    media_urls = split.get("media_urls") or []
    media_urls = upload_to_s3(media_urls)
    if media_urls:
        request.message.attachments = media_urls

    # who is actually the paying user (insuf manna)
    # session_id='None-68e75a27e96b2dffb8f3f5fd'


    # session_key = f"farcaster17-{cast_hash}"
    if thread_hash:
        session_key = f"FC-{thread_hash}"
    else:
        session_key = f"FC-{cast_hash}"

    print("- -- > session_key", session_key)

    # attempt to get session by session_key
    try:
        session = Session.load(session_key=session_key)
        request.session_id = str(session.id)

    # if no session found, create a new one
    except MongoDocumentNotFound:
        background_tasks = BackgroundTasks()
        request.creation_args = SessionCreationArgs(
            owner_id=str(user.id),
            agents=[str(abraham.id)],
            title=f"Farcaster session",
            session_key=session_key,
        )
        session = setup_session(
            background_tasks, 
            request.session_id,
            request.user_id,
            request
        )

        print("\n\n\n\n\n============")
        print(thread_hash, cast_hash, parent_hash, session.id)
        print("============")


        # if the cast is not the original, get the previous casts and add them to the session
        if thread_hash != cast_hash:
            prev_casts = await fetch_cast_ancestry(cast_hash, include_self=False)
            for pc in prev_casts:
                cast_hash_, author_fid_, author_username_, text_, media_urls_, timestamp_ = await unpack_cast(pc)
                media_urls_ = upload_to_s3(media_urls_)
                created_at = datetime.strptime(timestamp_, "%Y-%m-%dT%H:%M:%S.%fZ")
                if author_fid_ == TARGET_FID:
                    role = "assistant"
                    cast_user = abraham
                else:
                    role = "user"
                    cast_user = User.from_farcaster(author_fid_, author_username_)
                message = ChatMessage(
                    createdAt=created_at,
                    session=session.id,
                    channel=Channel(type="farcaster", key=cast_hash_),
                    role=role,
                    content=text_,
                    sender=cast_user.id,
                    attachments=media_urls_,
                )
                message.save()
    
    # Create context with selected model
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=request.message,
        update_config=SessionUpdateConfig(
            farcaster_hash=cast_hash,
            farcaster_author_fid=author_fid,
        ),
        llm_config=LLMConfig(model="claude-sonnet-4-5"),
        extra_tools={farcaster_tool.name: farcaster_tool}
    )

    # Add user message to session
    await add_chat_message(session, context)

    # Build LLM context
    context = await build_llm_context(
        session, 
        abraham, 
        context, 
        trace_id=str(uuid.uuid4()), 
    )
    new_messages = []
    
    # Execute the prompt session
    async for update in async_prompt_session(session, context, abraham):
        if update.type == UpdateType.ASSISTANT_MESSAGE:
            new_messages.append(update.message)
    return session, new_messages


@fastapi_app.post("/neynar/webhook")
async def neynar_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_neynar_signature: Optional[str] = Header(None),
):
    # --- Validity check ---
    raw = await request.body()

    if not x_neynar_signature:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-Neynar-Signature header")
    if not verify_neynar_signature(raw, x_neynar_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")
    
    try:
        evt_raw = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return JSONResponse({"status": "error", "message": "Invalid JSON body"}, status_code=200)

    # If not cast.created, just ack and stop early
    evt_type = evt_raw.get("type")
    if evt_type not in [
        "cast.created", 
        "reaction.created", 
        "reaction.deleted"
    ]:
        return JSONResponse({"status": "skip.type"}, status_code=200)

    if evt_type in ["reaction.created", "reaction.deleted"]:
        event = normalize_reaction_event(evt_raw)

    else:
        # Normalize and check routes
        event = normalize_cast_event(evt_raw)
        parent_fid = event["author"]["fid"]
        if parent_fid == TARGET_FID:
            return JSONResponse({"status": "skip.self"}, status_code=200)

        if not (is_reply(event, TARGET_FID) or is_mention(event, TARGET_FID)):
            return JSONResponse({"status": "skip.not_relevant"}, status_code=200)

        # de-dupe
        cast_hash = event["cast"]["hash"]
        if FarcasterEvent.find_one({"cast_hash": cast_hash}):
            return JSONResponse({"status": "duplicate_ignored"}, status_code=200)

    # --- Offload heavy work; ACK immediately ---
    if True:
        from modal_app import process_event
        process_event.spawn(event)   # fire-and-forget on Modal infra
    else:
        await process_cast(event)
    
    return JSONResponse({"status": "accepted"}, status_code=200)


@fastapi_app.get("/healthz")
def healthz():
    return {"ok": True}



# evt_raw = {
#     "created_at": 1759983454,
#     "type": "cast.created",
#     "data": {
#         "object": "cast",
#         "hash": "0x4bc7f2dd495ef182e4a558b607b05a5c98caf7ad",
#         "author": {
#             "object": "user",
#             "fid": 360240,
#             "username": "genekogan",
#             "display_name": "Gene Kogan",
#             "pfp_url": "https://imagedelivery.net/BXluQx4ige9GuW0Ia56BHw/36696a0b-7e67-409a-58fb-74e288c6f200/original",
#             "custody_address": "0x364db42663270e039b4b7c68e4f2caef8267a557",
#             "profile": {
#                 "bio": {
#                     "text": "programmer, primate"
#                 }
#             },
#             "follower_count": 35,
#             "following_count": 80,
#             "verifications": [
#                 "0x2eabc4a0bf73dbec9f9b53695cb42333f8f13f8f"
#             ],
#             "verified_addresses": {
#                 "eth_addresses": [
#                     "0x2eabc4a0bf73dbec9f9b53695cb42333f8f13f8f"
#                 ],
#                 "sol_addresses": [
#                     "hGt3T6LdcxRaZrqVbq5bXL7rdSxHae3zGKen3e3hvM3"
#                 ],
#                 "primary": {
#                     "eth_address": "0x2eabc4a0bf73dbec9f9b53695cb42333f8f13f8f",
#                     "sol_address": "hGt3T6LdcxRaZrqVbq5bXL7rdSxHae3zGKen3e3hvM3"
#                 }
#             },
#             "auth_addresses": [
#                 {
#                     "address": "0x2eabc4a0bf73dbec9f9b53695cb42333f8f13f8f",
#                     "app": {
#                         "object": "user_dehydrated",
#                         "fid": 9152
#                     }
#                 }
#             ],
#             "verified_accounts": [
#                 {
#                     "platform": "x",
#                     "username": "genekogan"
#                 }
#             ],
#             "power_badge": False,
#             "experimental": {
#                 "neynar_user_score": 0.42,
#                 "deprecation_notice": "The `neynar_user_score` field under `experimental` will be deprecated after June 1, 2025, as it will be formally promoted to a stable field named `score` within the user object."
#             },
#             "score": 0.42
#         },
#         "app": {
#             "object": "user_dehydrated",
#             "fid": 9152,
#             "username": "warpcast",
#             "display_name": "Warpcast",
#             "pfp_url": "https://i.imgur.com/3d6fFAI.png",
#             "custody_address": "0x02ef790dd7993a35fd847c053eddae940d055596"
#         },
#         "thread_hash": "0xfa9c7c17972b0a2598b83a128d328e5af6c6f640",
#         "parent_hash": "0xfa9c7c17972b0a2598b83a128d328e5af6c6f640",
#         "parent_url": None,
#         "root_parent_url": None,
#         "parent_author": {
#             "fid": 884285
#         },
#         "text": "say that initalian",
#         "timestamp": "2025-10-09T04:17:31.000Z",
#         "embeds": [],
#         "channel": None,
#         "reactions": {
#             "likes_count": 0,
#             "recasts_count": 0,
#             "likes": [],
#             "recasts": []
#         },
#         "replies": {
#             "count": 0
#         },
#         "mentioned_profiles": [],
#         "mentioned_profiles_ranges": [],
#         "mentioned_channels": [],
#         "mentioned_channels_ranges": [],
#         "event_timestamp": "2025-10-09T04:17:34.138Z"
#     }
# }

# async def test_evt_without_server():
#     event = normalize_cast_event(evt_raw)
#     await process_cast(event)


# if __name__ == "__main__":
#     import asyncio
#     print("---> test_evt_without_server 1")
#     asyncio.run(test_evt_without_server())