from datetime import datetime
from bson import ObjectId
import asyncio
import modal
import asyncio
import pytz

from eve.agent.session.models import ChatMessage

import ipfs
import eden
import tournament
import auction

from config import (
    logger,
    DB,
    APP_NAME,
    MODEL_NAME,
    FALLBACK_MODEL_NAME,
    GENERATION_COUNT,
    DESTROY_N_UPDATES,
    ABRAHAM_ADDRESS,
    MAX_PARALLEL_WORKERS,
    TIMEZONE,
    UPDATE_INTERVAL,
    GENESIS_TIME,
)

STATE = modal.Dict.from_name(
    f"{APP_NAME}-state", 
    create_if_missing=True
)


async def init_session(prompt: str) -> dict:
    logger.info(f"🎨 CREATION: {prompt[:80]}...")
    
    session = await eden.create_session(
        prompt, MODEL_NAME, FALLBACK_MODEL_NAME
    )
    result = await eden.validate_creation(
        session, MODEL_NAME, FALLBACK_MODEL_NAME
    )

    if result.error or not result.result_url:
        return {
            "success": False,
            "error": result.error or 'No result URL',
        }
    else:
        return {
            "success": True,
            "created_at": session.createdAt,
            "session_id": str(session.id),
            "message_id": str(session.messages[-1]),
            "media": result.result_url,
            "content": result.announcement,
        }
    

async def genesis(remote=False):
    genesis_creations = await eden.generate_creation_ideas()    
    creation_prompts = genesis_creations.creations[:GENERATION_COUNT]

    logger.info(f"✅ Generated {len(creation_prompts)} creation ideas for genesis")
    for i, idea in enumerate(genesis_creations.creations):
        logger.info(f"  Idea {i}: {idea[:100]}..." if len(idea) > 100 else f"  Idea {i}: {idea}")
    
    if remote:
        sessions = []
        batch_size = MAX_PARALLEL_WORKERS
        for i in range(0, len(creation_prompts), batch_size):
            batch = creation_prompts[i:i+batch_size]
            async for result in init_session_remote.map.aio(batch):
                sessions.append(result)
    else:
        sessions = [await init_session(prompt) for prompt in creation_prompts]

    sessions = [session for session in sessions if session.get('success')]

    try:
        tournament.create_session_batch(sessions)        
        logger.info(f"✅ Batch submitted {len(sessions)} sessions to the tournament")

    except Exception as e:
        logger.error(f"❌ Error submitting batch: {e}. Falling back to single submissions.")
        
        for session in sessions:
            tournament.create_session(
                session_id=session["session_id"],
                message_id=session["message_id"],
                created_at=session["created_at"],
                content=session["content"],
                media=session["media"]
            )

        logger.info(f"✅ Single submissions submitted {len(sessions)} sessions to the tournament")
        

async def update_session(session_data: tuple) -> dict:
    """Process updates for a single creation - check for blessings and respond if found."""

    session_id, session = session_data

    # Get new blessings for this session
    last_message_id = session.get('last_processed_message_id')
    new_blessings = await tournament.get_new_blessings(session_id, last_message_id)

    if not new_blessings:
        logger.info(f"📭 UPDATE {session_id}...: No new blessings found")
        return {
            "success": True,
            "action": "no_update",
            "session_id": session_id,
            "blessing_count": 0
        }
    
    # Found new blessings - process them
    logger.info(f"🙏 Session {session_id} -- {len(new_blessings)} new blessings")
    for i, blessing in enumerate(new_blessings, 1):
        logger.info(f"✨ Blessing {i} by {blessing['author']} : {blessing['cid']}")
        
    # create new message in response to blessings
    blessing_result, user_message = await eden.process_blessings_iteration(
        session_id, 
        new_blessings,
        MODEL_NAME,
        FALLBACK_MODEL_NAME
    )

    if blessing_result.error:
        logger.error(f"❌ UPDATE {session_id}...: Blessing processing failed: {blessing_result.error}")
        return
    
    logger.info(f"🖼️ UPDATE {session_id}...: Creation URL: {blessing_result.result_url}")
    
    tournament.update_session(
        session_id=session_id, 
        message_id=str(user_message.id), 
        created_at=user_message.createdAt,
        content=blessing_result.announcement,
        media=blessing_result.result_url, 
        closed=False
    )
    

async def update(remote=False):
    """Check all active creations for new blessings and respond if found."""

    today = datetime.now(pytz.timezone('US/Eastern'))    
    active_sessions = tournament.get_tournament_data(date_filter=today, closed_filter=False)
    session_ids = [str(s) for s in active_sessions.keys()]

    logger.info(f"✓ Active sessions: {', '.join(session_ids)}")

    if not active_sessions:
        logger.warning("⚠️ No active sessions found - nothing to update")
        return {
            "success": True,
            "total_sessions": 0,
            "updated_sessions": 0,
            "error_sessions": 0
        }
    
    if remote:
        session_data = list(zip(session_ids, active_sessions.values()))
        async for _ in update_session_remote.map.aio(session_data):
            pass
    else:
        for session_id, session in active_sessions.items():
            await update_session((session_id, session))


async def close():
    """Destroy all active sessions."""

    today = datetime.now(pytz.timezone('US/Eastern'))
    active_sessions = tournament.get_tournament_data(date_filter=today, closed_filter=False)
    
    if len(active_sessions) <= 1:
        logger.warning(f"⚠️ Only {len(active_sessions)} active session(s) found - nothing to close")
        if len(active_sessions) == 1:
            logger.info("✅ Tournament finished")

    # Sort by praise score (lowest first), with oldest winning ties
    session_scores = []
    for session_id, session_data in active_sessions.items():
        messages = session_data.get('messages', [])
        blessings = [msg for msg in messages if msg.get('author', '').lower() != ABRAHAM_ADDRESS.lower()]
        total_praises = sum(msg.get('praiseCount', 0) for msg in messages)
        total_blessings = len(blessings)
        praise_score = total_praises + (3 * total_blessings)
        session_scores.append({
            'session_id': session_id,
            'creation_data': session_data,
            'praise_score': praise_score,
            'total_praises': total_praises,
            'total_blessings': total_blessings,
            'start_time': session_data.get('firstMessageAt'),
            'creation_prompt': session_data.get('creation_prompt', '')[:80]
        })

    # Sort by praise_score ascending, then by start_time ascending (oldest first for ties)
    session_scores.sort(key=lambda x: (x['praise_score'], x['start_time']))
    
    # Remove bottom half
    total_sessions = len(session_scores)
    sessions_to_remove = total_sessions // 2
    sessions_to_close = session_scores[:sessions_to_remove]
    sessions_to_keep = session_scores[sessions_to_remove:]
    
    # Add concluding remarks to sessions being closed
    logger.info(f"💀 Addiing concluding remarks to {len(sessions_to_close)} sessions")

    sessions_closed, sessions_close_errors = 0, 0
    for session_info in sessions_to_close:
        session_id = session_info['session_id']
        try:
            assistant_message = await eden.close_session(
                session_id=session_id, 
                total_praises=session_info['total_praises'], 
                total_blessings=session_info['total_blessings']
            )

        # if error, create a fallback message
        except Exception as e:
            logger.error(f"❌ Error adding concluding remark for {session_id}: {str(e)}")
            assistant_message = ChatMessage(
                id=ObjectId(),
                role="assistant",
                content="Closed by Abraham without comment",
            )

        try:
            tournament.update_session(
                session_id=str(session_id),
                message_id=str(assistant_message.id),
                created_at=assistant_message.createdAt,
                media="",
                content=assistant_message.content,
                closed=True
            )    
            sessions_closed += 1

        except Exception as e:
            logger.error(f"❌ Error closing {session_id}: {str(e)}")
            sessions_close_errors += 1
            continue
    
    sessions_survived = len(sessions_to_keep) + sessions_close_errors

    # Check if tournament should finish
    if sessions_survived <= 1:
        logger.info("🏆 Tournament finished after close cycle!")

        if sessions_survived == 1:
            logger.info("🏆 Tournament done... let's close it all")
            # result = finish_tournament.remote()
            # logger.info(f"🏆 Tournament finished: {result}")
    
    logger.info("✅ Destroy cycle complete")


async def finish(winner_id):
    """Finish the tournament."""

    logger.info(f"🏆 Winner: {winner_id}")

    video_result = await eden.create_video(winner_id, MODEL_NAME, FALLBACK_MODEL_NAME)
    
    poster_image_url = ipfs.pin(video_result.poster_image_url)
    poster_image_hash = poster_image_url.split("/")[-1]

    video_url = ipfs.pin(video_result.video_url)
    video_hash = video_url.split("/")[-1]

    json_data = {
        "description": video_result.writeup,
        "external_url": "https://abraham.ai",
        "image": f"ipfs://{poster_image_hash}",
        "video": f"ipfs://{video_hash}",
        "name": video_result.title,
        "attributes": []
    }

    ipfs_url = ipfs.pin(json_data)
    ipfs_hash = ipfs_url.split("/")[-1]

    auction.set_token(ipfs_hash)
    # auction.start_genesis_auction()


async def start(remote=False, force=False):
    """Run Genesis to begin tourname."""

    if force:
        STATE["state"] = "off"
        tournament.close_all_open_sessions()

    if STATE.get("state", "off") != "off":
        logger.info("🏆 Tournament already running, skip...")
        return

    STATE["state"] = "genesis"
    STATE["update_count"] = 0
    
    try:
        if remote:  
            genesis_remote.remote()
        else:
            await genesis()
    except Exception as e:
        logger.error(f"❌ Error in genesis: {str(e)}")

    STATE["state"] = "waiting"


async def heartbeat(remote=False):
    """Periodic heartbeat to check state and run tournament updates."""

    state = STATE.get("state", "off")

    if state == "off":
        logger.info("🏆 Tournament not active, skip...")
        return
    
    elif state in ["genesis", "running", "closing", "finishing"]:
        logger.info("🏆 Tournament is already updating, skip turn...")
        return
    
    elif state == "waiting":
        logger.info("🏆 Tournament is ready for next update.")
        
        STATE["state"] = "running"
        
        try:
            if remote:
                update_remote.remote()
            else:
                await update()
        except Exception as e:
            logger.error(f"❌ Error in update: {str(e)}")
        
        STATE["update_count"] += 1
        
        logger.info(f"🏆 Done update #{STATE['update_count']}.")
        
        should_close = STATE["update_count"] % DESTROY_N_UPDATES == 0
        if should_close:
            STATE["state"] = "closing"
            try:
                if remote:
                    close_remote.remote()
                else:
                    await close()
            except Exception as e:
                logger.error(f"❌ Error in close: {str(e)}")
            
        STATE["state"] = "waiting"

        active_sessions = tournament.get_tournament_data(
            date_filter=datetime.now(pytz.timezone('US/Eastern')), 
            closed_filter=False
        )
        should_finish = len(active_sessions) == 1

        if should_finish:
            STATE["state"] = "finishing"            
            try:
                winner_id = list(active_sessions.keys())[0]
                if remote:
                    finish_remote.remote(winner_id)
                else:
                    await finish(winner_id)
            except Exception as e:
                logger.error(f"❌ Error in finish: {str(e)}")
            STATE["state"] = "off"


async def test_tournament():
    """
    Run a test of the whole tournament.
    """
    await start(force=True)
    await asyncio.sleep(60)
    for _ in range(2):
        await heartbeat()
        await asyncio.sleep(60)
    

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": DB})
    .apt_install("git", "libmagic1", "ffmpeg", "wget")
    .pip_install("web3", "eth-account", "requests", "jinja2", "python-dotenv", "pytz", "tenacity")
    .run_commands(
        "git clone https://github.com/edenartlab/eve.git /root/eve-repo",
        "cd /root/eve-repo && git checkout staging && pip install -e .",
    )
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("eden.py", "/root/eden.py")
    .add_local_file("chain.py", "/root/chain.py")
    .add_local_file("ipfs.py", "/root/ipfs.py")
    .add_local_file("tournament.py", "/root/tournament.py")
    .add_local_file("auction.py", "/root/auction.py")
    .add_local_file("contract_abi_tournament.json", "/root/contract_abi_tournament.json")
    .add_local_file("contract_abi_auction.json", "/root/contract_abi_auction.json")
)


app = modal.App(
    APP_NAME,
    image=image,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{DB}"),
        modal.Secret.from_name("abraham-secrets"),
    ]
)

@app.function(image=image, timeout=60 * 60, max_containers=MAX_PARALLEL_WORKERS)
async def init_session_remote(prompt: str) -> dict:
    return await init_session(prompt)

@app.function(image=image, timeout=60 * 60, max_containers=MAX_PARALLEL_WORKERS)
async def update_session_remote(session_data: tuple) -> dict:
    return await update_session(session_data)

@app.function(image=image, timeout=60 * 60)
async def finish_remote(winner_id: str):
    return await finish(winner_id)

@app.function(image=image, timeout=60 * 60)
async def genesis_remote():
    return await genesis()

@app.function(image=image, timeout=60 * 60)
async def close_remote():
    return await close()

@app.function(image=image, timeout=60 * 60)
async def update_remote():
    return await update()

_gh, _gm = GENESIS_TIME.split(':')
@app.function(image=image, timeout=60 * 60, schedule=modal.Cron(f"{_gm} {_gh} * * *", timezone=TIMEZONE))
async def start_remote():
    return await start(remote=True, force=True)

@app.function(image=image, timeout=60 * 60, schedule=modal.Period(minutes=UPDATE_INTERVAL))
async def heartbeat_remote():
    return await heartbeat(remote=True)

@app.local_entrypoint()
def local():
    asyncio.run(update())

if __name__ == "__main__":
    asyncio.run(local())