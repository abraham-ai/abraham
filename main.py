from datetime import datetime, timezone
from bson import ObjectId
from typing import Dict
import asyncio
import modal
import asyncio
import random
import pytz

from eve.agent.session.models import ChatMessage
import chain
import config
import ipfs


# Container image definition
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": config.DB})
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
    .add_local_file("contract_abi_tournament.json", "/root/contract_abi_tournament.json")
)

with image.imports():
    from config import (
        log_section,
        logger,
        MODEL_NAME,
        GENERATION_COUNT,
        MAX_PARALLEL_WORKERS,
        FALLBACK_MODEL_NAME,
        DESTROY_N_UPDATES,
        GENESIS_TIME,
        TIMEZONE,
    )
    import eden
    import chain


# Modal app setup
app = modal.App(
    config.APP_NAME,
    image=image,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{config.DB}"),
        modal.Secret.from_name("abraham-secrets"),
    ]
)

STATE = modal.Dict.from_name(
    f"{config.APP_NAME}-state", 
    create_if_missing=True
)


@app.function(
    image=image, 
    timeout=60 * 60, 
    max_containers=MAX_PARALLEL_WORKERS
)
async def init_creation(prompt: str) -> dict:
    """Setup Creation Session"""
    
    logger.info(f"🎨 CREATION: {prompt[:80]}...")
    
    for attempt in range(3):
        model_name = MODEL_NAME if attempt < 2 else FALLBACK_MODEL_NAME
        logger.info(f"Attempt {attempt + 1}/5 - Model: {model_name}")
        error_message = None

        try:
            session = await eden.create_session(prompt, model_name)
            result = await eden.validate_creation(session)

            if result.error or not result.result_url:
                error_message = result.error or 'No result URL'
                logger.warning(f"❌ Attempt {attempt + 1}/5 failed: {error_message}")
                
                # Exponential backoff with jitter
                if attempt < 4:  # Don't wait after the last attempt
                    wait_time = min(60, (2 ** attempt) + random.uniform(0, 1))
                    logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                    await asyncio.sleep(wait_time)
                continue
                
            logger.info(f"✅ CREATION: Artwork created successfully for session {session.id}")
            
            return {
                "success": True,
                "session_id": str(session.id),
                "session": session,
                "result_url": result.result_url,
                "announcement": result.announcement,
                "prompt": prompt
            }
            
        except Exception as e:
            error_message = f"Exception: {str(e)}"
            logger.error(f"❌ Attempt {attempt + 1}/5 exception: {error_message}")
            
            # Exponential backoff with jitter for exceptions too
            if attempt < 4:
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 1))
                logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                await asyncio.sleep(wait_time)
    
    logger.error(f"❌ CREATION: Failed after 5 attempts for prompt: {prompt[:80]}...")
    return {
        "success": False,
        "error": error_message or "Failed to create after 5 attempts",
        "prompt": prompt[:80]
    }


@app.function(
    image=image,
    timeout=60 * 60,
    max_containers=1,
)
async def genesis():
    genesis_creations = await eden.generate_creation_ideas()    
    creation_prompts = genesis_creations.creations[:GENERATION_COUNT]

    logger.info(f"✅ Generated {len(creation_prompts)} creation ideas for genesis")
    for i, idea in enumerate(genesis_creations.creations):
        logger.info(f"  Idea {i}: {idea[:100]}..." if len(idea) > 100 else f"  Idea {i}: {idea}")
    
    creations = []
    batch_size = MAX_PARALLEL_WORKERS
    for i in range(0, len(creation_prompts), batch_size):
        batch = creation_prompts[i:i+batch_size]
        async for result in init_creation.map.aio(batch):
            creations.append(result)
            logger.info(f"📥 CREATION RESULT {result.get('session_id')}: {'✅ Success' if result.get('success') else '❌ Failed'}")
        await asyncio.sleep(3)

    creations = [c for c in creations if c.get('success')]
    failed_creations = [c for c in creations if not c.get('success')]
    
    log_section(f"⛓️ SUBMITTING {len(creations)} CREATIONS TO BLOCKCHAIN SEQUENTIALLY")
    if len(creations) != GENERATION_COUNT:
        for failed_creation in failed_creations:
            logger.error(f"❌ Failed creation: {failed_creation.get('prompt')}")
        return

    for creation in creations:    
        creation['ipfs_url'] = ipfs.pin(creation['result_url'])

    log_section(f"⛓️ SUBMITTING {len(creations)} CREATIONS TO CHAIN")

    chain.clear_pending_and_reset()

    try:
        chain.create_batch_sessions_on_chain(creations)
    except Exception as e:
        logger.error(f"❌ Failed to submit creations to chain: {str(e)}")
        logger.error(f"Traceback: ", exc_info=True)
        logger.info("🔄 Fall back to individual submission")
        for creation in creations:
            chain.create_session_on_chain(
                session_id=creation['session_id'],
                message_id=creation['message_id'],
                content=creation['content'],
                ipfs_url=creation['ipfs_url']
            )
    


@app.function(
    image=image, 
    timeout=60 * 60, 
    max_containers=MAX_PARALLEL_WORKERS
)
async def process_creation_update(creation_data: tuple) -> dict:
    """Process updates for a single creation - check for blessings and respond if found."""
    session_id, creation = creation_data
    try:
        logger.info(f"🔄 UPDATE {session_id}...: Checking for new blessings")
                
        # Get new blessings for this session
        last_message_id = creation.get('last_processed_message_id')
        new_blessings = await chain.get_new_blessings(session_id, last_message_id)

        if not new_blessings:
            logger.info(f"📭 UPDATE {session_id}...: No new blessings found")
            return {
                "success": True,
                "action": "no_update",
                "session_id": session_id,
                "blessing_count": 0
            }
        
        # Found new blessings - process them
        logger.info(f"🎉 SESSION {session_id} -- {'🙏 '*5} BLESSING CONTENT ({len(new_blessings)} new blessings) {'🙏 '*5}")
        for i, blessing in enumerate(new_blessings, 1):
            logger.info(f"✨ Blessing {i} by {blessing['author']} : {blessing['content']}")
        
        # Process the blessings to create new artwork
        blessing_result, user_message = await eden.process_blessings_iteration(
            session_id, 
            new_blessings
        )

        if blessing_result.error:
            logger.error(f"❌ UPDATE {session_id}...: Blessing processing failed: {blessing_result.error}")
            return {
                "success": False,
                "action": "error",
                "session_id": session_id,
                "error": blessing_result.error,
                "blessing_count": len(new_blessings)
            }
        
        logger.info(f"🖼️ UPDATE {session_id}...: Creation URL: {blessing_result.result_url}")
        
        ipfs_url = ipfs.pin(blessing_result.result_url)
        
        chain.update_session_on_chain(
            session_id=session_id, 
            message_id=str(user_message.id), 
            ipfs_url=ipfs_url, 
            content=blessing_result.announcement
        )
        
        return {
            "success": True,
            "action": "updated",
            "session_id": session_id,
            "blessing_count": len(new_blessings),
            "artwork_url": blessing_result.result_url
        }
    
    except Exception as e:
        logger.error(f"\n❌ ERROR in process_creation_update: {str(e)}")
        logger.error(f"Traceback: ", exc_info=True)
        return {"error": str(e)}



@app.function(
    image=image, 
    timeout=60 * 60,
    max_containers=1,
)
async def destroy():
    """Destroy bottom half of sessions based on praise scores."""
    
    log_section("🏆 ELIMINATION BEGINNING")
    
    today = datetime.now(pytz.timezone('US/Eastern'))
    active_sessions = chain.get_contract_data(date_filter=today, closed_filter=False)
    
    if len(active_sessions) <= 1:
        logger.warning(f"⚠️ Only {len(active_sessions)} active session(s) found - nothing to destroy")
        if len(active_sessions) == 1:
            logger.info("✅ Tournament finished")
        return {
            "success": True,
            "total_sessions": len(active_sessions),
            "removed_sessions": 0,
            "remaining_sessions": len(active_sessions),
            "tournament_finished": len(active_sessions) == 1,
            "message": "Not enough sessions to destroy (need >1)"
        }
    
    # Sort by praise score (lowest first), with oldest winning ties
    session_scores = []
    for session_id, session_data in active_sessions.items():
        messages = session_data.get('messages', [])
        blessings = [msg for msg in messages if msg.get('author', '').lower() != config.ABRAHAM_ADDRESS.lower()]
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

    for i, session_info in enumerate(session_scores, 1):
        logger.info(
            f"{i:2d}. {session_info['session_id'][:8]}... | "
            f"Score: {session_info['praise_score']:3d} | "
            f"Praises: {session_info['total_praises']:2d} | "
            f"Blessings: {session_info['total_blessings']:2d} | "
            f"Started: {session_info['start_time']} | "
            f"{session_info['creation_prompt']}...")
    
    # Remove bottom half
    total_sessions = len(session_scores)
    sessions_to_remove = total_sessions // 2
    sessions_to_destroy = session_scores[:sessions_to_remove]
    sessions_to_keep = session_scores[sessions_to_remove:]
    
    # Add concluding remarks to sessions being destroyed
    logger.info(f"💀 ADDING CONCLUDING REMARKS TO {len(sessions_to_destroy)} DESTROYED SESSIONS")
    
    chain.clear_pending_and_reset()
    
    sessions_destroyed, sessions_destroy_errors = 0, 0
    for session_info in sessions_to_destroy:
        session_id = session_info['session_id']
        try:
            # close_session returns assistant_message
            assistant_message = await eden.close_session(
                session_id=session_id, 
                total_praises=session_info['total_praises'], 
                total_blessings=session_info['total_blessings']
            )

        # if error, create a fallback message
        except Exception as e:
            logger.error(f"❌ Error adding concluding remark for {session_id}: {str(e)}")
            logger.error(f"Traceback: ", exc_info=True)
            assistant_message = ChatMessage(
                id=ObjectId(),
                role="assistant",
                content="Closed by Abraham without comment",
            )

        try:
            tx_hash, receipt = chain.update_session_on_chain(
                session_id=str(session_id),
                message_id=str(assistant_message.id),
                ipfs_url="",
                content=assistant_message.content,
                closed=True
            )
            
            if receipt and receipt.status == 1:
                logger.info(f"✅ Final message published: {tx_hash.hex()}")
                sessions_destroyed += 1
            else:
                logger.error(f"❌ Failed to publish final message for {session_id}")
                sessions_destroy_errors += 1

            # Continue with destruction even if concluding remark fails
        except Exception as e:
            logger.error(f"❌ Error closing {session_id}: {str(e)}")
            logger.error(f"Traceback: ", exc_info=True)
            sessions_destroy_errors += 1
            continue
    
    sessions_survived = len(sessions_to_keep) + sessions_destroy_errors

    # Log survivors
    log_section(f"🎖️ SURVIVORS ({len(sessions_to_keep)} SESSIONS)")
    for i, session_info in enumerate(sessions_to_keep, 1):
        logger.info(
            f"🎖️ SURVIVOR {i}: {session_info['session_id'][:8]}... | "
            f"Score: {session_info['praise_score']} | "
            f"{session_info['creation_prompt']}")

    # Final verification
    log_section("💀 DESTROY SUMMARY")
    logger.info(f"📊 Starting sessions: {total_sessions}")
    logger.info(f"🎖️ Sessions expected to survive: {len(sessions_to_keep)}")
    logger.info(f"💀 Sessions expected to be destroyed: {len(sessions_to_destroy)}")
    logger.info(f"💀 Sessions destroyed: {sessions_destroyed}")
    logger.info(f"🎖️ Sessions actually surviving: {sessions_survived}")
        
    # Check if tournament should finish
    if sessions_survived <= 1:
        logger.info("🏆 Tournament finished after destroy cycle!")

        if sessions_survived == 1:
            logger.info("🏆 Tournament done... let's close it all")
            result = finish_tournament.remote()
            logger.info(f"🏆 Tournament finished: {result}")

        return {
            "success": True,
            "total_sessions": total_sessions,
            "removed_sessions": sessions_destroyed,
            "remaining_sessions": sessions_survived,
            "tournament_finished": True,
            "praise_scores": [
                {
                    "session_id": s['session_id'],
                    "praise_score": s['praise_score'],
                    "total_praises": s['total_praises'],
                    "total_blessings": s['total_blessings'],
                    "survived": s in sessions_to_keep
                }
                for s in session_scores
            ]
        }
    
    logger.info("✅ Destroy cycle complete")
    return {
        "success": True,
        "total_sessions": total_sessions,
        "removed_sessions": sessions_destroyed,
        "remaining_sessions": sessions_survived
    }


@app.function(
    image=image,
    timeout=60 * 60,
    max_containers=1,
)
async def finish_tournament():
    """Finish the tournament."""

    logger.info("🏆 CLOSING THE TOURNAMENT!! 123456")

    STATE["is_active"] = False

    return {
        "success": True,
        "tournament_finished": True
    }


@app.function(
    image=image, 
    timeout=60 * 60,
    max_containers=1,
)
async def update():
    """Check all active creations for new blessings and respond if found."""

    is_active = STATE.get("is_active", False)

    if not is_active:
        logger.info("🏆 Tournament not active, skipping update")
        return {
            "success": True,
            "skipped": True,
            "reason": "Tournament not active"
        }
    
    # Check if an update is already running
    is_running = STATE.get("update_running", False)
    genesis_running = STATE.get("genesis_running", False)
    if is_running or genesis_running:
        logger.warning("⚠️ Previous update or genesis still running, skipping this cycle")
        return {
            "success": True,
            "skipped": True,
            "reason": "Previous update or genesis still running"
        }
    
    # Mark update as running
    STATE["update_running"] = True
    
    try:
        log_section(f"🔄 UPDATE CYCLE - {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S EST')}")
        
        today = datetime.now(pytz.timezone('US/Eastern'))

        active_sessions = chain.get_contract_data(date_filter=today, closed_filter=False)
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
        
        chain.clear_pending_and_reset()
            
        update_results = []
        creation_data = list(zip(session_ids, active_sessions.values()))
        async for result in process_creation_update.map.aio(creation_data):
            update_results.append(result)

        for result in update_results:
            session_id = result.get('session_id')
            action = result.get('action')
            blessing_count = result.get('blessing_count', 0)
            success = result.get('success', False)
            
            if success and action == "updated":
                logger.info(f"📥 UPDATE RESULT {session_id}...: ✅ Responded to {blessing_count} blessings")
            elif success and action == "no_update":
                logger.info(f"📥 UPDATE RESULT {session_id}...: 📭 No new blessings")
            else:
                logger.error(f"📥 UPDATE RESULT {session_id}...: ❌ Error - {result.get('error')}")

        # Step 3: Process results
        total_sessions = len(active_sessions)
        updated_sessions = len([r for r in update_results if r.get('action') == 'updated'])
        no_update_sessions = total_sessions - updated_sessions
        error_sessions = len([r for r in update_results if not r.get('success')])
        total_blessings_processed = len([r for r in update_results if r.get('action') == 'updated'])
        
        logger.info("📊 UPDATE RESULTS")
        logger.info(f"📊 Total active sessions: {total_sessions}")
        logger.info(f"✅ Sessions updated with new art: {updated_sessions}")
        logger.info(f"📭 Sessions with no new blessings: {no_update_sessions}")
        logger.info(f"❌ Sessions with errors: {error_sessions}")
        logger.info(f"🙏 Total blessings processed: {total_blessings_processed}")        
        logger.info("✅ Update cycle complete")
        
        # Increment update counter
        update_count = STATE.get("update_count", 0) + 1
        STATE["update_count"] = update_count
        logger.info(f"📈 Update count: {update_count}")
        
        # Check if we should run destroy
        should_destroy = (update_count % DESTROY_N_UPDATES == 0)
        if should_destroy:
            logger.info(f"🎯 Running destroy cycle (update {update_count} is divisible by {DESTROY_N_UPDATES})")
            destroy_result = destroy.remote()
            logger.info(f"💀 Destroy result: {destroy_result}")
        else:
            logger.info(f"⏭️ Skipping destroy cycle (update {update_count}, next destroy at update {((update_count // DESTROY_N_UPDATES) + 1) * DESTROY_N_UPDATES})")

        ALL_RESULTS = {
            "success": True,
            "total_sessions": total_sessions,
            "updated_sessions": updated_sessions,
            "no_update_sessions": no_update_sessions,
            "error_sessions": error_sessions,
            "total_blessings_processed": total_blessings_processed,
            "update_count": update_count,
            "destroy_run": should_destroy
        }
        print(ALL_RESULTS)

        return ALL_RESULTS

    except Exception as e:
        logger.error(f"❌ Error in update: {str(e)}")
        logger.error(f"Traceback: ", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        # Always mark update as not running
        STATE["update_running"] = False


@app.function(
    image=image, 
    timeout=60 * 60,
    max_containers=1,
)
async def close_all_open_sessions():
    """Check all active creations for new blessings and respond if found."""

    log_section(f"🔄 UPDATE CYCLE - {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S EST')}")
        
    today = datetime.now(pytz.timezone('US/Eastern'))

    active_sessions = chain.get_contract_data(date_filter=today, closed_filter=False)
    
    if not active_sessions:
        logger.info("No active sessions found to close")
        logger.info("✅ Destroy cycle complete")
        return

    logger.info(f"Found {len(active_sessions)} active sessions to close: {list(active_sessions.keys())[:5]}...")
    
    chain.clear_pending_and_reset()

    # Only try to close sessions that actually exist and are not already closed
    sessions_closed = 0
    sessions_skipped = 0
    
    for session_id, session in active_sessions.items():
        try:
            # First check if session exists and is open on-chain
            # The subgraph might have stale data or data from a different contract
            logger.info(f"Checking session {session_id[:8]}... on chain")
            
            # Try to update the session
            tx_hash, receipt = chain.update_session_on_chain(
                session_id=str(session_id),
                message_id=str(ObjectId()),
                ipfs_url="",
                content="Closed by Abraham",
                closed=True
            )
            
            if receipt and receipt.status == 1:
                logger.info(f"✅ Session {session_id[:8]}... closed successfully")
                sessions_closed += 1
            else:
                logger.error(f"❌ Failed to close session {session_id[:8]}...")
                sessions_skipped += 1
                
        except Exception as e:
            # Convert exception to string to avoid serialization issues
            error_msg = str(e)
            
            if "Session not found" in error_msg or "RetryError" in error_msg:
                logger.info(f"⚠️ Session {session_id[:8]}... not found on chain (subgraph may be out of sync)")
                sessions_skipped += 1
                continue
            
            logger.error(f"❌ Error closing session {session_id[:8]}...: {error_msg}")
            sessions_skipped += 1
            # Don't re-raise to avoid serialization issues with Modal
            # Continue with other sessions
    
    logger.info(f"Session close summary: {sessions_closed} closed, {sessions_skipped} skipped")
    logger.info("✅ Close all sessions complete")


# Parse GENESIS_TIME to get hour and minute for cron
_genesis_hour, _genesis_minute = GENESIS_TIME.split(':')

@app.function(
    image=image,
    timeout=60 * 60,
    max_containers=1,
    schedule=modal.Cron(f"{_genesis_minute} {_genesis_hour} * * *", timezone=TIMEZONE),
)
async def scheduled_genesis():
    """Daily orchestrator that starts the genesis at the configured time."""
    log_section(f"🌅 DAILY GENESIS - {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S EST')}")
    logger.info(f"⏰ Starting daily tournament at {GENESIS_TIME} {TIMEZONE}")

    # Mark tournament as active
    STATE["is_active"] = True
    
    # Reset the update counter for the new day
    STATE["update_count"] = 0
    logger.info("🔄 Reset update counter for new tournament")
    
    log_section("🔄 CLOSING ALL OPEN SESSIONS")
    close_all_open_sessions.remote()

    # Start the genesis
    try:
        STATE["genesis_running"] = True
        genesis_result = genesis.remote()
        logger.info("✅ Genesis completed successfully")
        STATE["genesis_running"] = False
        return genesis_result
    except Exception as e:
        logger.error(f"❌ Genesis failed: {str(e)}")
        logger.error(f"Traceback: ", exc_info=True)
        return {"success": False, "error": str(e)}



@app.function(
    image=image,
    timeout=60 * 60,
    max_containers=1,
    schedule=modal.Period(minutes=config.UPDATE_INTERVAL),
)
async def scheduled_update():
    """Scheduled update that runs every UPDATE_INTERVAL minutes."""
    logger.info(f"⏰ Scheduled update triggered at {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S EST')}")
    return update.remote()


@app.function(
    image=image,
    timeout=60 * 60,
    max_containers=1,
)
async def sample_contract_call():
    chain.create_session_on_chain(
        session_id="1234567890", 
        message_id="1234567890", 
        content="this is a test",
        ipfs_url="https://gateway.pinata.cloud/ipfs/QmW14Kpx88b4pUBbE9SU2LFGvCRpfuaxy4Ao9Yb8q25f2t",
        nonce=None
    )


@app.local_entrypoint()
def main():
    """Local entrypoint for testing - resets state for fresh tournament."""
    logger.info("🚀 ABRAHAM TOURNAMENT ORCHESTRATOR")
    # reset STATE
    # STATE.clear()
    # close_all_open_sessions.remote()
    # scheduled_genesis.remote()
    sample_contract_call.remote()
    

# python -c "from chain import get_contract_data; print(len(get_contract_data()))"