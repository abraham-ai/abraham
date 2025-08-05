from datetime import datetime
import asyncio
import modal
import pytz

from chain import create_session_on_chain
import config


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
    .add_local_file("contract_abi.json", "/root/contract_abi.json")
)

with image.imports():
    from config import (
        log_section,
        logger,
        MODEL_NAME,
        GENERATION_COUNT,
        MAX_PARALLEL_WORKERS,
        FALLBACK_MODEL_NAME,
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

ACTIVE = modal.Dict.from_name(
    f"{config.APP_NAME}-active-creations", 
    create_if_missing=True
)
STATE = modal.Dict.from_name(
    f"{config.APP_NAME}-state", 
    create_if_missing=True
)


@app.function(image=image, timeout=60 * 60, max_containers=MAX_PARALLEL_WORKERS)
async def init_creation(prompt: str) -> dict:
    """Create artwork only (parallel) - no blockchain operations."""
    logger.info(f"🎨 ART CREATION: {prompt[:80]}...")
    
    # Try up to 5 times with exponential backoff
    import asyncio
    import random
    
    for attempt in range(5):
        model_name = MODEL_NAME if attempt < 2 else FALLBACK_MODEL_NAME
        logger.info(f"Attempt {attempt + 1}/5 - Model: {model_name}")
        error_message = None

        try:
            session = await eden.create_session(prompt, model_name)
            result = await eden.validate_creation(session)

            logger.info(f"ART CREATION RESULT: {result}")
            if result.error or not result.result_url:
                error_message = result.error or 'No result URL'
                logger.warning(f"❌ Attempt {attempt + 1}/5 failed: {error_message}")
                
                # Exponential backoff with jitter
                if attempt < 4:  # Don't wait after the last attempt
                    wait_time = min(60, (2 ** attempt) + random.uniform(0, 1))
                    logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                    await asyncio.sleep(wait_time)
                continue
                
            logger.info(f"✅ ART CREATION: Artwork created successfully for session {session.id}")
            
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
            logger.error(f"   Error type: {type(e).__name__}")
            if hasattr(e, '__cause__') and e.__cause__:
                logger.error(f"   Cause: {str(e.__cause__)}")
            
            # Exponential backoff with jitter for exceptions too
            if attempt < 4:
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 1))
                logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                await asyncio.sleep(wait_time)
    
    logger.error(f"❌ ART CREATION: Failed after 5 attempts for prompt: {prompt[:80]}...")
    return {
        "success": False,
        "error": error_message or "Failed to create artwork after 5 attempts",
        "prompt": prompt[:80]
    }


@app.function(
    image=image, 
    # schedule=modal.Cron("0 13 * * *"),
    timeout=60 * 60, 
    max_containers=1
)
async def genesis():
    """Create generation of creations to start the tournament."""
    try:
        log_section(f"🌅 GENESIS - {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Step 1: Clear the active array
        existing_items = list(ACTIVE.items())
        logger.info(f"Found {len(existing_items)} existing active creations. Deleting...")
        for session_id, _ in existing_items:
            logger.info(f"  {ACTIVE[session_id]}")
            del ACTIVE[session_id]
            logger.info(f"🗑️ Cleared session: {session_id}")
        
        logger.info("✅ Active creations array cleared")

        # Step 2: Generate creation ideas
        log_section(f"💡 GENERATING {GENERATION_COUNT} CREATION IDEAS")
        
        genesis_creations = await eden.generate_creation_ideas()
        
        logger.info(f"Generated {len(genesis_creations.creations)} creation ideas")
        for i, idea in enumerate(genesis_creations.creations):
            logger.info(f"  Idea {i}: {idea[:100]}..." if len(idea) > 100 else f"  Idea {i}: {idea}")
            
        creation_prompts = genesis_creations.creations[:GENERATION_COUNT]
        logger.info(f"✅ Generated {len(creation_prompts)} creation ideas for genesis")
        
        # Step 3: Clear any pending transactions and reset nonce tracking
        log_section(f"🗑️ CLEARING PENDING TRANSACTIONS AND RESETTING NONCE TRACKING")
        chain.clear_pending_and_reset()
        
        # Step 4: Create all artworks in parallel (no blockchain yet)
        log_section(f"🎨 CREATING {len(creation_prompts)} ARTWORKS IN PARALLEL")

        artwork_results = []
        try:
            # Process in smaller batches to avoid overwhelming Eve/MongoDB
            batch_size = MAX_PARALLEL_WORKERS
            for i in range(0, len(creation_prompts), batch_size):
                batch = creation_prompts[i:i+batch_size]
                logger.info(f"📦 Processing art batch {i//batch_size + 1}/{(len(creation_prompts) + batch_size - 1)//batch_size}")
                
                async for result in init_creation.map.aio(batch):
                    artwork_results.append(result)
                    success = result.get('success', False)
                    session_id = result.get('session_id', '?')
                    logger.info(f"📥 ART RESULT {session_id}: {'✅ Success' if success else '❌ Failed'}")
                
                # Add delay between batches to avoid overwhelming MongoDB
                if i + batch_size < len(creation_prompts):                    
                    logger.info("⏳ Waiting 2s before next batch to avoid overwhelming Eve...")
                    await asyncio.sleep(2)
                    
        except Exception as e:
            logger.error(f"❌ Error in parallel artwork creation: {str(e)}")

        # Step 5: Submit successful artworks to blockchain sequentially
        successful_artworks = [r for r in artwork_results if r.get('success')]
        failed_artworks = [r for r in artwork_results if not r.get('success')]
        
        log_section(f"⛓️ SUBMITTING {len(successful_artworks)} ARTWORKS TO BLOCKCHAIN SEQUENTIALLY")
        
        if successful_artworks:
            # Clear and reset nonce tracking before sequential submission
            chain.clear_pending_and_reset()
            
            creation_results = []
            for i, artwork in enumerate(successful_artworks, 1):
                try:
                    logger.info(f"⛓️ {i}/{len(successful_artworks)}: Submitting {artwork['session_id'][:8]}... to blockchain")
                    
                    # Upload to IPFS
                    ipfs_url = chain.upload_to_ipfs(artwork['result_url'])
                    
                    # Submit to blockchain (sequential - no nonce conflicts)
                    create_session_on_chain(
                        session_id=artwork['session_id'], 
                        message_id=str(artwork['session'].messages[-1]), 
                        content=artwork['announcement'],
                        ipfs_url=ipfs_url,
                        nonce=None  # Let it auto-allocate
                    )
                    
                    # Add to active sessions
                    ACTIVE[artwork['session_id']] = {
                        "session_id": artwork['session_id'],
                        "last_processed_message_id": str(artwork['session'].messages[-1]),
                        "creation_prompt": artwork['prompt']
                    }
                    
                    creation_results.append({
                        "success": True,
                        "session_id": artwork['session_id'],
                        "original_url": artwork['result_url'],
                        "ipfs_url": ipfs_url,
                        "announcement": artwork['announcement']
                    })
                    
                    logger.info(f"✅ {i}/{len(successful_artworks)}: Blockchain submission successful")
                    
                    # Small delay between blockchain submissions
                    await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"❌ {i}/{len(successful_artworks)}: Blockchain submission failed for {artwork['session_id']}: {e}")
                    creation_results.append({
                        "success": False,
                        "session_id": artwork['session_id'],
                        "error": f"Blockchain submission failed: {str(e)}"
                    })
            
            # Add failed artworks to results
            for artwork in failed_artworks:
                creation_results.append({
                    "success": False,
                    "error": artwork.get('error', 'Artwork creation failed'),
                    "prompt": artwork.get('prompt', 'Unknown')
                })
        else:
            logger.error("❌ No successful artworks to submit to blockchain")
            creation_results = failed_artworks

        # Step 6: Process results
        log_section("📊 GENESIS RESULTS")
        
        successful_count = sum(1 for r in creation_results if r.get('success'))
        failed_count = len(creation_results) - successful_count
        
        for result in creation_results:
            if result.get('success'):
                session_id = result.get('session_id', 'unknown')
                logger.info(f"✅ GENESIS: Session {session_id} created successfully")
            else:
                error = result.get('error', 'Unknown error')
                prompt = result.get('prompt', 'Unknown prompt')
                nonce = result.get('nonce', 'Unknown nonce')
                session_id = result.get('session_id', 'No session')
                logger.error(f"❌ GENESIS FAILURE:")
                logger.error(f"   Error: {error}")
                logger.error(f"   Prompt: {prompt}...")
                logger.error(f"   Session: {session_id}")
                logger.error(f"   Nonce: {nonce}")
        
        log_section("🌅 GENESIS SUMMARY")
        logger.info(f"🎯 Target creations: {GENERATION_COUNT}")
        logger.info(f"✅ Successful creations: {successful_count}")
        logger.info(f"❌ Failed creations: {failed_count}")
        
        return {
            "success": True,
            "target_count": GENERATION_COUNT,
            "successful_count": successful_count,
            "failed_count": failed_count
        }
        
    except Exception as e:
        logger.error(f"\n❌ ERROR in genesis: {str(e)}")
        logger.error(f"Traceback: ", exc_info=True)
        return {"error": str(e)}


@app.function(
    image=image, 
    timeout=60 * 60, 
    max_containers=MAX_PARALLEL_WORKERS
)
async def process_creation_update(session_data_tuple: tuple) -> dict:
    """Process updates for a single creation - check for blessings and respond if found."""
    session_id, creation_data = session_data_tuple
    try:
        logger.info(f"🔄 UPDATE {session_id}...: Checking for new blessings")
                
        # Get new blessings for this session
        last_message_id = creation_data.get('last_processed_message_id')
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
        logger.info(f"🎉 UPDATE {session_id}...: Found {len(new_blessings)} new blessings")
        logger.info(f"{'🙏 '*5} BLESSING CONTENT {'🙏 '*5}")
        
        for i, blessing in enumerate(new_blessings, 1):
            logger.info(f"✨ Blessing {i}:")
            logger.info(f"   👤 Author: {blessing['author']}")
            logger.info(f"   💬 Content: {blessing['content']}")
            logger.info(f"   🆔 Message ID: {blessing['uuid']}")
        
        # Process the blessings to create new artwork
        logger.info(f"🎨 UPDATE {session_id}...: Creating new artwork based on blessings...")
        
        # Process the blessings to create new artwork
        blessing_result, user_message = await eden.process_blessings_iteration(session_id, new_blessings)

        # Get the last message ID from new blessings
        last_blessing_id = new_blessings[-1]['uuid'] if new_blessings else None
        logger.debug(f"Last blessing message ID: {last_blessing_id}")

        logger.debug(f"Blessing result: {blessing_result}")
        logger.debug(f"User message: {user_message}")
        logger.debug(f"Last blessing ID: {last_blessing_id}")
        
        if blessing_result.error:
            logger.error(f"❌ UPDATE {session_id}...: Blessing processing failed: {blessing_result.error}")
            return {
                "success": False,
                "action": "error",
                "session_id": session_id,
                "error": blessing_result.error,
                "blessing_count": len(new_blessings)
            }
        
        logger.info(f"✅ UPDATE {session_id}...: Successfully created new artwork")
        logger.info(f"🖼️ UPDATE {session_id}...: Artwork URL: {blessing_result.result_url}")
        
        # Publish to blockchain
        logger.info(f"🔗 UPDATE {session_id}...: Publishing to blockchain...")
        
        ipfs_url = chain.upload_to_ipfs(blessing_result.result_url)
        chain.update_session_on_chain(
            session_id=session_id, 
            message_id=str(user_message.id), 
            ipfs_url=ipfs_url, 
            content=blessing_result.announcement
        )
        logger.info(f"✅ UPDATE {session_id}...: Published to blockchain: {ipfs_url}")
        
        # Update the session data in Modal Dict (but don't remove it)
        logger.info(f"💾 UPDATE {session_id}...: Updating session data in Modal Dict...")
        updated_data = creation_data.copy()
        updated_data['last_processed_message_id'] = str(user_message.id)
        
        ACTIVE[session_id] = updated_data
        
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
    # schedule=modal.Cron(f"30 13 * * *"),
    timeout=60 * 60
)
async def update():
    """Check all active creations for new blessings and respond if found."""
    try:
        log_section(f"🔄 UPDATE CYCLE - {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Step 1: Load all active sessions
        log_section("📋 LOADING ACTIVE SESSIONS")
        logger.info(f"Found {len(list(ACTIVE.items()))} active sessions")
        
        active_sessions = {}
        for session_id, creation_data in ACTIVE.items():
            active_sessions[session_id] = creation_data
            logger.info(f"✓ Loaded session: {session_id}... - {creation_data}")
            
        if not active_sessions:
            logger.warning("⚠️ No active sessions found - nothing to update")
            return {
                "success": True,
                "total_sessions": 0,
                "updated_sessions": 0,
                "error_sessions": 0
            }
        
        # Step 2: Clear any pending transactions and reset nonce tracking
        log_section(f"🗑️ CLEARING PENDING TRANSACTIONS AND RESETTING NONCE TRACKING")
        chain.clear_pending_and_reset()
        
        # Step 3: Process all sessions in parallel to check for blessings (without pre-allocated nonces)
        log_section(f"🔄 CHECKING {len(active_sessions)} SESSIONS FOR BLESSINGS (PARALLEL)")
        
        update_results = []
        try:
            # Prepare data for parallel processing without pre-allocated nonces
            # Each process_creation_update will handle its own nonce allocation if needed
            session_ids = [str(s) for s in list(active_sessions.keys())]
            creation_data_list = list(active_sessions.values())
            session_data_tuples = list(zip(session_ids, creation_data_list))

            async for result in process_creation_update.map.aio(session_data_tuples):
                update_results.append(result)
                session_id = result.get('session_id', 'unknown')[:8]
                action = result.get('action', 'unknown')
                blessing_count = result.get('blessing_count', 0)
                success = result.get('success', False)
                
                if success and action == "updated":
                    logger.info(f"📥 UPDATE RESULT {session_id}...: ✅ Responded to {blessing_count} blessings")
                elif success and action == "no_update":
                    logger.info(f"📥 UPDATE RESULT {session_id}...: 📭 No new blessings")
                else:
                    logger.error(f"📥 UPDATE RESULT {session_id}...: ❌ Error - {result.get('error')}")
                    
        except Exception as e:
            logger.error(f"❌ Error in parallel update processing: {str(e)}")
        
        # Step 3: Process results
        log_section("📊 UPDATE RESULTS")
        
        total_sessions = len(active_sessions)
        updated_sessions = 0
        no_update_sessions = 0
        error_sessions = 0
        total_blessings_processed = 0
        
        for result in update_results:
            action = result.get('action', 'unknown')
            blessing_count = result.get('blessing_count', 0)
            
            if result.get('success'):
                if action == "updated":
                    updated_sessions += 1
                    total_blessings_processed += blessing_count
                    session_id = result.get('session_id', 'unknown')[:8]
                    logger.info(f"✅ UPDATED {session_id}...: Processed {blessing_count} blessings")
                elif action == "no_update":
                    no_update_sessions += 1
            else:
                error_sessions += 1
                session_id = result.get('session_id', 'unknown')[:8]
                logger.error(f"❌ ERROR {session_id}...: {result.get('error')}")
        
        log_section("🔄 UPDATE SUMMARY")
        logger.info(f"📊 Total active sessions: {total_sessions}")
        logger.info(f"✅ Sessions updated with new art: {updated_sessions}")
        logger.info(f"📭 Sessions with no new blessings: {no_update_sessions}")
        logger.info(f"❌ Sessions with errors: {error_sessions}")
        logger.info(f"🙏 Total blessings processed: {total_blessings_processed}")        
        logger.info("✅ Update cycle complete")
        
        return {
            "success": True,
            "total_sessions": total_sessions,
            "updated_sessions": updated_sessions,
            "no_update_sessions": no_update_sessions,
            "error_sessions": error_sessions,
            "total_blessings_processed": total_blessings_processed
        }
        
    except Exception as e:
        logger.error(f"\n❌ ERROR in update: {str(e)}")
        logger.error(f"Traceback: ", exc_info=True)
        logger.info("❌ Update cycle failed")
        return {"error": str(e)}


@app.function(
    image=image, 
    # schedule=modal.Cron(f"0 16 * * *"),
    timeout=60 * 60
)
async def destroy():
    """Rank all active creations by praise and remove the bottom half."""
    try:
        # Step 1: Load all active sessions
        log_section(f"💀 DESTROY CYCLE - {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        log_section("📋 LOADING ACTIVE SESSIONS FOR RANKING")
        active_sessions = {}
        dict_items = list(ACTIVE.items())
        logger.info(f"Found {len(dict_items)} active sessions")
        
        for session_id, creation_data in dict_items:
            active_sessions[session_id] = creation_data
            logger.info(f"✓ Loaded session: {session_id}... - {creation_data['creation_prompt'][:60]}...")
        
        if len(active_sessions) <= 1:
            logger.warning(f"⚠️ Only {len(active_sessions)} active session(s) found - nothing to destroy")
            if len(active_sessions) == 1:
                logger.info("🏆 Tournament finished - winner remains!")
                # Reset state for next day's tournament
                STATE["genesis_date"] = None
                STATE["last_update"] = 0
                STATE["last_destroy"] = 0
                logger.info("✅ Tournament state reset for next day's tournament")
            return {
                "success": True,
                "total_sessions": len(active_sessions),
                "removed_sessions": 0,
                "remaining_sessions": len(active_sessions),
                "tournament_finished": len(active_sessions) == 1,
                "message": "Not enough sessions to destroy (need >1)"
            }
        
        # Step 2: Calculate praise scores for each session
        log_section(f"📊 CALCULATING PRAISE SCORES FOR {len(active_sessions)} SESSIONS")

        logger.info(f"📊 Scoring {len(active_sessions.keys())} sessions...")

        session_ids = [str(s) for s in list(active_sessions.keys())]
        sessions = chain.get_contract_data(session_ids)
        session_scores = []
        for session_id, session_data in sessions.items():
            logger.info(f"📊 Scoring session: {session_id}...")
            messages = session_data.get('messages', [])
            total_praises = sum(msg.get('praiseCount', 0) for msg in messages)
            total_blessings = sum(1 for msg in messages if msg.get('author', '').lower() != config.ABRAHAM_ADDRESS.lower())
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
            logger.info(f"THE SESSION FIRST TIME IS: {session_data.get('firstMessageAt')}")
            logger.info(f"📊 {session_id}...: Score={praise_score} (praises={total_praises}, blessings={total_blessings})")
            
        # Step 3: Sort by praise score (lowest first), with oldest winning ties
        log_section("🏆 RANKING SESSIONS BY PRAISES")
        
        # Sort by praise_score ascending, then by start_time ascending (oldest first for ties)
        session_scores.sort(key=lambda x: (x['praise_score'], x['start_time']))
        logger.info("📊 PRAISE RANKINGS (lowest to highest):")
        
        for i, session_info in enumerate(session_scores, 1):
            # Convert start_time to datetime if it's a timestamp
            start_time = session_info['start_time']
            if isinstance(start_time, (int, float, str)):
                try:
                    # Try to convert timestamp to datetime
                    if isinstance(start_time, str):
                        start_time = float(start_time)
                    start_time_dt = datetime.fromtimestamp(start_time, tz=pytz.utc)
                    formatted_time = start_time_dt.strftime('%m-%d %H:%M')
                except (ValueError, TypeError):
                    formatted_time = str(start_time)
            else:
                # Already a datetime object
                formatted_time = start_time.strftime('%m-%d %H:%M')
                
            logger.info(
                f"{i:2d}. {session_info['session_id'][:8]}... | "
                f"Score: {session_info['praise_score']:3d} | "
                f"Praises: {session_info['total_praises']:2d} | "
                f"Blessings: {session_info['total_blessings']:2d} | "
                f"Started: {formatted_time} | "
                f"{session_info['creation_prompt']}...")
        
        # Step 4: Remove bottom half
        total_sessions = len(session_scores)
        sessions_to_remove = total_sessions // 2  # Integer division for bottom half
        remaining_sessions = total_sessions - sessions_to_remove
        
        log_section(f"💀 REMOVING BOTTOM {sessions_to_remove} OF {total_sessions} SESSIONS")
        logger.info(f"🎯 Removing: {sessions_to_remove} sessions")
        logger.info(f"🎯 Keeping: {remaining_sessions} sessions")
        
        # Get the sessions to remove (bottom half)
        sessions_to_destroy = session_scores[:sessions_to_remove]
        sessions_to_keep = session_scores[sessions_to_remove:]
        
        # Step 4.5: Add concluding remarks to sessions being destroyed
        log_section(f"💭 ADDING CONCLUDING REMARKS TO {len(sessions_to_destroy)} DESTROYED SESSIONS")
        
        for session_info in sessions_to_destroy:
            session_id = session_info['session_id']
            try:
                # close_session returns assistant_message
                assistant_message = await eden.close_session(
                    session_id=session_id, 
                    total_praises=session_info['total_praises'], 
                    total_blessings=session_info['total_blessings']
                )

                tx_hash, receipt = chain.update_session_on_chain(
                    session_id=str(session_id),
                    message_id=str(assistant_message.id),
                    ipfs_url="",
                    content=assistant_message.content,
                    closed=True
                )
                
                if receipt and receipt.status == 1:
                    logger.info(f"✅ Final message published: {tx_hash.hex()}")
                else:
                    logger.error(f"❌ Failed to publish final message for {session_id}")
                    
            except Exception as e:
                logger.error(f"❌ Error adding concluding remark for {session_id}: {str(e)}")
                logger.error(f"Traceback: ", exc_info=True)
                raise e
                # Continue with destruction even if concluding remark fails
        
        # Remove from Modal Dict
        removed_count = 0
        for session_info in sessions_to_destroy:
            session_id = session_info['session_id']
            if not session_id in ACTIVE:
                logger.warning(f"⚠️ Session {session_id}... not found in dict for removal")
                continue
            del ACTIVE[session_id]
            removed_count += 1
            logger.info(
                f"💀 DESTROYED {session_id}... | "
                f"Score: {session_info['praise_score']} | "
                f"{session_info['creation_prompt']}")
            
        # Log survivors
        log_section(f"🎖️ SURVIVORS ({len(sessions_to_keep)} SESSIONS)")
        for i, session_info in enumerate(sessions_to_keep, 1):
            logger.info(
                f"🎖️ SURVIVOR {i}: {session_info['session_id'][:8]}... | "
                f"Score: {session_info['praise_score']} | "
                f"{session_info['creation_prompt']}")

        # Final verification
        final_sessions = list(ACTIVE.keys())
        final_count = len(final_sessions)
        
        log_section("💀 DESTROY SUMMARY")
        logger.info(f"📊 Starting sessions: {total_sessions}")
        logger.info(f"💀 Sessions destroyed: {removed_count}")
        logger.info(f"🎖️ Sessions surviving: {final_count}")
        logger.info(f"🎯 Expected survivors: {remaining_sessions}")
        
        if final_count != remaining_sessions:
            logger.warning(f"⚠️ Final count ({final_count}) doesn't match expected ({remaining_sessions})")
        
        # Check if tournament should finish
        if final_count <= 1:
            logger.info("🏆 Tournament finished after destroy cycle!")
            # Reset state for next day's tournament
            STATE["genesis_date"] = None
            STATE["last_update"] = 0
            STATE["last_destroy"] = 0
            logger.info("✅ Tournament state reset for next day's tournament")
            return {
                "success": True,
                "total_sessions": total_sessions,
                "removed_sessions": removed_count,
                "remaining_sessions": final_count,
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
            "removed_sessions": removed_count,
            "remaining_sessions": final_count,
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

    except Exception as e:
        logger.error(f"\n❌ ERROR in destroy: {str(e)}")
        logger.error(f"Traceback: ", exc_info=True)
        logger.info("❌ Destroy cycle failed")
        return {"error": str(e)}


@app.function(
    image=image,
    schedule=modal.Cron(f"*/{config.CYCLE_CHECK_INTERVAL} * * * *", timezone=config.TIMEZONE),
    timeout=60 * 60,
    max_containers=1,
)
async def orchestrate():
    """Simple tournament orchestrator - runs genesis at scheduled time, then update/destroy cycles."""
    now = datetime.now(pytz.timezone(config.TIMEZONE))
    today = now.date().isoformat()
    current_time = now.strftime("%H:%M")
    current_minute = now.hour * 60 + now.minute
    
    logger.info(f"🕐 ORCHESTRATE at {current_time}")
    
    # Get state
    genesis_done_today = STATE.get("genesis_date") == today
    last_update = STATE.get("last_update", 0)
    last_destroy = STATE.get("last_destroy", 0)
    
    # 1. Run genesis if it's time and we haven't done it today
    if not genesis_done_today and current_minute >= config.GENESIS_MIN:
        logger.info(f"🌅 Starting genesis for {today}")
        try:
            genesis.remote()
            STATE["genesis_date"] = today
            STATE["last_update"] = current_minute
            STATE["last_destroy"] = current_minute
            logger.info("✅ Genesis started")
        except Exception as e:
            logger.error(f"❌ Genesis failed: {e}")
        return
    
    # 2. If genesis not done yet, wait
    if not genesis_done_today:
        minutes_until = config.GENESIS_MIN - current_minute
        logger.info(f"⏰ Waiting for genesis time (in {minutes_until} minutes)")
        return
    
    # 3. Tournament is active - check how many sessions we have
    active_count = len(list(ACTIVE.items()))
    logger.info(f"   Active sessions: {active_count}")
    
    # 4. If 1 or fewer sessions, tournament is over - reset for tomorrow
    if active_count <= 1:
        logger.info("🏆 Tournament finished - resetting for tomorrow")
        STATE["genesis_date"] = None
        STATE["last_update"] = 0
        STATE["last_destroy"] = 0
        return
    
    # 5. Check what cycles are due
    minutes_since_destroy = current_minute - last_destroy
    minutes_since_update = current_minute - last_update
    destroy_due = minutes_since_destroy >= config.DESTROY_INTERVAL
    update_due = minutes_since_update >= config.UPDATE_INTERVAL
    
    # 6. Run update if due (either standalone or before destroy)
    if update_due:
        logger.info(f"🔄 Running update ({minutes_since_update}min since last)")
        try:
            update.remote()
            STATE["last_update"] = current_minute
            logger.info("✅ Update completed")
        except Exception as e:
            logger.error(f"❌ Update failed: {e}")
    
    # 7. Run destroy if due (after update if both were due)
    if destroy_due:
        logger.info(f"💀 Running destroy ({minutes_since_destroy}min since last)")
        try:
            destroy.remote()
            STATE["last_destroy"] = current_minute
            logger.info("✅ Destroy started")
        except Exception as e:
            logger.error(f"❌ Destroy failed: {e}")
    
    # 8. If neither was due, log current status
    if not update_due and not destroy_due:
        next_update = config.UPDATE_INTERVAL - minutes_since_update
        next_destroy = config.DESTROY_INTERVAL - minutes_since_destroy
        logger.info(f"⏰ Next update in {next_update}min, destroy in {next_destroy}min")


@app.local_entrypoint()
def main():
    """Local entrypoint for testing - resets state for fresh tournament."""
    logger.info("🚀 ABRAHAM TOURNAMENT ORCHESTRATOR")
    logger.info(f"   Genesis: {config.GENESIS_TIME} ({config.TIMEZONE})")
    logger.info(f"   Intervals: Update {config.UPDATE_INTERVAL}min, Destroy {config.DESTROY_INTERVAL}min")
    
    # Reset state for debugging
    old_date = STATE.get("genesis_date", None)
    STATE["genesis_date"] = None
    STATE["last_update"] = 0
    STATE["last_destroy"] = 0
    
    logger.info(f"   Previous genesis: {old_date}")
    logger.info("   ✅ State reset")
    
    # Clear active sessions
    existing_items = list(ACTIVE.items())
    if existing_items:
        logger.info(f"   Clearing {len(existing_items)} active sessions")
        for session_id, _ in existing_items:
            del ACTIVE[session_id]
        logger.info("   ✅ Active sessions cleared")
    
    # Clear all pending transactions from mempool
    logger.info("🗑️ Clearing pending transactions from mempool...")
    try:
        chain.clear_pending_and_reset()
        logger.info("   ✅ Mempool cleared and nonce tracking reset")
    except Exception as e:
        logger.error(f"   ❌ Failed to clear mempool: {e}")
    
    # Show timing
    now = datetime.now(pytz.timezone(config.TIMEZONE))
    current_time = now.strftime("%H:%M")
    logger.info(f"   Current time: {current_time}")
    logger.info("✅ Ready for tournament")
