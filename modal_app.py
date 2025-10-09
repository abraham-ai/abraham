"""
Consolidated Modal app orchestration for Abraham services.
Deploy with: modal deploy modal_app.py
"""

import os
import modal

# Configuration
DB = os.environ.get("DB", "STAGE")
APP_NAME = f"abraham-{DB}"

# Build the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": DB})
    .apt_install("git", "libmagic1", "ffmpeg", "wget")
    .pip_install("web3", "eth-account", "requests", "jinja2", "python-dotenv", "pytz", "tenacity", "httpx")
    .run_commands(
        'echo hellooo2oo'
    )
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
    # Add the function files
    .add_local_file("farcaster2.py", "/root/farcaster2.py")
    .add_local_dir("eden_utils", "/root/eden_utils")
)

# Create the app
app = modal.App(APP_NAME)

# Secrets
SECRETS = [
    modal.Secret.from_name("eve-secrets"),
    modal.Secret.from_name(f"eve-secrets-{DB}"),
    modal.Secret.from_name("abraham-secrets"),
]

# ======================================================================================
#                               Farcaster Webhook Functions
# ======================================================================================

@app.function(
    image=image,
    secrets=SECRETS,
    timeout=120,
    min_containers=1,
    max_containers=10,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def webhook():
    """Farcaster webhook handler."""
    from farcaster2 import fastapi_app
    return fastapi_app

@app.function(
    image=image,
    secrets=SECRETS,
    timeout=900,
    max_containers=50,
)
async def process_event(event: dict) -> None:
    """Process Farcaster events."""
    from farcaster2 import process_cast
    await process_cast(event)

# ======================================================================================
#                               Creation/Drafting Functions
# ======================================================================================

@app.function(
    image=image,
    secrets=SECRETS,
    timeout=600,
    max_containers=1,
    # schedule=modal.Cron("30 10 * * *", timezone="America/New_York"),  # 10:30 AM EST/EDT
)
async def draft_proposals(genesis_session: str = None):
    """Draft new creation proposals."""
    return {"success": True}
    from eden_utils.init_creations import draft_proposals as draft_proposals_impl
    return await draft_proposals_impl(genesis_session)

@app.function(
    image=image,
    secrets=SECRETS,
    timeout=900,
    max_containers=10,
)
async def run_creation(session_id: str, title: str, proposal: str):
    """Run a creation session."""
    return {"success": True}
    from eden_utils.init_creations import run_creation_session
    return await run_creation_session(session_id, title, proposal)

@app.function(
    image=image,
    secrets=SECRETS,
    timeout=7200,
    max_containers=1,
    schedule=modal.Cron("50 10 * * *", timezone="America/New_York"),  # 10:45 AM EST/EDT
)
async def testfunc():
    """Run draft proposals from command line.
    Usage: modal run modal_app.py::run_drafts
    """
    return {"success": True}
    
    from farcaster2 import AbrahamCreation
    from eve.agent.session.models import Session, ChatMessage
    

    from datetime import datetime, timedelta
    import pytz

    today = datetime.now(pytz.utc).strftime("%Y-%m-%d")

    # cutoff_date = datetime.now() - timedelta(days=1)

    active_creations = AbrahamCreation.find({
        "status": "active", 
        # "createdAt": {"$gt": cutoff_date}
        "day": today
    })
    print("--------------------------------")
    for a in active_creations:
        print(a.id, a.createdAt)
    print("--------------------------------")


    if len(active_creations) == 0:
        print("No active creations found.")
        return {"success": False, "reason": "no active creations"}

    max_messages = -1
    max_session = None
    max_creation = None

    for creation in active_creations:
        session = Session.from_mongo(creation.session_id)
        messages = session.get_messages()
        num_messages = len(messages)
        print(num_messages, session.id)
        if num_messages > max_messages:
            max_messages = num_messages
            max_session = session
            max_creation = creation

    if max_session is None:
        print("No active creations or sessions found.")
        return {"success": False, "reason": "no active creations"}

    # from bson import ObjectId
    winner_id = str(max_session.id)


    import eden
    import ipfs
    import auction
    
    # logger.info(f"🏆 Winner: {winner_id}")
    MODEL_NAME            = "claude-sonnet-4-5"
    FALLBACK_MODEL_NAME   = "gpt-4o"
    
    video_result = await eden.create_video(winner_id, MODEL_NAME, FALLBACK_MODEL_NAME)

    
    
    # video_result_poster_image_url = "https://dtut5r9j4w7j4.cloudfront.net/1fe140cb77347d479c9ba05237e37326b66c48c8f6f8e5b632fe4f7af423d2ef.jpg"
    # video_url_url = "https://dtut5r9j4w7j4.cloudfront.net/591517521621312417d5f305871b0d27a2d400bab0eb49fa18639af2b7027370.mp4"
    # poster_image_url = ipfs.pin(video_result_poster_image_url)  #video_result.poster_image_url)
    # video_url = ipfs.pin(video_url_url)
    
    poster_image_url = ipfs.pin(video_result.poster_image_url)
    poster_image_hash = poster_image_url.split("/")[-1]

    
    video_url = ipfs.pin(video_result.video_url)
    video_hash = video_url.split("/")[-1]

    # json_data = {
    #     "description": "This is a test output from the Abraham tournament.", # video_result.writeup,
    #     "external_url": "https://abraham.ai/covenant",
    #     "image": f"ipfs://{poster_image_hash}",
    #     "video": f"ipfs://{video_hash}",
    #     "name": "The Covenant Test", # video_result.title,
    #     "attributes": []
    # }

    json_data = {
        "description": video_result.writeup,
        "external_url": "https://abraham.ai/covenant",
        "image": f"ipfs://{poster_image_hash}",
        "video": f"ipfs://{video_hash}",
        "name": video_result.title,
        "attributes": []
    }

    ipfs_url = ipfs.pin(json_data)
    ipfs_hash = ipfs_url.split("/")[-1]

    auction.set_token(ipfs_hash)


    # mark saved creation as completed
    max_creation.status = "closed"
    max_creation.save()


    return {"success": True}


# ======================================================================================
#                               Local Entrypoints
# ======================================================================================

@app.local_entrypoint()
async def run_drafts(genesis_session: str = None):
    """Run draft proposals from command line.
    Usage: modal run modal_app.py::run_drafts
    """
    genesis_session = "68cac5f94750193ce8f59a35"  # None
    result = draft_proposals.remote(genesis_session)
    print(f"Draft proposals completed. Genesis session: {result['genesis_session']}")
    return result



@app.local_entrypoint()
async def test(genesis_session: str = None):
    """Run draft proposals from command line.
    Usage: modal run modal_app.py::run_drafts
    """
    result = testfunc.remote()
    print(result)
    return {"success": "ok"}