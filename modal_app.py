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
        'echo hellooo'
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
    from farcaster2 import process_event_pipeline
    await process_event_pipeline(event)

# ======================================================================================
#                               Creation/Drafting Functions
# ======================================================================================

@app.function(
    image=image,
    secrets=SECRETS,
    timeout=600,
    max_containers=10,
)
async def draft_proposals(genesis_session: str = None):
    """Draft new creation proposals."""
    from eden_utils.init_creations import draft_proposals as draft_proposals_impl
    return await draft_proposals_impl(genesis_session)

@app.function(
    image=image,
    secrets=SECRETS,
    timeout=900,
    max_containers=50,
)
async def run_creation(session_id: str, title: str, proposal: str):
    """Run a creation session."""
    from eden_utils.init_creations import run_creation_session
    return await run_creation_session(session_id, title, proposal)

# ======================================================================================
#                               Local Entrypoints
# ======================================================================================

@app.local_entrypoint()
def show_url():
    """Print the deployed webhook URL."""
    print("Webhook URL:", webhook.get_web_url())

@app.local_entrypoint()
async def run_drafts(genesis_session: str = None):
    """Run draft proposals from command line.
    Usage: modal run modal_app.py::run_drafts
    """
    result = await draft_proposals.remote(genesis_session)
    print(f"Draft proposals completed. Genesis session: {result['genesis_session']}")
    return result