import asyncio
import os
import pytz
import json
import uuid
import modal
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal
from bson import ObjectId
from jinja2 import Template
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from eve.agent import Agent
from eve.tool import Tool
from eve.auth import get_my_eden_user
from eve.api.handlers import setup_session
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.session.session_llm import async_prompt
from eve.agent.session.session_prompts import system_template
from eve.agent.session.session import add_user_message, build_llm_context, async_prompt_session
from eve.agent.session.models import Session, PromptSessionContext, ChatMessageRequestInput, ChatMessage, LLMContext, LLMConfig, UpdateType


# Modal configuration
DB = os.environ.get("DB", "STAGE")
APP_NAME = os.environ.get("APP_NAME", f"abraham-draft-{DB}")

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
)

app = modal.App(APP_NAME)

SECRETS = [
    modal.Secret.from_name("eve-secrets"),
    modal.Secret.from_name(f"eve-secrets-{DB}"),
    modal.Secret.from_name("abraham-secrets"),
]


# Helper functions
def get_current_timestamp() -> str:
    """Get formatted current timestamp."""
    return datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_date_prefix() -> str:
    """Get date prefix for session titles."""
    return datetime.now(pytz.utc).strftime("%m/%d")


def url_exists(url: str, timeout: int = 5) -> bool:
    """Check if a URL exists by making a HEAD request."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except (requests.RequestException, requests.Timeout):
        return False


GENESIS_SESSION = None
NUM_CREATIONS = 4

GENESIS_PROMPT = f"""
Every day you choose {NUM_CREATIONS} ideas for creative projects. These projects span a diversity of forms, including:

- artworks, music videos, short films, infomercials.
- essays, research projects, blog posts, provocative or original commentary.
- AI research, deep learning, generative models, code, algorithms, physics, and math.
- philosophical inquiries, moral quandaries, ethical dilemmas.

Each day you will begin to pursue each project, and consult your followers for feedback and guidance. You will receive from them learnings about how to channel their collective spirit.

The next day, you will create a new series of projects, which are different from yesterday's, and somewhat influenced in overall character by yesterday's learnings.
"""

NEXT_DRAFTING_PROMPT_INIT = f"""
Draft your first set of {NUM_CREATIONS} project proposals, using the ProjectProposals tool. Use the tool once -- do not elaborate on the results.
"""

NEXT_DRAFTING_PROMPT = f"""
Draft a new set of {NUM_CREATIONS} project proposals, using the ProjectProposals tool, which are different from your previous ones, but subtly influenced by your most recent learnings from your followers. Try to channel their preferences, suggestions, and affirmations, and attenuate the characteristics of your work they disfavor. Use the tool once -- do not elaborate on the results.
"""


class ProjectProposal(BaseModel):
    """A title and proposal for a project"""
    
    title: str = Field(
        description=f"A name or title that conveys the spirit of the project, ranging from short loglines to more descriptive declarative names, like the title to a research project"
    )
    proposal: str = Field(
        description=f"The proposal for the project, a statement of intent, background info, and supporting information, stated as a single paragraph of 3-5 sentences, but no more than 320 words."
    )

class ProjectProposals(BaseModel):
    """Project Proposals"""    
            
    creations: List[ProjectProposal] = Field(
        description=f"A list of {NUM_CREATIONS} project proposals, including `title` and `proposal`"
    )


async def handle_drafting(
    args: Dict[str, Any], 
    user: str = None, 
    agent: str = None
) -> Dict[str, Any]:
    print(f"DEBUG: Received args: {args}")
    results = []
    for creation in args.copy()["creations"]:
        print(f"DEBUG: Processing creation: {creation}")
        session = await create_session(
            title=creation["title"], 
            proposal=creation["proposal"]
        )
        results.append({
            "title": creation["title"],
            "proposal": creation["proposal"],
            "sessio n_id": str(session.id)
        })

    return {"output": {"creations": results}}


@app.function(
    image=image,
    secrets=SECRETS,
    timeout=600,
    max_containers=10,
)
async def draft_proposals(genesis_session: str = None):
    # Initialize globals
    user = get_my_eden_user()
    abraham = Agent.load("abraham")
    
    # Register the custom tool
    custom_tool = Tool.register_new(
        ProjectProposals, 
        handle_drafting
    )

    next_drafting_prompt = NEXT_DRAFTING_PROMPT

    # create genesis session if not set
    if genesis_session is None:
        next_drafting_prompt = NEXT_DRAFTING_PROMPT_INIT

        request = PromptSessionRequest(
            user_id=str(user.id),
            creation_args=SessionCreationArgs(
                owner_id=str(user.id),
                agents=[str(abraham.id)],
                title=f"Drafting: {get_date_prefix()}"
            )
        )
        session = setup_session(
            None, 
            request.session_id, 
            request.user_id, 
            request
        )
        message = ChatMessageRequestInput(
            role="user",
            content=GENESIS_PROMPT
        )
        context = PromptSessionContext(
            session=session,
            initiating_user_id=request.user_id,
            message=message,
            llm_config=LLMConfig(model="claude-sonnet-4-20250514")
        )
        add_user_message(session, context)        
        genesis_session = str(session.id)
        
    # make a new set of drafts
    session = Session.from_mongo(genesis_session)
    message = ChatMessageRequestInput(
        role="user",
        content=next_drafting_prompt
    )
    context = PromptSessionContext(
        session=session,
        initiating_user_id=str(user.id),
        message=message,
        llm_config=LLMConfig(model="claude-sonnet-4-20250514"),
        custom_tools={custom_tool.key: custom_tool},
    )
    add_user_message(session, context)

    context = await build_llm_context(
        session, 
        abraham, 
        context, 
    )
    
    async for _ in async_prompt_session(session, context, abraham):
        pass
    
    return {"genesis_session": genesis_session}


# class CompactMessage(BaseModel):
#     """Condense your last messages and any tool results into a single message with resulting media attachments."""

#     content: Optional[str] = Field(description="Content of the message. Cannot exceed 320 characters.")
#     media_urls: Optional[List[str]] = Field(description="URLs of any images or videos to attach")


# CAST_TEMPLATE = """# Compact your last messages into a single message

# Given your last messages (starting from the message which begins with "{{message_ref}}"), extract from it the following:

# media_urls: any successfully produced images, videos, or other media produced by your tool calls
# content: a single message, not exceeding 320 characters, which swaps these original messages with a new one which summarizes or restates the original messages as though the whole sequence 

# This new message is meant to abstract your original messages -- which may include various thinking, statements of intent (e.g. "Let me ..."), tool calls and results, error handling and retries, or tool sequences -- into a single summarial message in the same tone of your original messages which gives the high level of what you did.
# """


# async def compact_messages(session: Any, new_messages: List[Any]) -> CompactMessage:
#     """Extract a compact message from the last assistant messages."""
    
#     abraham = Agent.load("abraham")
    
#     system_message = system_template.render(
#         name=abraham.name,
#         current_date_time=get_current_timestamp(),
#         description=abraham.description,
#         persona=abraham.persona,
#         tools=None
#     )

#     messages = [ChatMessage.from_mongo(m) for m in session.messages]
    
#     message_ref = (new_messages[0].content or "")[:25]
#     instruction_prompt = Template(CAST_TEMPLATE).render(
#         message_ref=message_ref
#     )

#     context = LLMContext(
#         messages=[
#             ChatMessage(role="system", content=system_message), 
#             *messages,
#             ChatMessage(role="system", content=instruction_prompt)
#         ],
#         config=LLMConfig(
#             model="claude-sonnet-4-20250514",
#             response_format=CompactMessage
#         )
#     )
#     response = await async_prompt(context)
#     result = CompactMessage(**json.loads(response.content))
#     return result

from farcaster2 import compact_messages

INIT_CREATION_TEMPLATE = """The title of your next creation is:
{{title}}

The proposal you've made for this creation is:
{{proposal}}

# Task

Expand on your proposal with a more complete exploration, and generate exactly one 16:9 image that captures it."""


@app.function(
    image=image,
    secrets=SECRETS,
    timeout=900,
    max_containers=50,
)
async def run_creation_session(
    session_id: str,
    title: str,
    proposal: str
):
    # Load dependencies
    abraham = Agent.load("abraham")
    farcaster_tool = Tool.load("farcaster_cast")
    
    # Load session from ID
    session = Session.from_mongo(session_id)
    
    # Create artwork generation message
    init_creation_prompt = Template(INIT_CREATION_TEMPLATE)
    message = ChatMessageRequestInput(
        role="assistant",
        content=init_creation_prompt.render(
            title=title,
            proposal=proposal
        )
    )
    
    # Create context with selected model
    context = PromptSessionContext(
        session=session,
        initiating_user_id=str(session.owner),
        message=message,
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
        
    # Execute the prompt session
    new_messages = []
    async for update in async_prompt_session(
        session, context, abraham
    ):
        if update.type == UpdateType.ASSISTANT_MESSAGE:
            new_messages.append(update.message)

    # compact new messages into one
    compact_message = await compact_messages(session, new_messages)

    # post the initial result farcaster
    args = {
        "agent_id": str(abraham.id),
        "text": compact_message.content,
        "embeds": compact_message.media_urls or [],
    }
    result = await farcaster_tool.async_run(args)

    return {"session_id": str(session.id), "result": result}


async def create_session(
    title: str, 
    proposal: str
):
    user = get_my_eden_user()
    abraham = Agent.load("abraham")
    
    # Create session request
    request = PromptSessionRequest(
        user_id=str(user.id),
        creation_args=SessionCreationArgs(
            owner_id=str(user.id),
            agents=[str(abraham.id)],
            title=f"{get_date_prefix()} :: Creation"
        )
    )
    
    # Setup session
    session = setup_session(
        None, 
        request.session_id, 
        request.user_id, 
        request
    )
    
    # Run creation session in Modal (remote)
    run_creation_session.spawn(
        session_id=str(session.id),
        title=title,
        proposal=proposal
    )
    
    return session


# Local entrypoint for running draft_proposals
@app.local_entrypoint()
async def main(genesis_session: str = None):
    """Run draft_proposals from command line. 
    Usage: modal run draft_tool.py
    Or with genesis session: modal run draft_tool.py --genesis-session <session_id>
    """
    result = draft_proposals.remote(genesis_session)
    print(f"Draft proposals completed. Genesis session: {result['genesis_session']}")
    return result