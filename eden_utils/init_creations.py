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

from eden_utils.compact_messages import compact_messages
from farcaster2 import FarcasterEvent


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

NUM_CREATIONS = 2

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
    results = []
    for creation in args.copy()["creations"]:
        session = await create_session(
            title=creation["title"], 
            proposal=creation["proposal"]
        )
        results.append({
            "title": creation["title"],
            "proposal": creation["proposal"],
            "session_id": str(session.id)
        })

    return {"output": {"creations": results}}


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
    
    async for m in async_prompt_session(session, context, abraham):
        pass
    
    return {"genesis_session": genesis_session}


INIT_CREATION_TEMPLATE = """
The title of your next creation is:
{{title}}

The proposal you've made for this creation is:
{{proposal}}

# Task

Expand on your proposal with a more complete exploration, and generate exactly one 16:9 image that captures it."""


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

    if len(new_messages) == 0:
        return {
            "status": "failed", 
            "session_id": str(session.id), 
            "error": "No new messages"
        }

    # compact new messages into one
    compact_message = await compact_messages(session, new_messages)

    # post the initial result farcaster
    result = await farcaster_tool.async_run({
        "agent_id": str(abraham.id),
        "text": compact_message.content,
        "embeds": compact_message.media_urls or [],
    })

    if result.get('status') != 'completed':
        return {
            "status": "failed", 
            "session_id": str(session.id), 
            "error": result.get('error')
        }

    # update session key to the hash
    cast_hash = result.get('output')[0].get('cast_hash')
    session.session_key = f"FC-{cast_hash}"
    session.save()

    for output in result.get('output'):
        event = FarcasterEvent(
            session_id=session.id,
            message_id=new_messages[0].id,
            cast_hash=output.get('cast_hash'),
            status="completed",
            event=None
        )
        event.save()

    return {
        "status": "completed", 
        "session_id": str(session.id), 
        "result": result
    }


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
            title=f"{get_date_prefix()} :: Creation",
        )
    )
    
    # Setup session
    session = setup_session(
        None, 
        request.session_id, 
        request.user_id, 
        request
    )
    
    # run session remotely
    from modal_app import run_creation
    run_creation.spawn(
        session_id=str(session.id),
        title=title,
        proposal=proposal
    )
    
    return session

