import asyncio
import pytz
import json
import uuid
import pytz
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal
from bson import ObjectId
from fastapi import BackgroundTasks
from jinja2 import Template
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field
from jinja2 import Template
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.session.models import PromptSessionContext, ChatMessageRequestInput, LLMConfig, UpdateType
from eve.agent.session.session import add_user_message, build_llm_context, async_prompt_session
from eve.auth import get_my_eden_user
from eve.agent import Agent
from eve.tool import Tool
from eve.tools.tool_handlers import handlers
from eve.agent.session.models import (
    ChatMessage, ChatMessageRequestInput, LLMConfig, 
    LLMContext, PromptSessionContext, Session
)

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

from config import (
    logger,
    DEBUG,
    MODEL_NAME,
    GENERATION_COUNT,
    USER_ID
)



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


user = get_my_eden_user()
abraham = Agent.load("abraham")


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
        description=f"A list of {NUM_CREATIONS} project proposals, including title and statement"
    )


async def handle_drafting(
    args: Dict[str, Any], 
    user: str = None, 
    agent: str = None
) -> Dict[str, Any]:
    results = {}
    for creation in args.copy()["creations"]:
        session = await create_session(
            title=creation["title"], 
            proposal=creation["proposal"]
        )
        results.append({
            "title": creation["title"],
            "session_id": str(session.id)
        })

    return {"output": {"creations": results}}


# Create and register the tool
custom_tool = Tool.register(
    ProjectProposals, 
    handle_drafting
)


async def draft_proposals(genesis_session: str = None):
    background_tasks = BackgroundTasks()

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
            background_tasks, 
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
        initiating_user_id=request.user_id,
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
        











class CompactMessage(BaseModel):
    """Condense your last messages and any tool results into a single message with resulting media attachments."""

    content: Optional[str] = Field(description="Content of the message. Cannot exceed 320 characters.")
    media_urls: Optional[List[str]] = Field(description="URLs of any images or videos to attach")


CAST_TEMPLATE = """# Compact your last messages into a single message

Given your last messages (starting from the message which begins with "{{message_ref}}"), extract from it the following:

media_urls: any successfully produced images, videos, or other media produced by your tool calls
content: a single message, not exceeding 320 characters, which swaps these original messages with a new one which summarizes or restates the original messages as though the whole sequence 

This new message is meant to abstract your original messages -- which may include various thinking, statements of intent (e.g. "Let me ..."), tool calls and results, error handling and retries, or tool sequences -- into a single summarial message in the same tone of your original messages which gives the high level of what you did.
"""


async def compact_messages(session: Any, new_messages: List[ChatMessage]) -> CompactMessage:
    """Extract a compact message from the last assistant messages."""
    system_message = system_template.render(
        name=abraham.name,
        current_date_time=get_current_timestamp(),
        description=abraham.description,
        persona=abraham.persona,
        tools=None
    )

    messages = [ChatMessage.from_mongo(m) for m in session.messages]
    
    message_ref = (new_messages[0].content or "")[:25]
    instruction_prompt = Template(CAST_TEMPLATE).render(
        message_ref=message_ref
    )

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message), 
            *messages,
            ChatMessage(role="system", content=instruction_prompt)
        ],
        config=LLMConfig(
            model="claude-sonnet-4-20250514",
            response_format=CompactMessage
        )
    )
    response = await async_prompt(context)
    result = CompactMessage(**json.loads(response.content))
    return result



INIT_CREATION_TEMPLATE = """The title of your next creation is:
{{title}}

The proposal you've made for this creation is:
{{proposal}}

# Task

Expand on your proposal with a more complete exploration, and generate exactly one 16:9 image that captures it."""


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     before_sleep=lambda retry_state: logger.info(f"Retrying create_session (attempt {retry_state.attempt_number}/3)...")
# )


async def run_creation_session(
    session: Session,
    title: str,
    proposal: str
): 
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

    return session


async def create_session(
    title: str, 
    proposal: str
) -> Session:
    background_tasks = BackgroundTasks()
    
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
        background_tasks, 
        request.session_id, 
        request.user_id, 
        request
    )
    
    # run creation session in background
    asyncio.create_task(run_creation_session(session, title, proposal))
    
    return session


if __name__ == "__main__":
    asyncio.run(draft_proposals(genesis_session=GENESIS_SESSION))
    # import time
    # while True:
    #     print("sleep")
    #     time.sleep(5)
