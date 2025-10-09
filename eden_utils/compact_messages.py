"""
Utility for compacting messages for Farcaster casts.
"""

import json
from typing import List, Optional, Any
from jinja2 import Template
from pydantic import BaseModel, Field

from eve.agent.agent import Agent
from eve.agent.session.session_llm import async_prompt
from eve.agent.session.session_prompts import system_template
from eve.agent.session.models import ChatMessage, LLMContext, LLMConfig, Session
from eden import get_current_timestamp


class CompactMessage(BaseModel):
    """Condense your last messages and any tool results into single message with resulting media attachments."""

    content: Optional[str] = Field(description="Content of the message. Cannot exceed 320 characters.")
    media_urls: Optional[List[str]] = Field(description="URLs of any images or videos to attach")


CAST_TEMPLATE = """
# Compact your last messages into a single message

Given your last messages (starting from the message which begins with "{{message_ref}}"), extract from it the following:

media_urls: any successfully produced images, videos, or other media produced by your tool calls
content: a single message, not exceeding 320 characters, which swaps these original messages with a new one which summarizes or restates the original messages as though the whole sequence 

This new message is meant to abstract your original messages -- which may include various thinking, statements of intent (e.g. "Let me ..."), tool calls and results, error handling and retries, or tool sequences -- into a single summarial message in the same tone of your original messages which gives the high level of what you did.
"""


async def compact_messages(session: Session, new_messages: List[ChatMessage]) -> CompactMessage:
    """Extract a compact message from the last assistant messages."""

    abraham = Agent.load("abraham")

    system_message = system_template.render(
        name=abraham.name,
        current_date_time=get_current_timestamp(),
        description=abraham.description,
        persona=abraham.persona,
        tools=None
    )

    messages = session.get_messages()
    message_ref = (new_messages[0].content or "")[:25]
    instruction = Template(CAST_TEMPLATE).render(message_ref=message_ref)

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message), 
            *messages,
            ChatMessage(role="system", content=instruction)
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5",
            response_format=CompactMessage
        )
    )
    response = await async_prompt(context)
    result = CompactMessage(**json.loads(response.content))
    
    return result