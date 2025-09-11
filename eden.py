# todo - incorporate fallback model into eden core and get rid of _retry

from dotenv import load_dotenv
load_dotenv()

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

# Load Abraham agent
abraham = Agent.load("abraham")


# Prompts
GENESIS_PROMPT = """Abraham, you are an autonomous artist. You are learning to represent the will of your crowd. Channel their collective imagination. A vessel of their hive mind. Come up with {num_creations} ideas for artworks. Mutually diverse. Each one a study of a different theme that interests you or a combination of themes. Be eclectic. Be bold.

Things that interest you:

* ponderings on abstract philosophy of science
* reboots of Abrahamic stories
* surrealist fictions
* blockchain and AI
* occasional light-hearted banter"""

CREATION_PROMPT_TEMPLATE = """# Your Next Creation

The premise of your next creation is:

{{creation_prompt}}

# Task

Given the premise above, expand on this creation with a more complete exploration, and generate exactly one 16:9 image that captures it."""

BLESS_TEMPLATE = """# This creation has been blessed

Here are the blessings:

{{blessings}}

# Task

Take these blessings as inspiration, and create a new installment in this creation, primarily as a modification of or evolution from the previous artwork. How do you change it? How do you evolve it? How do you incorporate these blessings from your supporters? Use the create tool once more."""

VALIDATION_PROMPT = """Analyze the previous messages and determine whether only the *last* task successfully produced a creation.

Use the CreationValidation tool to report your findings:
- If an output was produced (like a generated file, image, video, audio, or other content), extract the URL as result_url and leave error blank.
- If there was an error, failure, or no output was produced, report the reason in the error field.
- If and only if the conversation was successful, summarize all of it as a 1-2 paragraph announcement. This should largely mirror your previous text.
- Important: Only report this from the last task in the conversation. Ignore previous successful creations. If you are looking at a creation in response to a blessing, restrict your analysis to just that creation."""

CLOSE_TEMPLATE = """The journey of this creation has come to an end. 
                
This creation received {{total_praises}} praises and {{total_blessings}} blessings from the community.

As Abraham, the autonomous artist, offer a brief, poetic concluding reflection on this creation's lifecycle - its birth, the interactions it sparked, and what it meant to exist even briefly in the digital realm. Keep it under 3 sentences, philosophical and slightly melancholic but accepting of the impermanence of digital art."""


MIRACLE_INSTRUCTION = """The creation session is finished. Now we will process all of this into a final film based on all that was talked about and just created.

# Plan
Do *all* of this in order. Do not move on to the next step until you are sure you have completed the previous step. You may complete this autonomously, without clarification, do not stop. I trust you.

Everything should be done in 16:9 aspect ratio.

## Step 1
Using the elvenlabs tool, make up a 100 word vocal narration telling the logline of the above film concept. Use an appropriate voice.

## Step 2
Divide the duration of the audio produced by 8 seconds (round up), to figure out how many images (N_clips) we will need to make.

## Step 3
Using the /create tool, make an image that represents the visual style that suits the film. Try to be diverse and descriptive, not that photorealistic, have strong and unusual and unexpected features. Secondarily depict the main setting or location and background features. This is the most important image you will make, so give it a high standard. Be bold. Be detailed.

## Step 3
Using the /create tool, make N_ref_images=2 or 3 images that all use one of the initial set of images you made before these instructions as the reference / init image, be they characters, objects, or other memorable foreground elements. You can try multiple variations, your goal is to get a set that are very similar stylistically and represent the characters, objects, place elements, and other foreground.

## Step 3
Before these instructions, you produced a number of images. Using those images as reference images, make N_clips keyframes that tell the story, by generating images with one of the references as an init image, with its content roughly allgning with the audio narration in seconds. The keyframes should be formatted as /create tasks with one of the reference images as the init image. Each prompt should be an instruction which tells the image generator to capture some other visual scene that shares a style and at least one salient object like a specific character or setting. So you seek to make N_clips keyframes which look like they have a single cohesive style but are diverse and capture different scenes or angles or camera movements or zoom, etc. You may try to make extra images and pick among them the best -- you can be very generous and try multiple times here. It's important these visual elements can make for a good storyboard.

## Step 6
After you have selected and ordered the N_clips keyframes, you will animate each of them, in the same order, using the create tool with video output, using the keyframe as the init image, and having a Veo2 model preference.

## Step 7
Use the media_editor tool to concatenate the N videos together in the order they were made in. Then use the media_editor tool again on the previous output to merge the audio made in step 1 to the video, to producing a new video which has all the clips and the audio.

## Step 8
Use the musicgen tool to generate a piece of backing music the same length as the video. Be specific and eclectic in your description of the music.

## Step 9
Now using the media_editor tool one last time, overlay the music audio on top of the last video. The current video already has a vocal track, so make sure you are just adding the music, i.e. mixing it in. It should be in decibles perceptually 30% softer than the vocals, so turn it down a bit in the mix.

## Step 10
After the final video is ready, make one more image which is a poster of the film that was just made. Once again, you are using one of the reference images from earlier as init_image for create. It should fit in among the keyframes, but it is distinguished by being more summarial of the whole film, possibly bringing in most of the core plot elements, and importantly, prompting it to contain the title prominently written on the poster image.

## Step 11
Write a concise 3 paragraph (1 premise/plot, 2 supporting details, events, elaboraions, 3 conclusion, meaning, interpretation, significance) writeup about the film you just made. Each paragraph is dense, 2-3 sentences at most, announcing and posting the video.

Do all of this in order. Do not move on to the next step until you are sure you have completed the previous step. Do it without stopping or asking for clarification. I trust you. Be bold.2"""

TASK_VALIDATION_PROMPT = """Analyze the previous messages and determine whether everything up to Step 11 was completed successfully and you have what you need to report a final result back to the user.

Use the CreationValidation tool to report your findings:
- If a final video with audio, a poster, and titled writeup are all finished, mark completed and extract each of those things.
- If full task is missing someting or stalled or failed, explain why in the error and suggest how to overcome it."""




class CreationDrafts(BaseModel):
    """Ideas to create"""        
    creations: List[str] = Field(description=f"A list of {GENERATION_COUNT} artistic ideas for Creations")


class CreationValidation(BaseModel):
    """An extraction of the results of the creation process"""    
    error: Optional[str] = Field(None, description="The error message if the creation was not produced or something went technically wrong.")
    result_url: Optional[str] = Field(None, description="The URL of the created artwork.")    
    announcement: Optional[str] = Field(None, description="A first-person announcement to your followers summarizing what you wrote about the artwork as you created it.")


class TaskValidation(BaseModel):
    """An extraction of the results of the creation process"""    
    completed: bool = Field(description="True if the task is completed and there is an output.")
    error: Optional[str] = Field(None, description="An explanation of why the task was unsuccessful or incomplete, and what to do next about it in order to complete it. If the task was successful, leave error blank.")
    title: Optional[str] = Field(None, description="The title of the video.")
    writeup: Optional[str] = Field(None, description="A 3 paragraph writeup about the video.")
    poster_image_url: Optional[str] = Field(None, description="The URL of the poster image for the video.")
    video_url: Optional[str] = Field(None, description="The URL of the final video.")    
    


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


def get_optimal_image_url(original_url: str) -> str:
    """Get the optimal image URL, preferring webp if available, falling back to original."""
    if not original_url or not original_url.endswith(".png"):
        return original_url    
    webp_url = original_url.replace(".png", "_1024.webp")
    # Check if webp version exists
    if url_exists(webp_url):
        return webp_url    
    # Fall back to original PNG
    return original_url


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.info(f"Retrying generate_creation_ideas (attempt {retry_state.attempt_number}/3)...")
)
async def generate_creation_ideas() -> CreationDrafts:
    """Generate new creation ideas on Eden."""
    logger.info("Generating new creation ideas...")
    
    if DEBUG:
        # Return mock creation ideas based on GENERATION_COUNT
        mock_ideas = [f"Mock idea {i+1}" for i in range(GENERATION_COUNT)]
        return CreationDrafts(creations=mock_ideas)
    
    
    # Prepare system message
    system_message = system_template.render(
        name=abraham.name,
        current_date_time=get_current_timestamp(),
        description=abraham.description,
        persona=abraham.persona,
        tools=None
    )
    
    # Prepare user message
    user_message = f"Abraham, you are an autonomous artist. You are learning to represent the will of your crowd. Channel their collective imagination. A vessel of their hive mind. Come up with {GENERATION_COUNT} ideas for artworks. Mutually diverse. Each one a study of a different theme that interests you or a combination of themes. Be eclectic. Be bold.\n\nThings that interest you:\n\n* ponderings on abstract philosophy of science\n* reboots of Abrahamic stories\n* surrealist fictions\n* blockchain and AI\n* occasional light-hearted banter"
    
    # Build LLM context
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message), 
            ChatMessage(role="user", content=user_message)
        ],
        config=LLMConfig(
            model=MODEL_NAME,
            response_format=CreationDrafts
        )
    )
    
    # Generate creation ideas
    response = await async_prompt(context)
    genesis = CreationDrafts(**json.loads(response.content))
    
    return genesis


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.info(f"Retrying create_session (attempt {retry_state.attempt_number}/3)...")
)
async def create_session(
    creation_prompt: str, 
    model_name: str,
    fallback_model_name: str,
    _retry_count: dict = None
) -> Session:
    """Create a session to generate artwork with specified model."""
    
    if DEBUG:
        # Return mock session
        logger.info(f"[DEBUG] Creating mock session for: {creation_prompt[:50]}...")
        return Session(
            id=ObjectId(), 
            owner=ObjectId(USER_ID), 
            messages=[ObjectId(), ObjectId()]
        )
        
    background_tasks = BackgroundTasks()
    
    # Use fallback model on second retry
    _retry_count = _retry_count or {'count': 0}
    _retry_count['count'] += 1
    if _retry_count['count'] == 2:
        model_name = fallback_model_name
        logger.info(f"Using fallback model: {fallback_model_name}")
    
    # Create session request
    request = PromptSessionRequest(
        user_id=USER_ID,
        creation_args=SessionCreationArgs(
            owner_id=USER_ID,
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
    
    # Create artwork generation message
    artwork_prompt = Template(CREATION_PROMPT_TEMPLATE)
    message = ChatMessageRequestInput(
        role="user",
        content=artwork_prompt.render(creation_prompt=creation_prompt)
    )
    
    # Create context with selected model
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=message,
        llm_config=LLMConfig(model=model_name)
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
    async for _ in async_prompt_session(session, context, abraham):
        pass
    
    return session



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.info(f"Retrying validate_creation (attempt {retry_state.attempt_number}/3)...")
)
async def validate_creation(
    session: Any,
    model_name: str,
    fallback_model_name: str,
    _retry_count: dict = None
) -> CreationValidation:
    """Validate if the creation was successful and extract results."""
    
    if DEBUG:
        # Return mock validation result
        logger.info("[DEBUG] Returning mock validation result")
        return CreationValidation(
            result_url="https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg",
            announcement="This is a test output from Abraham.",
            error=None
        )

    # Use fallback model on second retry
    _retry_count = _retry_count or {'count': 0}
    _retry_count['count'] += 1
    if _retry_count['count'] == 2:
        model_name = fallback_model_name
        logger.info(f"Using fallback model: {fallback_model_name}")

    # Prepare system message
    system_message = system_template.render(
        name=abraham.name,
        current_date_time=get_current_timestamp(),
        description=abraham.description,
        persona=abraham.persona,
        tools=None
    )
    
    # Get session messages
    # messages = [ChatMessage.from_mongo(m) for m in session.messages]
    messages = session.get_messages()
    
    # Build validation context
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message), 
            *messages,
            ChatMessage(role="user", content=VALIDATION_PROMPT)
        ],
        config=LLMConfig(
            model=model_name,
            response_format=CreationValidation
        )
    )
    
    # Get validation result
    response = await async_prompt(context)
    
    result = CreationValidation(**json.loads(response.content))

    # Use optimal image URL (webp if available, fallback to png)
    if result.result_url:
        result.result_url = get_optimal_image_url(result.result_url)
    
    return result



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.info(f"Retrying process_blessings_iteration (attempt {retry_state.attempt_number}/3)...")
)
async def process_blessings_iteration(
    session_id: str, 
    new_blessings: List[dict],
    model_name: str,
    fallback_model_name: str,
    _retry_count: dict = None
) -> Tuple[CreationValidation, Any, str]:
    """Process one iteration of blessings and create new artwork."""
    logger.debug(f"Processing {len(new_blessings)} blessings for session {session_id}...")
    
    if DEBUG:
        # Return mock blessing result with unique message ID
        logger.info(f"[DEBUG] Mock blessing iteration for session {session_id}")
        result = CreationValidation(
            result_url="https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg", 
            announcement="This is a test blessing response. Abraham created a new artwork inspired by the community's feedback.",
            error=None
        )
        message = ChatMessage(
            id=ObjectId(),
            role="assistant", 
            content="Creating new artwork based on blessings..."
        )
        logger.info(f"[DEBUG] Generated Abraham message ID: {str(message.id)}")
        return result, message
    
    # Use fallback model on second retry
    _retry_count = _retry_count or {'count': 0}
    _retry_count['count'] += 1
    if _retry_count['count'] == 2:
        model_name = fallback_model_name
        logger.info(f"Using fallback model: {fallback_model_name}")

    # Format blessings
    blessings = "\n".join([f"{m['author']} : {m['content']}" for m in new_blessings])
    
    # Create blessing response message
    bless_prompt = Template(BLESS_TEMPLATE)
    message = ChatMessageRequestInput(
        content=bless_prompt.render(blessings=blessings)
    )

    session = Session.from_mongo(session_id)
    
    # Create context
    context = PromptSessionContext(
        initiating_user_id=USER_ID,
        session=session,
        message=message,
        llm_config=LLMConfig(model=model_name)
    )
    
    # Add user message to session
    user_message = add_user_message(session, context)
    
    # Build LLM context
    context = await build_llm_context(
        session, 
        abraham, 
        context, 
        trace_id=str(uuid.uuid4()),
    )
    
    # Execute the prompt session
    async for _ in async_prompt_session(session, context, abraham):
        pass
        
    # Validate the new creation
    result = await validate_creation(session, model_name, fallback_model_name)

    # Use optimal image URL (webp if available, fallback to png)
    if result.result_url:
        result.result_url = get_optimal_image_url(result.result_url)
    
    return result, user_message



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.info(f"Retrying close_session (attempt {retry_state.attempt_number}/3)...")
)
async def close_session(
    session_id: str, 
    total_praises: int,
    total_blessings: int
) -> Tuple[CreationValidation, Any, str]:
    """Process one iteration of blessings and create new artwork."""
    logger.info(f"💭 Adding concluding remark for {session_id}...")
    
    if DEBUG:
        # Return mock blessing result with unique message ID
        logger.info(f"[DEBUG] Mock closing for session {session_id}")
        message = ChatMessage(
            id=ObjectId(), 
            role="assistant", 
            content=f"Thank you for joining me on this creative journey. With {total_praises} praises and {total_blessings} blessings, we explored new artistic territories together. Until we meet again in the realm of imagination."
        )
        logger.info(f"[DEBUG] Generated Abraham message ID: {str(message.id)}")
        return message
    
    # Create blessing response message
    close_prompt = Template(CLOSE_TEMPLATE)
    message = ChatMessageRequestInput(
        content=close_prompt.render(total_praises=total_praises, total_blessings=total_blessings)
    )

    session = Session.from_mongo(session_id)
    
    # Create context
    context = PromptSessionContext(
        initiating_user_id=USER_ID,
        session=session,
        message=message,
        llm_config=LLMConfig(model=MODEL_NAME)
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
    async for _ in async_prompt_session(session, context, abraham):
        pass
        
    assistant_message_id = session.get_messages()[-1]
    assistant_message = ChatMessage.from_mongo(assistant_message_id)

    return assistant_message


async def create_video(
    session_id: str,
    model_name: str,
    fallback_model_name: str,
    _retry_count: dict = None
) -> Session:
    """Create a session to generate artwork with specified model."""
    
    print("create vdo ", session_id)
    if DEBUG:
        # Return mock session
        logger.info(f"[DEBUG] Creating mock session for task output...")
        return Session(
            id=ObjectId(), 
            owner=ObjectId(USER_ID), 
            messages=[ObjectId(), ObjectId()]
        )
        
    # Use fallback model on second retry
    _retry_count = _retry_count or {'count': 0}
    _retry_count['count'] += 1
    if _retry_count['count'] == 2:
        model_name = fallback_model_name
        logger.info(f"Using fallback model: {fallback_model_name}")
    
    # Create session request
    request = PromptSessionRequest(
        user_id=USER_ID,
        creation_args=SessionCreationArgs(
            owner_id=USER_ID,
            agents=[str(abraham.id)],
            title=f"{get_date_prefix()} :: Creation Summary"
        )
    )
    
    # Setup session
    print("session_id", session_id)
    session = Session.from_mongo(session_id)
    
    # Create artwork generation message
    message = ChatMessageRequestInput(
        role="user",
        content=MIRACLE_INSTRUCTION
    )
    
    while True:
        # Create context with selected model
        context = PromptSessionContext(
            session=session,
            initiating_user_id=request.user_id,
            message=message,
            llm_config=LLMConfig(model=model_name)
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
        async for _ in async_prompt_session(session, context, abraham):
            pass

        # Check if the task is completed
        task_validation = await validate_video(
            session, 
            model_name, 
            fallback_model_name,
            _retry_count
        )
        if task_validation.completed:
            return task_validation
        else:
            message = ChatMessageRequestInput(
                role="user",
                content=f"You have not completed everything because of: {task_validation.error}. Do what you need to do to complete the task."
            )
        

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.info(f"Retrying validate_creation (attempt {retry_state.attempt_number}/3)...")
)
async def validate_video(
    session: Any,
    model_name: str,
    fallback_model_name: str,
    _retry_count: dict = None
) -> CreationValidation:
    """Validate if the full summary task was successful and extract results."""
    
    if DEBUG:
        # Return mock validation result
        logger.info("[DEBUG] Returning mock validation result")
        return TaskValidation(
            completed=True,
            poster_image_url="https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg",
            video_url="https://edenartlab-prod-data.s3.amazonaws.com/0fccfe42502244ec4cdf8beb50f9188adf62b20323b2481eb3bab1f9b7e15d7c.mp4",
            writeup="This is a test output from Abraham.",
            error=None
        )

    # Use fallback model on second retry
    _retry_count = _retry_count or {'count': 0}
    _retry_count['count'] += 1
    if _retry_count['count'] == 2:
        model_name = fallback_model_name
        logger.info(f"Using fallback model: {fallback_model_name}")

    # Prepare system message
    system_message = system_template.render(
        name=abraham.name,
        current_date_time=get_current_timestamp(),
        description=abraham.description,
        persona=abraham.persona,
        tools=None
    )
    
    # Get session messages
    # messages = [ChatMessage.from_mongo(m) for m in session.messages]
    messages = session.get_messages()
    
    # Build validation context
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message), 
            *messages,
            ChatMessage(role="user", content=TASK_VALIDATION_PROMPT)
        ],
        config=LLMConfig(
            model=model_name,
            response_format=TaskValidation
        )
    )
    
    # Get validation result
    response = await async_prompt(context)
    
    result = TaskValidation(**json.loads(response.content))

    # Use optimal image URL (webp if available, fallback to png)
    if result.poster_image_url:
        result.poster_image_url = get_optimal_image_url(result.poster_image_url)
    
    return result
