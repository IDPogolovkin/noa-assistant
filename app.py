#
# app.py
#
# Noa assistant server application. Provides /mm endpoint.
#

import asyncio
from datetime import datetime
from io import BytesIO
import os
import traceback
from typing import Annotated, Dict, List, Tuple, Optional
import glob
import openai
import anthropic
import groq
import json
import base64
import aiohttp
from pydantic import BaseModel, ValidationError
from pydub import AudioSegment
from fastapi import FastAPI, status, Form, File, UploadFile, Request
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from models import Capability, TokenUsage, SearchAPI, VisionModel, GenerateImageService, MultimodalRequest, MultimodalResponse, ExtractLearnedContextRequest, ExtractLearnedContextResponse, Message
from web_search import WebSearch, DataForSEOWebSearch, SerpWebSearch, PerplexityWebSearch
from vision import Vision, GPT4Vision, ClaudeVision
from vision.utils import process_image
from generate_image import ReplicateGenerateImage
from assistant import Assistant, AssistantResponse, GPTAssistant, ClaudeAssistant, extract_learned_context, CustomModelAssistant

from langdetect import detect
from transliterate import translit as russian_translit
from qaznltk import qaznltk as qnltk 
from io import BytesIO
from pydub import AudioSegment
from dotenv import load_dotenv

# Initialize Kazakh transliterator
qn = qnltk.QazNLTK()

load_dotenv()

####################################################################################################
# Configuration
####################################################################################################

EXPERIMENT_AI_PORT = os.environ.get('EXPERIMENT_AI_PORT',8000)
gpt_key = os.getenv("OPENAI_API_KEY")
openai.api_key = gpt_key
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", None)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)

API_TOKEN = os.getenv("YANDEX_API_TOKEN")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")


####################################################################################################
# Server API 
####################################################################################################

app = FastAPI()

class Checker:
    def __init__(self, model: BaseModel):
        self.model = model

    def __call__(self, data: str = Form(...)):
        print(f"Received mm data in Checker: {data}")
        try:
            return self.model.model_validate_json(data)
        except ValidationError as e:
            print(f"Validation error in Checker: {e}")
            raise HTTPException(
                detail=jsonable_encoder(e.errors()),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        
async def transcribe(client, audio_bytes: bytes, prompt) -> str:
    # Create a file-like object for Whisper API to consume
    audio = AudioSegment.from_file(BytesIO(audio_bytes))
    buffer = BytesIO()
    buffer.name = "voice.mp4"
    audio.export(buffer, format="mp4")
    buffer.seek(0)
    # Whisper
    transcript = await client.audio.transcriptions.create(
        model="whisper-1",
        file=buffer,
        prompt=prompt
    )
    return transcript.text

def transliterate_text(text, lang):
    if lang == 'ru':
        return russian_translit(text, reversed=True)
    elif lang == 'kk':
        return qn.convert2latin_iso9(text)
    return text

async def generate_audio_async(text: str, model="tts-1", voice="alloy") -> bytes:
    """
    Generates audio from input text using OpenAI's TTS API.
    """
    client = app.state.openai_client

    # Create the speech audio
    response = await client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )

    # Stream the response content to a BytesIO buffer
    audio_buffer = BytesIO()
    for chunk in response.iter_bytes():
        audio_buffer.write(chunk)
    audio_buffer.seek(0)  # Reset the pointer to the beginning of the buffer

    return audio_buffer.getvalue()  # Return the audio bytes

def validate_assistant_model(model: str | None, models: List[str]) -> str:
    """
    Ensures a valid model is selected.

    Parameters
    ----------
    model : str | None
        Model name to use.
    models : List[str]
        List of valid models. The first model is the default model.

    Returns
    -------
    str
        If the model name is in the list, returns it as-is, otherwise returns the first model in the
        list by default.
    """
    if model is None or model not in models:
        return models[0]
    return model

def get_assistant(app, mm: MultimodalRequest) -> Tuple[Assistant, str | None]:
    assistant_model = mm.assistant_model

    # Default assistant if none selected
    # if mm.assistant is None or (mm.assistant not in [ "gpt", "claude", "groq", "egov"]):
    #     return app.state.assistant, None    # None for assistant_model will force assistant to use its own internal default choice

    # Set default assistant to 'egov' if not provided
    if not mm.assistant:
        mm.assistant = 'egov'
    
    # Return assistant and a valid model for it
    if mm.assistant == "egov":
        return CustomModelAssistant(), None  # None for assistant_model will force assistant to use its own internal default choice
    elif mm.assistant == "gpt":
        assistant_model = validate_assistant_model(model=mm.assistant_model, models=[ "gpt-4o", "gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview", "gpt-4-1106-preview" ])
        if mm.openai_key and len(mm.openai_key) > 0:
            return GPTAssistant(client=openai.AsyncOpenAI(api_key=mm.openai_key)), assistant_model
        return GPTAssistant(client=app.state.openai_client), assistant_model
    elif mm.assistant == "claude":
        assistant_model = validate_assistant_model(model=mm.assistant_model, models=[ "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229" ])
        return ClaudeAssistant(client=app.state.anthropic_client), assistant_model
    elif mm.assistant == "groq":
        assistant_model = validate_assistant_model(model=mm.assistant_model, models=[ "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it" ])
        return GPTAssistant(client=app.state.groq_client), assistant_model # Groq uses GPTAssistant
    
    # Should never fall through to here
    return None, ""

def get_web_search_provider(app, mm: MultimodalRequest) -> WebSearch:
    # Use provider specified in request options
    if mm.search_api == SearchAPI.SERP:
        return SerpWebSearch(save_to_file=options.save, engine=mm.search_engine.value, max_search_results=mm.max_search_results)
    elif mm.search_api == SearchAPI.DATAFORSEO:
        return DataForSEOWebSearch(save_to_file=options.save, max_search_results=mm.max_search_results)
    elif mm.search_api == SearchAPI.PERPLEXITY:
        if mm.perplexity_key and len(mm.perplexity_key) > 0:
            return PerplexityWebSearch(api_key=mm.perplexity_key)
        return PerplexityWebSearch(api_key=PERPLEXITY_API_KEY)

    # Default provider
    return app.state.web_search

def get_vision_provider(app, mm: MultimodalRequest) -> Vision | None:
    # Use provider specified 
    if mm.vision in [VisionModel.GPT4O, VisionModel.GPT4Vision ]:
        return GPT4Vision(client=app.state.openai_client, model=mm.vision)
    elif mm.vision in [VisionModel.CLAUDE_HAIKU, VisionModel.CLAUDE_SONNET, VisionModel.CLAUDE_OPUS]:
        return ClaudeVision(client=app.state.anthropic_client, model=mm.vision)
    
    # Default provider
    return app.state.vision

@app.get('/health')
async def api_health():
    return {"status":200,"message":"running ok"}

MAX_FILES = 100
AUDIO_DIR = "audio"

def get_next_filename():
    existing_files = sorted(glob.glob(f"{AUDIO_DIR}/audio*.wav"))
    # if audio directory does not exist, create it
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
    if len(existing_files) < MAX_FILES:
        return f"{AUDIO_DIR}/audio{len(existing_files) + 1}.wav"
    else:
        # All files exist, so find the oldest one to overwrite
        oldest_file = min(existing_files, key=os.path.getmtime)
        return oldest_file
    
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"Validation error: {exc}")
    print(f"Request URL: {request.url}")
    print(f"Request headers: {request.headers}")

    # Try to get the form data
    try:
        form = await request.form()
        print("Received form data:")
        for field in form:
            value = form[field]
            if isinstance(value, UploadFile):
                print(f"{field}: {value.filename} ({value.content_type})")
            else:
                print(f"{field}: {value}")
    except Exception as e:
        print(f"Error reading form data: {e}")

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.post("/mm")
async def api_mm(
    request: Request,
    mm: Annotated[Optional[str], Form()] = None,
    messages: Annotated[Optional[str], Form()] = None,
    location: Annotated[Optional[str], Form()] = None,
    time: Annotated[Optional[str], Form()] = None,
    prompt: Annotated[Optional[str], Form()] = None,
    assistant: Annotated[Optional[str], Form()] = None,
    gps: Annotated[Optional[str], Form()] = None,
    audio: UploadFile = None,
    image: UploadFile = None
):
    try:
        form = await request.form()
        print("Received form data:")
        for field in form:
            value = form[field]
            if isinstance(value, UploadFile):
                print(f"{field}: {value.filename} ({value.content_type})")
            else:
                print(f"{field}: {value}")
        if mm:
            # If 'mm' field is provided, parse it directly
            mm_data = json.loads(mm)
            mm = MultimodalRequest(**mm_data)
        else:
            # Construct the mm object from the form data
            mm_dict = {
                'messages': json.loads(messages) if messages else [],
                'address': location,
                'local_time': time,
                'prompt': prompt,
                'assistant': assistant or 'egov',  # Default to 'egov'
                'gps': json.loads(gps) if gps else None,
                # Include other fields if necessary
            }
            mm = MultimodalRequest(**mm_dict)
        print(f"Constructed mm object: {mm}")

        # Transcribe voice prompt if it exists
        voice_prompt = ""
        if audio:
            audio_bytes = await audio.read()
            if mm.testing_mode:
                # Save audio file
                filepath = get_next_filename()
                with open(filepath, "wb") as f:
                    f.write(audio_bytes)
            if mm.openai_key and len(mm.openai_key) > 0:
                client = openai.AsyncOpenAI(api_key=openai.api_key)
            else:
                # Initialize your OpenAI client here
                client = openai.AsyncOpenAI(api_key=openai.api_key)
            voice_prompt = await transcribe(client=client, audio_bytes=audio_bytes, prompt="")

        # Construct final prompt
        if not mm.prompt or mm.prompt.strip() == "":
            mm.prompt = voice_prompt  # Set the prompt to the transcribed audio
            user_prompt = f"limit answer to 20 words {voice_prompt}"
        else:
            user_prompt = f"limit answer to 20 words {mm.prompt} {voice_prompt}"
            # Set mm.prompt to user_prompt to ensure it's not None

        mm.prompt = f"limit answer to 20 words {user_prompt}"

        print(f"Final user_prompt: {user_prompt}")
        # **Add this validation**
        # if not user_prompt:
        #     raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        if not mm.messages:
            mm.messages = [Message(role='user', content=mm.prompt)]

        # Image data
        image_bytes = (await image.read()) if image else None
        # preprocess image
        if image_bytes:
            image_bytes = process_image(image_bytes)
        # Location data
        address = mm.address

        # User's local time
        local_time = mm.local_time

        # Image generation (bypasses assistant altogether)
        if mm.generate_image != 0:
            if mm.generate_image_service == GenerateImageService.REPLICATE:
                generate_image = ReplicateGenerateImage()
                image_url = await generate_image.generate_image(
                    query=user_prompt,
                    use_image=True,
                    image_bytes=image_bytes
                )
                return MultimodalResponse(
                    user_prompt=user_prompt,
                    response="",
                    image=image_url,
                    token_usage_by_model={},
                    capabilities_used=[Capability.IMAGE_GENERATION],
                    total_tokens=0,
                    input_tokens=0,
                    output_tokens=0,
                    timings="",
                    debug_tools=""
                )

        # Get assistant tool providers
        web_search: WebSearch = get_web_search_provider(app=request.app, mm=mm)
        vision: Vision = get_vision_provider(app=request.app, mm=mm)
        
        # Call the assistant and deliver the response
        try:
            assistant, assistant_model = get_assistant(app=app, mm=mm)
            assistant_response: AssistantResponse = await assistant.send_to_assistant(
                prompt=user_prompt,
                noa_system_prompt=mm.noa_system_prompt,
                image_bytes=image_bytes,
                message_history=mm.messages,
                learned_context={},
                local_time=local_time,
                location_address=address,
                model=assistant_model,
                web_search=web_search,
                vision=vision,
                speculative_vision=mm.speculative_vision
            )

            print(f"Assistant_response {assistant_response}")

            # Detect the language of the assistant's response
            language = detect(assistant_response.response)
            print(f"Detected language: {language}")

            # Transliterate the response text if necessary
            display_text = transliterate_text(assistant_response.response, language)
            print(f"Display text: {display_text}")

            # Generate audio using OpenAI's TTS API
            audio_data = await generate_audio_async(assistant_response.response)
            if audio_data:
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            else:
                audio_base64 = None

            # Create the debug dict
            debug_data = {
                "topic_changed": assistant_response.topic_changed,
                "timings": assistant_response.timings,
                "debug_tools": assistant_response.debug_tools
            }

            response_data = MultimodalResponse(
                user_prompt=user_prompt,
                response=display_text,  # Use the transliterated text for display
                message=display_text,
                image=assistant_response.image,
                audio=audio_base64,
                token_usage_by_model=assistant_response.token_usage_by_model,
                capabilities_used=assistant_response.capabilities_used,
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                debug=debug_data
            )

            # Log the response data
            print(f"Response data being sent to client: {response_data.model_dump_json()}")

            return response_data
        
        except Exception as e:
            print(f"{traceback.format_exc()}")
            raise HTTPException(400, detail=f"===RESPONSE ERROR==={str(e)}: {traceback.format_exc()}")

    except Exception as e:
        print(f"{traceback.format_exc()}")
        raise HTTPException(400, detail=f"{str(e)}: {traceback.format_exc()}")
    
async def translate_text(text, target_language):
    url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {API_TOKEN}"
    }
    body = {
        "folderId": FOLDER_ID,
        "texts": [text],
        "targetLanguageCode": target_language
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Error translating text: {await response.text()}")
            result = await response.json()
            return result['translations'][0]['text']
    
@app.post("/translator")
async def translator_endpoint(
    audio: Optional[UploadFile] = File(None),
    source_text: Optional[str] = Form(None),
    target_language: Optional[str] = Form("ru")  # default to Russian
):
    """
    A dedicated Translator endpoint returning a MultimodalResponse, like /mm.
    
    1) Transcribe audio if present.
    2) Combine transcription + source_text for a final 'transcribed_text'.
    3) Translate into `target_language`.
    4) Transliterate if RU/KK.
    5) Generate TTS audio (base64).
    6) Return a MultimodalResponse object.
    """
    try:
        transcribed_text = ""

        # -- (1) Transcribe audio if present --
        if audio:
            audio_bytes = await audio.read()
            # Reuse openai.AsyncOpenAI from your app.state
            client = app.state.openai_client
            transcribed_text = await transcribe(client=client, audio_bytes=audio_bytes, prompt="Бұл мәтін қазақ тілінде болады.")
            print(f"Transcribed text: {transcribed_text}")

        # -- (2) Combine user-provided text + transcribed_text --
        if source_text:
            if transcribed_text.strip():
                transcribed_text += f" {source_text}"
            else:
                transcribed_text = source_text

        if not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="No text to translate (empty audio transcription or source_text).")

        # -- (3) Translate text into target_language using Yandex --
        translated_text = await translate_text(text=transcribed_text, target_language=target_language)
        print(f"Translated text: {translated_text}")

        # -- (4) Detect language, transliterate if RU/KK --
        language = detect(translated_text)
        print(f"Detected language of translation: {language}")
        supported_languages = ['ru', 'kk', 'en', 'uk']

        if language in supported_languages:
            display_text = transliterate_text(translated_text, language)
        else:
            display_text = translated_text

        print(f"Final display text (transliterated if RU/KK): {display_text}")

        # -- (5) Generate TTS audio from the final translation --
        audio_data = await generate_audio_async(translated_text)
        audio_base64 = base64.b64encode(audio_data).decode("utf-8") if audio_data else None

        # We can store debug info in a dict (like in /mm)
        debug_data = {
            "step": "translator_endpoint",
            "target_language": target_language
        }

        # -- (6) Return a MultimodalResponse object --
        # We do not show an image, so `image=None`.
        # We'll reuse "transcribed_text" in user_prompt, 
        # and store the final "display_text" in 'response' and 'message'.
        response_data = MultimodalResponse(
            user_prompt=transcribed_text,
            response=display_text,
            message=display_text,
            image=None,
            audio=audio_base64,
            token_usage_by_model={},
            capabilities_used=[],  # add any relevant capabilities
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            debug=debug_data
        )

        # Log the response
        print(f"Translator Response data: {response_data.model_dump_json()}")
        return response_data

    except Exception as e:
        print(f"{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"===RESPONSE ERROR=== {str(e)}: {traceback.format_exc()}")

@app.post("/extract_learned_context")
async def api_extract_learned_context(request: Request, params: Annotated[str, Form()]):
    try:
        params: ExtractLearnedContextRequest = Checker(ExtractLearnedContextRequest)(data=params)
        print(params)

        token_usage_by_model: Dict[str, TokenUsage] = {}

        # Perform extraction
        try:
            learned_context = await extract_learned_context(
                client=request.app.state.openai_client,
                message_history=params.messages,
                model="gpt-3.5-turbo-1106",
                existing_learned_context=params.existing_learned_context,
                token_usage_by_model=token_usage_by_model
            )

            return ExtractLearnedContextResponse(
                learned_context=learned_context,
                token_usage_by_model=token_usage_by_model
            )
        except Exception as e:
            print(f"{traceback.format_exc()}")
            raise HTTPException(400, detail=f"{str(e)}: {traceback.format_exc()}")

    except Exception as e:
        print(f"{traceback.format_exc()}")
        raise HTTPException(400, detail=f"{str(e)}: {traceback.format_exc()}")


####################################################################################################
# Program Entry Point
####################################################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="store", help="Perform search query and exit")
    parser.add_argument("--location", action="store", default="San Francisco", help="Set location address used for all queries (e.g., \"San Francisco\")")
    parser.add_argument("--save", action="store", help="Save DataForSEO response object to file")
    parser.add_argument("--search-api", action="store", default="perplexity", help="Search API to use (perplexity, serp, dataforseo)")
    parser.add_argument("--assistant", action="store", default="egov", help="Assistant to use (gpt, claude, groq)")
    parser.add_argument("--server", action="store_true", help="Start server")
    parser.add_argument("--image", action="store", help="Image filepath for image query")
    parser.add_argument("--vision", action="store", help="Vision model to use (gpt-4o, gpt-4-vision-preview, claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229)", default="gpt-4o")
    options = parser.parse_args()

    # AI clients
    app.state.openai_client = openai.AsyncOpenAI(api_key=openai.api_key)
    app.state.anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    app.state.groq_client = groq.AsyncGroq(api_key="123123")

    # Instantiate a default web search provider
    app.state.web_search = None
    if options.search_api == "serp":
        app.state.web_search = SerpWebSearch(save_to_file=options.save, engine="google")
    elif options.search_api == "dataforseo":
        app.state.web_search = DataForSEOWebSearch(save_to_file=options.save)
    elif options.search_api == "perplexity":
        app.state.web_search = PerplexityWebSearch(api_key=PERPLEXITY_API_KEY)
    else:
        raise ValueError("--search-api must be one of: serp, dataforseo, perplexity")

    # Instantiate a default vision provider
    app.state.vision = None
    if options.vision in [ "gpt-4o", "gpt-4-vision-preview" ]:
        app.state.vision = GPT4Vision(client=app.state.openai_client, model=options.vision)
    elif VisionModel(options.vision) in [VisionModel.CLAUDE_HAIKU, VisionModel.CLAUDE_SONNET, VisionModel.CLAUDE_OPUS]:
        app.state.vision = ClaudeVision(client=app.state.anthropic_client, model=options.vision)
    else:
        raise ValueError("--vision must be one of: gpt-4o, gpt-4-vision-preview, claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229")

    # Instantiate a default assistant
    if options.assistant == "egov":
        app.state.assistant = CustomModelAssistant()
    elif options.assistant == "gpt":
        app.state.assistant = GPTAssistant(client=app.state.openai_client)
    elif options.assistant == "claude":
        app.state.assistant = ClaudeAssistant(client=app.state.anthropic_client)
    elif options.assistant == "groq":
        app.state.assistant = GPTAssistant(client=app.state.groq_client)
    else:
        raise ValueError("--assistant must be one of: gpt, claude, groq, egov")

    # Load image if one was specified (for performing a test query)
    image_bytes = None
    if options.image:
        with open(file=options.image, mode="rb") as fp:
            image_bytes = fp.read()

    # Test query
    if options.query:
        async def run_query() -> str:
            return await app.state.assistant.send_to_assistant(
                prompt=options.query,
                image_bytes=image_bytes,
                message_history=[],
                learned_context={},
                local_time=datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),  # e.g., Friday, March 8, 2024, 11:54 AM
                location_address=options.location,
                model=None,
                web_search=app.state.web_search,
                vision=app.state.vision,

            )
        response = asyncio.run(run_query())
        print(response)

    # Run server
    if options.server:
        import uvicorn

        app.state.assistant = CustomModelAssistant()  # Set default assistant

        uvicorn.run(app, host="0.0.0.0", port=int(EXPERIMENT_AI_PORT), log_level="debug")