import logging
import os
import warnings
from contextlib import asynccontextmanager
import io
import time
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Request, Body, Depends
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import asyncio
from pydub import AudioSegment

# Import TTS from runme.py - this is the key import that gives us all the TTS functionality
from runme import TTS

# Import LLM Handlers
from llm_handler import LLMHandler
from groq_handler import GroqHandler

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
import logging
import os
import torch
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
API_KEY = os.environ.get("API_KEY")
VOICE_DIR = os.environ.get("VOICE_DIR")

# LLM Configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "local").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
MAX_HISTORY_EXCHANGES = int(os.environ.get("MAX_HISTORY_EXCHANGES", "10"))

# TTS Configuration
STREAM_BY_SENTENCE = os.environ.get("STREAM_BY_SENTENCE", "true").lower() == "true"

if not API_KEY:
    logger.warning("Security Warning: API_KEY environment variable not set. Authentication will fail.")

if not VOICE_DIR:
    logger.warning("Warning: VOICE_DIR environment variable not set. Voice samples must be configured.")

# FastAPI Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("API Startup: Loading models...")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Voice directory: {VOICE_DIR}")

    try:
        # Initialize TTS from runme.py with voice_dir
        app.state.tts = TTS(device=DEVICE, voice_dir=VOICE_DIR)
        app.state.tts.load_model()
        logger.info("TTS model loaded successfully!")
    except Exception as e:
        logger.exception(f"Fatal Error: Failed to load TTS model: {e}")
        app.state.tts = None

    # Initialize LLM
    app.state.llm = None
    logger.info(f"Configuring LLM Provider: {LLM_PROVIDER}")

    if LLM_PROVIDER == "local":
        logger.info(f"Loading local LLM model: {LLM_MODEL}")
        try:
            app.state.llm = LLMHandler(model_id=LLM_MODEL, device=DEVICE)
            logger.info("Local LLM handler loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load local LLM model {LLM_MODEL}: {e}")
    elif LLM_PROVIDER == "groq":
        logger.info(f"Configuring Groq LLM model: {GROQ_MODEL}")
        try:
            app.state.llm = GroqHandler(
                model_name=GROQ_MODEL,
                max_history_exchanges=MAX_HISTORY_EXCHANGES
            )
            logger.info("Groq LLM handler configured successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize Groq client for model {GROQ_MODEL}: {e}")
    else:
        logger.warning(f"Invalid LLM_PROVIDER specified: '{LLM_PROVIDER}'. No LLM will be loaded.")

    yield

    # Shutdown
    logger.info("API Shutdown: Cleaning up resources...")
    app.state.tts = None
    app.state.llm = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    logger.info("Resources cleaned up.")

# Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(key: str = Depends(api_key_header)):
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server configuration error: API Key not set."
        )
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_NAME} header."
        )
    if key != API_KEY:
try:
    # existing code
except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key."
    )
        )
    return True

# Request Models
class TTSRequest(BaseModel):
    text: str
    stream_by_sentence: bool = None  # None means use server default

class LLMRequest(BaseModel):
    prompt: str
    stream_by_sentence: bool = None  # None means use server default

# FastAPI Application
app = FastAPI(
    title="SesameAI TTS API",
    description="API for generating speech using SesameAI TTS and LLM.",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Check if the API, TTS model, and LLM are healthy."""
    tts_ready = app.state.tts is not None
    llm_ready = app.state.llm is not None

    if tts_ready and llm_ready:
        return {
            "status": "ok",
            "message": f"TTS and LLM ({LLM_PROVIDER}) ready"
        }
    elif tts_ready:
        return {
            "status": "degraded",
            "message": "TTS ready, LLM not loaded"
        }
    elif llm_ready:
        return {
            "status": "degraded",
            "message": "LLM ready, TTS not loaded"
        }
    else:
        return {
            "status": "error",
            "message": "Neither TTS nor LLM loaded"
        }

@app.post(
    "/synthesize",
    response_class=Response,
    dependencies=[Depends(get_api_key)]
)
async def synthesize_speech(request: Request, payload: TTSRequest):
    """Synthesize speech from text using TTS."""
    if not request.app.state.tts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS service is not ready."
        )

    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty."
        )

    try:
        # Use request-specific setting if provided, otherwise use server default
        stream_by_sentence = payload.stream_by_sentence if payload.stream_by_sentence is not None else STREAM_BY_SENTENCE
        
        if stream_by_sentence:
            # Split text into sentences and process each
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Process all sentences and concatenate
            audio_segments = []
            for sentence in sentences:
                segment = request.app.state.tts.generate_audio_segment(sentence)
                audio_segments.append(segment)
            
            # Concatenate all segments
            if len(audio_segments) > 1:
                audio_segment = audio_segments[0]
                for segment in audio_segments[1:]:
                    audio_segment += segment
            else:
                audio_segment = audio_segments[0]
        else:
            # Process entire text at once
            audio_segment = request.app.state.tts.generate_audio_segment(text)
        
        # Export to WAV
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        logger.exception(f"Error during speech synthesis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to synthesize speech: {e}"
        )

@app.post(
    "/synthesize/stream",
    response_class=StreamingResponse,
    dependencies=[Depends(get_api_key)]
)
async def synthesize_speech_stream(request: Request, payload: TTSRequest):
    """Stream synthesized speech from text using TTS."""
    if not request.app.state.tts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS service is not ready."
        )

    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty."
        )

    try:
        stream_by_sentence = payload.stream_by_sentence if payload.stream_by_sentence is not None else STREAM_BY_SENTENCE

async def audio_stream_generator():
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    for sentence in sentences:
                    logger.info(f"Streaming audio for sentence: '{sentence}'")
                    async for audio_chunk in request.app.state.tts.generate_streaming(
                        sentence,
                        chunk_size=20
                    ):
                        # Convert chunk to WAV format
                        buffer = io.BytesIO()
                        audio_chunk = audio_chunk.to(torch.float32)
                        audio_segment = AudioSegment(
                            (audio_chunk.cpu().numpy() * 32767).astype("int16").tobytes(),
                            frame_rate=request.app.state.tts.generator.sample_rate,
                            sample_width=2,
                            channels=1
                        )
                        audio_segment.export(buffer, format="wav")
                        buffer.seek(0)
                        yield buffer.read()
                        await asyncio.sleep(0.005)  # Reduced sleep time to minimize gaps
            except Exception as e:
                logger.error(f"Error in audio stream generation: {e}")
                raise

        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
            headers={"X-Stream-Type": "sentence" if stream_by_sentence else "complete"}
        )

    except Exception as e:
        logger.exception(f"Error during speech synthesis streaming: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to synthesize speech stream: {e}"
        )

@app.post(
    "/generate_stream",
response_class=StreamingResponse,
    dependencies=[Depends(get_api_key)])
async def generate_stream(request: Request, payload: LLMRequest):
    try:
        # Existing code
    except Exception as e:
        raise HTTPException(status_code=
    """Generate LLM response and stream speech."""
    if not request.app.state.tts or not request.app.state.llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS or LLM service is not ready."
        )

    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input prompt cannot be empty."
        )

    # Get LLM Response
    try:
        logger.info("Getting response from LLM...")
        t_llm_start = time.time()
        llm_response = request.app.state.llm.get_response(prompt)
        t_llm_end = time.time()
        logger.info(f"LLM response received in {t_llm_end - t_llm_start:.2f}s: '{llm_response[:100]}...'")

        if not llm_response or not llm_response.strip():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM returned empty response"
            )
    except Exception as e:
        logger.exception(f"Error getting LLM response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM response: {e}"
        )

    # Use request-specific setting if provided, otherwise use server default
    stream_by_sentence = payload.stream_by_sentence if payload.stream_by_sentence is not None else STREAM_BY_SENTENCE

    async def audio_stream_generator():
        try:
            if stream_by_sentence:
                # Split text into sentences
                sentences = re.split(r'(?<=[.!?])\s+', llm_response)
                sentences = [s.strip() for s in sentences if s.strip()]

                for sentence in sentences:
                    # Generate audio segment using runme.py's TTS
                    audio_segment = request.app.state.tts.generate_audio_segment(sentence)
                    
                    # Export to WAV and stream
                    buffer = io.BytesIO()
                    audio_segment.export(buffer, format="wav")
                    buffer.seek(0)
                    yield buffer.read()
            else:
                # Process entire response at once
                audio_segment = request.app.state.tts.generate_audio_segment(llm_response)
                
                # Export to WAV and stream
                buffer = io.BytesIO()
                audio_segment.export(buffer, format="wav")
                buffer.seek(0)
                yield buffer.read()

        except Exception as e:
            logger.error(f"Error in audio generation: {e}")
            raise

    return StreamingResponse(audio_stream_generator(), media_type="audio/wav")

@app.post(
    "/generate/stream",
    response_class=StreamingResponse,
    dependencies=[Depends(get_api_key)]
)
async def generate_llm_stream(request: Request, payload: LLMRequest):
    """Generate LLM response and stream speech."""
    if not request.app.state.tts or not request.app.state.llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    try:
        detail="TTS or LLM service is not ready."
        prompt = payload.prompt.strip()
        if not prompt:
            raise ValueError("Prompt is empty.")
    except Exception as e:
        print(f"Error
try:
    # Original code here
except HTTPException as e:
    raise HTTPException(status_code=400, detail="Bad Request")
except Exception as e:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail
        )

    try:
        logger.info("Getting response from LLM...")
        t_llm_start = time.time()
        llm_response = request.app.state.llm.get_response(prompt)
        t_llm_end = time.time()
        logger.info(f"LLM response received in {t_llm_end - t_llm_start:.2f}s: '{llm_response[:100]}...'")

if not llm_response or not llm_response.strip():
    raise HTTPException(status_code=400, detail="Invalid LLM response")
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM returned empty response"
            )

        stream_by_sentence = payload.stream_by_sentence if payload.stream_by_sentence is not None else STREAM_BY_SENTENCE

        async def audio_stream_generator():
            try:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', llm_response) if s.strip()]
                for sentence in sentences:
                    logger.info(f"Streaming audio for LLM response sentence: '{sentence}'")
                    async for audio_chunk in request.app.state.tts.generate_streaming(
                        sentence,
                        chunk_size=20
                    ):
                        # Convert chunk to WAV format
                        buffer = io.BytesIO()
                        audio_chunk = audio_chunk.to(torch.float32)
                        audio_segment = AudioSegment(
                            (audio_chunk.cpu().numpy() * 32767).astype("int16").tobytes(),
                            frame_rate=request.app.state.tts.generator.sample_rate,
                            sample_width=2,
                            channels=1
                        )
                        audio_segment.export(buffer, format="wav")
                        buffer.seek(0)
                        yield buffer.read()
                        await asyncio.sleep(0.005)  # Reduced sleep time to minimize gaps

            except Exception as e:
                logger.error(f"Error in audio stream generation: {e}")
                raise

        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
            headers={
                "X-Stream-Type": "sentence" if stream_by_sentence else "complete",
                "X-LLM-Provider": LLM_PROVIDER
            }
        )

    except Exception as e:
        logger.exception(f"Error during LLM speech streaming: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate speech stream: {e}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 