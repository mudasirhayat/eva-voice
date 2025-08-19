# v1

This document describes version 1 of the SesameAI TTS API.

## Endpoints

### `GET /health`

Checks the health and status of the API service.

**Summary:** Check API Health

**Responses:**

*   **`200 OK`**: Service is running. The response body provides more details:
    *   `{"status": "ok", "message": "TTS models and context loaded."}`: Everything loaded successfully.
    *   `{"status": "degraded", "message": "TTS model loaded, but context failed."}`: The main TTS model is loaded, but the voice context (from `VOICE_DIR`) could not be prepared. Synthesis requests will likely fail.
    *   `{"status": "error", "message": "TTS models failed to load during startup. Check logs."}`: Core models failed to load during startup. The service is unlikely to function.
*   **`503 Service Unavailable`**: (Potential future implementation) If critical components fail, the service might return 503 directly.

**Example Request:**

```bash
curl http://localhost:8000/health
```

### `POST /synthesize`

Synthesizes speech from the provided text using the pre-configured custom voice.

**Summary:** Synthesize Speech from Text

**Request Body:**

*   **Media Type:** `application/json`
*   **Schema:**
    ```json
    {
      "text": "string"
    }
    ```
    *   `text` (string, required): The text to synthesize into speech.

**Responses:**

*   **`200 OK`**: Successful synthesis.
    *   **Content-Type:** `audio/wav`
    *   **Body:** The raw WAV audio data.
*   **`400 Bad Request`**: Invalid input.
    *   **Content-Type:** `application/json`
    *   **Body:** `{"detail": "Input text cannot be empty."}`
*   **`503 Service Unavailable`**: The TTS service components (model or voice context) are not loaded or ready.
    *   **Content-Type:** `application/json`
    *   **Body:** `{"detail": "TTS service is not ready. Check startup logs."}`
*   **`500 Internal Server Error`**: An unexpected error occurred during synthesis.
    *   **Content-Type:** `application/json`
    *   **Body:** `{"detail": "Failed to synthesize speech: <error message>"}`

**Example Request:**

```bash
curl -X POST "http://localhost:8000/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world. This is a test of the Text to Speech API."}' \
     --output output.wav
```
This saves the resulting audio to `output.wav`.

# v2

Version 2 introduces API key authentication, LLM integration, and streaming audio output.

## Authentication

All v2 endpoints require API key authentication via the `X-API-Key` HTTP header. The API key must match the `API_KEY` environment variable configured on the server.

**Example of an authenticated request:**
```bash
curl -X POST "http://localhost:8000/endpoint" \
     -H "X-API-Key: your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your text here"}'
```

**Authentication Responses:**
*   **`401 Unauthorized`**: Missing or invalid API key.
    *   Missing key: `{"detail": "Missing X-API-Key header."}`
    *   Invalid key: `{"detail": "Invalid API Key."}`
*   **`503 Service Unavailable`**: Server's API key is not configured.
    *   `{"detail": "Server configuration error: API Key not set."}`

## Configuration

The API can be configured using environment variables:

*   **`API_KEY`**: (Required) The secret key for API authentication.
*   **`VOICE_DIR`**: (Required) Path to the directory containing voice reference audio files. The directory should contain WAV files that will be used as reference samples for the voice.
*   **`LLM_PROVIDER`**: (`local` or `groq`, default: `local`) The LLM provider to use.
*   **`LLM_MODEL`**: (For `local` provider, default: `Qwen/Qwen2.5-7B-Instruct-AWQ`) The Hugging Face model ID.
*   **`GROQ_MODEL`**: (For `groq` provider, default: `llama3-8b-8192`) The Groq model to use.
*   **`GROQ_API_KEY`**: (Required if `LLM_PROVIDER=groq`) Your Groq API key.
*   **`DEVICE`**: (`cuda` or `cpu`, default: `cuda` if available) Device for model inference.
*   **`WHISPER_MODEL`**: (default: `base`) Whisper model size for voice context transcription.

## Voice Directory Structure

The `VOICE_DIR` should contain:
- WAV files (supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`)
- Files will be automatically transcribed using Whisper ASR
- Transcriptions are cached in `_tts_context_cache.pt`
- Generated transcripts are saved in `metadata_auto_generated.csv`

Example directory structure:
```
VOICE_DIR/
├── voice1.wav
├── voice2.wav
├── voice3.wav
├── _tts_context_cache.pt      # Generated cache file
└── metadata_auto_generated.csv # Generated transcripts
```

## Endpoints

### `GET /health` (Enhanced)

The health check endpoint now includes LLM status in addition to TTS status.

**Responses:**
*   **`200 OK`**: Service status with detailed message:
    *   `{"status": "ok", "message": "TTS models, context, and LLM (local) loaded."}`: All components ready.
    *   `{"status": "degraded", "message": "TTS loaded, but LLM (groq) failed to load. Check logs."}`: Partial functionality.
    *   `{"status": "error", "message": "Core TTS and/or LLM components failed to load. Check logs."}`: Service impaired.

### `POST /generate_stream`

Receives a text prompt, processes it through the configured LLM, synthesizes the response using the custom voice, and streams the audio back to the client.

**Summary:** Generate Speech Stream from LLM Response

**Authentication:** Required (`X-API-Key` header)

**Request Body:**
*   **Media Type:** `application/json`
*   **Schema:**
    ```json
    {
      "prompt": "string"
    }
    ```
    *   `prompt` (string, required): The text prompt to send to the LLM.

**Responses:**
*   **`200 OK`**: Successful generation.
    *   **Content-Type:** `audio/wav`
    *   **Body:** A stream of WAV audio chunks (one per sentence).
    *   Note: The stream consists of concatenated WAV files, one per sentence.
*   **`400 Bad Request`**: Invalid input.
    *   `{"detail": "Input prompt cannot be empty."}`
*   **`401 Unauthorized`**: Authentication failed.
*   **`500 Internal Server Error`**: LLM or TTS processing error.
    *   `{"detail": "Failed to get response from LLM: <error message>"}`
*   **`503 Service Unavailable`**: Required components not loaded.
    *   `{"detail": "Service is not ready (LLM or TTS components failed to load). Check startup logs."}`

**Example Request:**
```bash
curl -X POST "http://localhost:8000/generate_stream" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_api_key_here" \
     -d '{"prompt": "Tell me a short story about a robot learning to play the guitar."}' \
     --output streamed_output.wav
```

**Notes:**
1.  The audio is streamed sentence by sentence, allowing for immediate playback as the content is generated.
2.  Each sentence is processed through the TTS model and streamed as a complete WAV chunk.
3.  The output stream is a concatenation of WAV files. While many audio players can handle this, it's not a single valid WAV file.
4.  For real-time playback, clients should either:
    *   Use an audio player that can handle streamed input
    *   Buffer and reconstruct the audio stream
    *   Process the WAV chunks individually 

# v3

Version 3 introduces configurable speech synthesis modes and enhanced conversation history management.

## New Configuration Options

Additional environment variables for v3:

*   **`MAX_HISTORY_EXCHANGES`**: (default: `10`) Maximum number of conversation exchanges to maintain in history.
*   **`STREAM_BY_SENTENCE`**: (`true` or `false`, default: `true`) Whether to process text sentence by sentence.

## Enhanced Endpoints

### `POST /synthesize` (Enhanced)

Now supports configurable speech synthesis modes.

**Request Body:**
*   **Media Type:** `application/json`
*   **Schema:**
    ```json
    {
      "text": "string",
      "stream_by_sentence": boolean  // optional
    }
    ```
    *   `text` (string, required): The text to synthesize into speech.
    *   `stream_by_sentence` (boolean, optional): Override server's synthesis mode.

### `POST /generate_stream` (Enhanced)

Now supports both configurable speech synthesis modes and conversation history.

**Request Body:**
*   **Media Type:** `application/json`
*   **Schema:**
    ```json
    {
      "prompt": "string",
      "stream_by_sentence": boolean  // optional
    }
    ```
    *   `prompt` (string, required): The text prompt to send to the LLM.
    *   `stream_by_sentence` (boolean, optional): Override server's synthesis mode.

## Speech Synthesis Modes

The API now supports two modes of text processing:

1. **Sentence-by-Sentence Mode** (`stream_by_sentence=true`):
   - Splits input text into sentences
   - Processes each sentence individually
   - Better for real-time streaming and natural pauses
   - Default mode
   - Recommended for longer texts
   - Enables immediate playback of early sentences

2. **Full Text Mode** (`stream_by_sentence=false`):
   - Processes entire text as one unit
   - Generates a single audio stream
   - Better prosody for short texts
   - Must wait for complete processing before playback

Configure the mode:
- Server-wide: Set `STREAM_BY_SENTENCE` environment variable
- Per-request: Include `stream_by_sentence` in request body

## Example Requests

1. Basic synthesis with default mode:
```bash
curl -X POST "http://localhost:8000/synthesize" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_api_key_here" \
     -d '{"text": "Hello world!"}' \
     --output output.wav
```

2. Synthesis with mode override:
```bash
curl -X POST "http://localhost:8000/synthesize" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_api_key_here" \
     -d '{
       "text": "Hello world!",
       "stream_by_sentence": false
     }' \
     --output output.wav
```

3. LLM generation with default mode:
```bash
curl -X POST "http://localhost:8000/generate_stream" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_api_key_here" \
     -d '{"prompt": "Tell me a story"}' \
     --output output.wav
```

4. LLM generation with mode override:
```bash
curl -X POST "http://localhost:8000/generate_stream" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_api_key_here" \
     -d '{
       "prompt": "Tell me a story",
       "stream_by_sentence": false
     }' \
     --output output.wav
```

## Best Practices

1. **Speech Synthesis Mode Selection:**
   - Use sentence-by-sentence mode for:
     - Long texts or responses
     - Real-time streaming requirements
     - When immediate playback is important
   - Use full text mode for:
     - Short phrases or single sentences
     - When prosody consistency is critical
     - When processing overhead should be minimized

2. **Conversation Management:**
   - History is maintained automatically
   - Limited to last `MAX_HISTORY_EXCHANGES` exchanges
   - Each exchange includes user prompt and assistant response
   - History is cleared on server restart

3. **Performance Considerations:**
   - Sentence-by-sentence mode has higher processing overhead
   - Full text mode may have longer initial latency
   - Consider response length when choosing modes
   - Monitor memory usage with large history settings

# Running the Server

## Prerequisites

1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Configure environment variables in `.env` file:
    ```env
    # Required
    API_KEY=your_secret_api_key_here
    VOICE_DIR=/path/to/your/voice/samples

    # Optional (shown with defaults)
    LLM_PROVIDER=local
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
    # If using Groq:
    # LLM_PROVIDER=groq
    # GROQ_API_KEY=your_groq_api_key
    # GROQ_MODEL=llama3-8b-8192
    ```

## Starting the Server

There are two ways to run the server:

### 1. Using Python Directly

```bash
python api.py
```
This runs the server in development mode with auto-reload enabled.

### 2. Using Uvicorn Directly (Recommended for Production)

```bash
# Basic usage
uvicorn api:app --host 0.0.0.0 --port 8000

# With reload for development
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Production settings (adjust workers based on your hardware)
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

**Note:** The server binds to `0.0.0.0` to be accessible from other machines. Use `127.0.0.1` or `localhost` for local-only access.

## Verifying the Server

Once running, you can verify the server is working:

1.  Check the health endpoint:
    ```bash
    curl http://localhost:8000/health
    ```

2.  Test the authenticated stream endpoint:
    ```bash
    curl -X POST "http://localhost:8000/generate_stream" \
         -H "X-API-Key: your_api_key_here" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Say hello!"}' \
         --output test.wav
    ```

## Common Issues

1.  **"API Key not configured"**: Ensure your `.env` file contains a valid `API_KEY`.
2.  **CUDA/GPU Issues**: If you encounter CUDA errors:
    *   Set `DEVICE=cpu` in your `.env` file
    *   Or ensure you have the correct CUDA version installed for your PyTorch version
3.  **Port Already in Use**: If port 8000 is taken:
    *   Use a different port: `--port 8001`
    *   Or find and stop the process using port 8000
4.  **Model Loading Failures**:
    *   Check you have sufficient RAM/VRAM
    *   Ensure all dependencies are correctly installed
    *   Check the logs for specific error messages 