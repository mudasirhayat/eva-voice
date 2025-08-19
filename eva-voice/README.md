# SesameAI TTS API

A FastAPI-based Text-to-Speech API using SesameAI's models, with support for custom voice samples and LLM integration.

## Features

- Text-to-Speech synthesis using SesameAI models
- Custom voice sample support with automatic transcription
- LLM integration (local models or Groq API)
- Sentence-by-sentence or full-text processing
- Real-time audio streaming
- GPU acceleration support
- Docker containerization

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Python 3.11 (for local development)
- Hugging Face account and access token

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd sesameai-tts
```

2. Create a `.env` file:
```bash
# Required
API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token_here  # From https://huggingface.co/settings/tokens

# Optional
VOICE_DIR=/path/to/your/voices  # defaults to ./voices
LLM_PROVIDER=local             # 'local' or 'groq'
GROQ_API_KEY=your_groq_key    # Required if using Groq
NO_TORCH_COMPILE=1            # Prevents torch compilation issues
```

3. Prepare your voice samples:
```bash
mkdir -p voices
# Copy your voice samples (WAV, MP3, FLAC, OGG, M4A) to the voices directory
```

4. Build and run with Docker Compose:
```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## Voice Sample Configuration

The API supports two ways of providing voice samples:

1. **Default Samples**: Uses predefined samples from `samples.py`
2. **Custom Directory**: Uses your own voice samples from a specified directory

When using a custom directory:
- Supported formats: WAV, MP3, FLAC, OGG, M4A
- Files are automatically transcribed using Whisper ASR
- Transcriptions and processed data are cached for faster subsequent loads

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | Authentication key for the API | Required |
| `VOICE_DIR` | Path to voice samples directory | `./voices` |
| `DEVICE` | Device for model inference | `cuda` |
| `LLM_PROVIDER` | LLM provider to use | `local` |
| `LLM_MODEL` | Local LLM model ID | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| `GROQ_MODEL` | Groq model to use | `llama3-8b-8192` |
| `GROQ_API_KEY` | Groq API key | Required if using Groq |
| `STREAM_BY_SENTENCE` | Process text sentence by sentence | `true` |

## API Endpoints

### Health Check
```http
GET /health
```
Returns the status of TTS and LLM services.

### Text-to-Speech
```http
POST /synthesize
Content-Type: application/json
X-API-Key: your_api_key

{
    "text": "Text to synthesize",
    "stream_by_sentence": true  // optional
}
```
Generates speech from text.

### LLM with TTS
```http
POST /generate_stream
Content-Type: application/json
X-API-Key: your_api_key

{
    "prompt": "Your prompt here",
    "stream_by_sentence": true  // optional
}
```
Generates LLM response and converts it to speech.

## Development Setup

For local development without Docker:

1. Ensure Python 3.11 is installed:
```bash
python3.11 --version
```

2. Set up environment variables:
```bash
export NO_TORCH_COMPILE=1
export HF_TOKEN=your_huggingface_token_here
```

3. Log in to Hugging Face:
```bash
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the API:
```bash
uvicorn api:app --reload
```

6. For CLI usage:
```bash
python runme.py --voice-dir /path/to/voices
```

## Troubleshooting

1. **GPU Issues**:
   - Ensure NVIDIA drivers are installed
   - Verify NVIDIA Container Toolkit is installed
   - Check `nvidia-smi` works on the host

2. **Voice Sample Issues**:
   - Verify audio file formats are supported
   - Check file permissions
   - Look for transcription errors in logs

3. **LLM Issues**:
   - For local models, ensure sufficient GPU memory
   - For Groq, verify API key and connectivity
   - Check model compatibility

## License

[Your License Here]

## Contributing

[Your Contribution Guidelines Here]
