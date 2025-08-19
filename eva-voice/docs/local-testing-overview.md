# Local Testing Overview

## Core Functionality

The project provides an interactive command-line interface (`runme.py`) to generate speech using a custom voice cloned from provided audio samples. It leverages the `sesameai` Text-to-Speech (TTS) library. Optionally, it can integrate with a Large Language Model (LLM) - either a local Hugging Face model or the Groq API - to generate conversational responses, which are then synthesized into speech using the custom voice.

## Key Components

1.  **`runme.py` (Main Script):**
    *   **Argument Parsing:** Handles command-line arguments for specifying the voice directory (`--voice-dir`), computation device (`--device`), output WAV file (`--output-wav`), LLM provider (`--llm-provider`), and specific LLM models (`--llm-model`, `--groq-model`).
    *   **TTS Initialization (`TTS` class):**
        *   Loads the `sesame/csm-1b` TTS model.
        *   Manages custom voice context:
            *   On the first run with a new `--voice-dir`, it uses the `openai-whisper` library (specifically the `base` model) to transcribe all audio files found in the directory.
            *   It then processes these transcriptions and corresponding audio tensors to create context tokens and masks required by the `sesameai` TTS model.
            *   This processed context is cached in a `_tts_context_cache.pt` file within the voice directory for much faster loading on subsequent runs. It also saves the generated transcripts to `metadata_auto_generated.csv`.
        *   Provides methods (`generate_audio_segment`, `say`) to synthesize text into audio segments using the loaded model and voice context. It splits input text into sentences and generates/plays them incrementally for lower latency.
        *   Patches `pydub`'s audio playback (`ffplay`) for better stability on Windows.
    *   **LLM Integration:**
        *   Loads environment variables (e.g., `GROQ_API_KEY` from `.env`).
        *   Initializes either `llm_handler.LLMHandler` (for local models) or `groq_handler.GroqHandler` based on the `--llm-provider` argument.
        *   If loading a local LLM fails, it gracefully falls back to a mode where it simply speaks the user's direct input instead of generating an LLM response.
    *   **Interactive Loop:**
        *   Prompts the user for text input.
        *   (If LLM enabled) Sends input to the LLM handler to get a response.
        *   Sends the LLM response (or the user's raw input in fallback mode) to the `TTS` instance's `say` method.
        *   The `say` method generates audio sentence by sentence, plays it back, and optionally saves the full combined audio to `combined_output.wav` (overwritten each time) or the file specified by `--output-wav`.

2.  **`llm_handler.py` (`LLMHandler` class):**
    *   Handles interaction with local Hugging Face `transformers` models.
    *   Uses `pipeline("text-generation", ...)` to load and run inference with specified models (defaulting to `Qwen/Qwen2.5-7B-Instruct-AWQ`).
    *   Requires `accelerate` and potentially specific quantization libraries (like `auto-awq`) depending on the model.
    *   Provides a `get_response` method to generate text based on user input and conversation history (though history management seems basic).

3.  **`groq_handler.py` (`GroqHandler` class):**
    *   Handles interaction with the Groq API.
    *   Requires the `GROQ_API_KEY` environment variable.
    *   Uses the `groq` Python client library.
    *   Provides a `get_response` method to generate text using a specified Groq model (defaulting to `llama3-8b-8192`). Manages conversation history for context.

4.  **`sesameai` library (External Dependency):**
    *   The core TTS engine responsible for voice cloning and speech synthesis. `runme.py` uses its `load_csm_1b` function and `generator` object.

5.  **`whisper` library (External Dependency):**
    *   Used within `runme.py` for automatically transcribing user-provided audio files to generate the text component needed for the TTS voice context.

6.  **Supporting Files:**
    *   `requirements.txt`: Lists Python dependencies (e.g., `torch`, `torchaudio`, `pydub`, `sesameai`, `openai-whisper`, `transformers`, `accelerate`, `groq`, `python-dotenv`).
    *   `README.md`: Provides setup instructions, usage examples, and explains the arguments.
    *   `CUSTOM_VOICE_README.md`: Details how to prepare audio files for custom voice cloning.
    *   `.env`: Used to store sensitive information like API keys (e.g., `GROQ_API_KEY`).
    *   `.gitignore`: Specifies files/directories to be ignored by Git.

## Workflow Summary

1.  **Setup:** Install dependencies (`requirements.txt`), potentially including specific PyTorch/Triton versions for Windows CUDA, and ensure FFmpeg is installed and in the system PATH. Set `GROQ_API_KEY` in `.env` if using Groq.
2.  **Prepare Voice:** Place reference `.wav` or `.mp3` files in a dedicated directory.
3.  **Run:** Execute `python runme.py --voice-dir path/to/voice/dir [optional LLM args]`.
4.  **First Run (Voice Dir):** The script transcribes audio using Whisper, generates TTS context, and caches it (`_tts_context_cache.pt`).
5.  **Load Models:** Loads the SesameAI TTS model and the selected LLM (or prepares the Groq client).
6.  **Interact:**
    *   User types a message.
    *   LLM (if active) generates a response.
    *   The response (or user input) is sent to the TTS engine.
    *   TTS generates speech using the custom voice context.
    *   Audio is played back incrementally.
    *   Full audio is saved to `combined_output.wav` or `--output-wav`.
7.  **Subsequent Runs:** Loading is faster as the voice context is loaded from the cache. 