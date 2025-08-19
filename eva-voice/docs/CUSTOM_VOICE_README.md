# Custom Voice Context for SesameAI TTS Runner

This document explains how to prepare and use custom voice reference audio with the `runme.py` script, which now features automatic transcription and caching.

## Overview

The script uses reference audio samples to guide the TTS model in generating speech with a specific voice characteristic. Thanks to automatic transcription via OpenAI Whisper, you only need to provide the audio files for your custom voice.

## Preparing the Voice Directory

To use a specific voice, you simply need to:

1.  Create a directory for your voice (e.g., `voices/my_voice`).
2.  Place your reference audio files (e.g., `.wav`, `.mp3`, `.flac`) directly inside this directory.

**Example Directory Structure:**

```
my_custom_voice/
├── sample_01.wav
├── sample_02.wav
├── sample_03.flac
└── another_sample.mp3
```

**Tips for Reference Audio:**

*   **Quality:** Use high-quality audio with minimal background noise for the best results.
*   **Clarity:** Ensure the speech is clear and representative of the voice you want to clone.
*   **Amount:** Provide a reasonable amount of data (e.g., 1-5 minutes total distributed across several files) for good voice cloning.
*   **Length:** Individual files can range from a few seconds up to a minute or so (Whisper handles longer files well, but shorter, distinct clips might be better for context variety).
*   **Format:** Common audio formats like WAV, MP3, FLAC, Ogg, M4A are supported.
*   **Processing:** Audio will be automatically converted to mono and resampled to the model's required sample rate (24kHz) during processing.

## How It Works: Automatic Transcription and Caching

When you run `runme.py` and provide the path to your voice directory using `--voice-dir`:

1.  **Cache Check:** The script first looks for a cache file named `_tts_context_cache.pt` inside your voice directory.
2.  **Cache Hit (Fast Startup):** If a valid cache file is found, the script loads the pre-processed voice context directly from it. This is very fast.
3.  **Cache Miss (First Run / Regeneration):** If the cache file is missing or invalid:
    *   The script scans the directory for supported audio files.
    *   It loads the OpenAI Whisper ASR model (the "base" model by default). *Note: This might trigger a download of the Whisper model the first time it's used.*)
    *   It transcribes each audio file one by one. This is the most time-consuming step and its duration depends on the amount of audio and your hardware (GPU highly recommended).
    *   It processes the audio (resampling, mono conversion) and pairs it with the generated transcript.
    *   It tokenizes these pairs using the SesameAI model components.
    *   Finally, it saves the resulting tokens (the voice context) into the `_tts_context_cache.pt` file in your voice directory for future runs.
    *   It will also attempt to save the automatically generated transcripts to `metadata_auto_generated.csv` in the same directory for your reference (this file is not used by the script itself).

**Important Considerations:**

*   **First Run Time:** Be patient during the first run for a new voice directory; transcription takes time.
*   **Whisper Dependency:** Ensure you have installed `openai-whisper` (`pip install -U openai-whisper`).
*   **Transcription Accuracy:** The quality of the voice cloning depends partly on Whisper's transcription accuracy. While generally very good, errors can occur, especially with noisy audio or unusual words.
*   **Cache Invalidation:** Currently, the script only checks for the existence of the cache file. If you add, remove, or modify audio files in the voice directory, you should **manually delete the `_tts_context_cache.pt` file** to force the script to regenerate the context with the updated audio.

## Running the Script with a Custom Voice

1.  Ensure your voice directory is prepared with only your audio files.
2.  Navigate to the script directory (`sesameai-tts`) in your terminal.
3.  Run `runme.py`, providing the path to your voice directory using the `--voice-dir` argument:

    ```bash
    # Example using a voice defined in the directory 'voices/my_voice'
    python runme.py --voice-dir voices/my_voice

    # Example specifying CPU and saving output to a file
    python runme.py --voice-dir voices/another_voice --device cpu --output-wav interaction_output.wav
    ```

4.  The script will either load the cached context or perform the first-run transcription/caching.
5.  Follow the prompts to enter text for speech synthesis.

See the main `README.md` for general script usage and other arguments. 