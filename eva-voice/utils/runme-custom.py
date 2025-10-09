#!/usr/bin/env python3
"""
SesameAI Text-to-Speech Model Runner

This script provides a user-friendly interface for interacting with the SesameAI Text-to-Speech model,
allowing users to generate high-quality speech from provided text input.
"""
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
import torch
import torchaudio
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydub import AudioSegment
from pydub.playback import play
from sesameai.generator import Segment, load_csm_1b
from sesameai.watermarking import CSM_1B_GH_WATERMARK, watermark
from samples import voice1
import argparse
import csv
import whisper
import glob
from transformers import pipeline

# Import the new LLMHandler
from llm_handler import LLMHandler 
from groq_handler import GroqHandler # Added GroqHandler import

# Added dotenv import
from dotenv import load_dotenv

# Suppress unnecessary warnings and configure environment
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism for better stability

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class TTS:
    """Wrapper class for text-to-speech functionality using SesameAI models."""
    
    def __init__(self, device: str = "cuda", model_repo: str = "sesame/csm-1b", voice_dir: Optional[str] = None) -> None:
        """
        Initialize the Text-to-Speech engine.
        
        Args:
            device: Device to run inference on ("cuda" or "cpu")
            model_repo: HuggingFace repository ID for the model
            voice_dir: Path to the directory containing voice reference audio files.
        """
        self.device = device
        self.model_repo = model_repo
        self.voice_dir = voice_dir
        self.generator = None
        self.cached_context_tokens = []
        self.cached_context_masks = []
        
        # Configure audio playback
        self._patch_audio_playback()
        
    def _patch_audio_playback(self) -> None:
        """Patch the audio playback functionality to avoid issues with ffplay."""
        from pydub import playback
        
        def patched_play_with_ffplay(seg: AudioSegment) -> None:
            """Enhanced playback function that properly cleans up temporary files."""
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            seg.export(path, format="wav")
            command = ["ffplay", path, "-nodisp", "-autoexit", "-loglevel", "quiet"]
            subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(path)  # Clean up temporary file
            
        playback._play_with_ffplay = patched_play_with_ffplay

    def load_model(self) -> None:
        """Load the TTS model and prepare context for generation."""
        print("\nLoading SesameAI TTS model...")
        try:
            # Redirect stdout to suppress download messages
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                self.generator = load_csm_1b(self.device)
            finally:
                # Restore stdout
                sys.stdout.close()
                sys.stdout = original_stdout
                
            print("\nModel loaded successfully!")
            
            # Prepare context for generation
            self._prepare_context()
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def _prepare_context(self) -> None:
        """Precompute or load cached context tokens using audio files from the specified voice directory."""
        if not self.generator:
            raise ValueError("Model not loaded. Call load_model() first.")
        if not self.voice_dir:
            raise ValueError("Voice directory must be provided during TTS initialization.")

        voice_path = Path(self.voice_dir)
        cache_filename = "_tts_context_cache.pt"
        cache_path = voice_path / cache_filename
        metadata_auto_filename = "metadata_auto_generated.csv"
        metadata_auto_path = voice_path / metadata_auto_filename
        
        supported_audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]

        # --- Try to load from cache first ---
        if cache_path.is_file():
            try:
                print(f"Attempting to load cached context from: {cache_path}")
                # Simple check: Does the cache file exist? We could add more robust checks (e.g., comparing file lists/timestamps)
                cached_data = torch.load(cache_path, map_location='cpu') # Load to CPU first
                if 'tokens' in cached_data and 'masks' in cached_data:
                    self.cached_context_tokens = [t.to(self.device) for t in cached_data['tokens']] # Move to target device
                    self.cached_context_masks = [m.to(self.device) for m in cached_data['masks']] # Move to target device
                    if self.cached_context_tokens and self.cached_context_masks:
                        print(f"Successfully loaded cached context for {len(self.cached_context_tokens)} segments.")
                        return # Context loaded, skip generation
                    else:
                         print("Cache file was empty or invalid. Regenerating context.")
                else:
                    print("Cache file format invalid. Regenerating context.")
            except Exception as e:
                print(f"Failed to load cache file ({cache_path}): {e}. Regenerating context.")
        else:
            print("No cache file found. Generating context from audio files.")

        # --- Cache miss or invalid cache: Generate context ---
        print(f"Scanning {self.voice_dir} for audio files...")
        audio_files = []
        for ext in supported_audio_extensions:
            audio_files.extend(voice_path.glob(ext))
        
        if not audio_files:
            raise FileNotFoundError(f"No supported audio files ({supported_audio_extensions}) found in {self.voice_dir}")

        print(f"Found {len(audio_files)} potential audio files. Starting transcription...")
        
        # Load Whisper model (consider making model size configurable)
        try:
            # Use "cuda" if available and desired for Whisper, otherwise "cpu"
            whisper_device = "cuda" if "cuda" in self.device else "cpu"
            print(f"Loading Whisper ASR model (base) onto device: {whisper_device}...")
            asr_model = whisper.load_model("base", device=whisper_device) 
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Fatal Error: Failed to load Whisper ASR model: {e}")
            print("Please ensure 'openai-whisper' is installed ('pip install -U openai-whisper')")
            print("and that you have sufficient memory/compute resources.")
            raise

        segments_to_process: List[Dict[str, Any]] = []
        generated_metadata = []
        
        for audio_path in audio_files:
            print(f"Processing: {audio_path.name} ...", end=" ", flush=True)
            try:
                # 1. Transcribe
                # Handle transcription options if needed (e.g., language detection)
                result = asr_model.transcribe(str(audio_path), fp16=torch.cuda.is_available() and whisper_device=="cuda")
                transcript = result["text"].strip()
                print(f"Transcript: '{transcript[:50]}...'", end=" ", flush=True)
                if not transcript:
                    logger.warning(f"Skipping {audio_path.name} due to empty transcript.")
                    print("[Skipped: Empty Transcript]")
                    continue
                
                # 2. Load Audio Tensor
                audio_tensor = self._load_audio(str(audio_path))
                print("[Loaded Audio]", end=" ", flush=True)

                segments_to_process.append(
                    {'text': transcript, 'audio': audio_tensor, 'path': str(audio_path)}
                )
                generated_metadata.append({'filename': audio_path.name, 'transcript': transcript})
                print("[OK]")

            except Exception as e:
                logger.error(f"Failed to process {audio_path.name}: {e}")
                print(f"[Error: {e}]")

        if not segments_to_process:
             raise ValueError(f"Could not successfully process any audio files from {self.voice_dir}. Check logs for errors.")

        # 3. Tokenize Segments for TTS Context
        self.cached_context_tokens = []
        self.cached_context_masks = []
        print(f"\nTokenizing {len(segments_to_process)} processed segments for TTS context...")
        for segment_data in segments_to_process:
            segment = Segment(text=segment_data['text'], speaker=1, audio=segment_data['audio'])
            try:
                tokens, masks = self.generator._tokenize_segment(segment)
                # Ensure tokens/masks are on CPU before saving cache
                self.cached_context_tokens.append(tokens.cpu())
                self.cached_context_masks.append(masks.cpu())
            except Exception as e:
                 logger.error(f"Failed to tokenize segment from {Path(segment_data['path']).name}: {e}")

        if not self.cached_context_tokens:
             raise ValueError("Failed to tokenize any reference segments after processing. Cannot proceed.")

        # 4. Save Cache
        print(f"Saving context cache for {len(self.cached_context_tokens)} segments to: {cache_path}")
        try:
            cache_data = {
                'tokens': self.cached_context_tokens,
                'masks': self.cached_context_masks
            }
            torch.save(cache_data, cache_path)
            print("Context cache saved successfully.")
            
            # Also save the generated metadata for user reference
            try:
                with open(metadata_auto_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['filename', 'transcript']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(generated_metadata)
                print(f"Auto-generated transcripts saved to: {metadata_auto_path}")
            except Exception as meta_e:
                logger.error(f"Could not save auto-generated metadata csv: {meta_e}")
                
        except Exception as e:
            logger.error(f"Failed to save context cache to {cache_path}: {e}")
            # Continue without cache, but warn the user
            print(f"Warning: Could not save context cache. Context will be regenerated next time.")
            
        # Move cached tensors to the target device for current use
        self.cached_context_tokens = [t.to(self.device) for t in self.cached_context_tokens]
        self.cached_context_masks = [m.to(self.device) for m in self.cached_context_masks]

        print(f"Reference audio context prepared and cached using {len(self.cached_context_tokens)} segments.")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file for model consumption.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio tensor
        """
        # Normalize path for cross-platform compatibility
        audio_path = Path(audio_path)
        logger.debug(f"Loading audio: {audio_path}")
        audio_tensor, sample_rate = torchaudio.load(str(audio_path))

        # Convert stereo to mono if necessary
        if audio_tensor.shape[0] > 1:
            logger.debug("Converting stereo to mono")
            audio_tensor = audio_tensor.mean(dim=0)

        # Resample if sample rates differ
        if sample_rate != self.generator.sample_rate:
            logger.debug(f"Resampling from {sample_rate}Hz to {self.generator.sample_rate}Hz")
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, self.generator.sample_rate
            )

        return audio_tensor.squeeze()

    def generate_with_context(
        self, 
        prompt: str, 
        speaker: int = 1, 
        max_audio_length_ms: int = 10000, 
        temperature: float = 0.9, 
        topk: int = 50
    ) -> torch.Tensor:
        """
        Generate audio from text using cached context.
        
        Args:
            prompt: Text to synthesize
            speaker: Speaker ID
            max_audio_length_ms: Maximum duration in milliseconds
            temperature: Sampling temperature (higher = more random)
            topk: Top-k sampling parameter
            
        Returns:
            Audio tensor
        """
        self.generator._model.reset_caches()
        with torch.inference_mode():
            # Use mixed precision throughout the generation process
            with torch.autocast(self.device, dtype=torch.bfloat16):
                # Tokenize the new prompt
                gen_tokens, gen_masks = self.generator._tokenize_text_segment(prompt, speaker)
                # Combine cached tokens with new prompt tokens
                prompt_tokens = (
                    torch.cat(self.cached_context_tokens + [gen_tokens], dim=0)
                    .long()
                    .to(self.device)
                )
                prompt_tokens_mask = (
                    torch.cat(self.cached_context_masks + [gen_masks], dim=0)
                    .bool()
                    .to(self.device)
                )

                samples = []
                curr_tokens = prompt_tokens.unsqueeze(0)
                curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
                curr_pos = (
                    torch.arange(0, prompt_tokens.size(0))
                    .unsqueeze(0)
                    .long()
try:
    audio_frames = torchaudio.compliance.kaldi.resample_waveform(
        waveform, self.sample_rate, self.target_sample_rate)
except Exception as e:
    print(f"Error resampling waveform: {e}")
                max_seq_len = 2048 - max_audio_frames
                if curr_tokens.size(1) >= max_seq_len:
                    raise ValueError(f"Input too long ({curr_tokens.size(1)} tokens). Maximum is {max_seq_len} tokens.")

                for _ in range(max_audio_frames):
                    sample = self.generator._model.generate_frame(
                        curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                    )
                    if torch.all(sample == 0):
                        break
                    samples.append(sample)
                    curr_tokens = torch.cat(
                        [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
                    ).unsqueeze(1)
                    curr_tokens_mask = torch.cat(
                        [
                            torch.ones_like(sample).bool(),
                            torch.zeros(1, 1).bool().to(self.device),
                        ],
                        dim=1,
                    ).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1

            # Decode audio from tokens
            audio = (
                self.generator._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))
                .squeeze(0)
                .squeeze(0)
            )

            # Apply watermarking
            audio, wm_sample_rate = watermark(
                self.generator._watermarker, audio, self.generator.sample_rate, CSM_1B_GH_WATERMARK
            )
            audio = torchaudio.functional.resample(
                audio, orig_freq=wm_sample_rate, new_freq=self.generator.sample_rate
            )

        return audio

    def generate_audio_segment(
        self, 
        prompt: str, 
        fade_duration: int = 50, 
        start_silence_duration: int = 500, 
        end_silence_duration: int = 100
    ) -> AudioSegment:
        """
        Generate an AudioSegment from text with proper silence padding and fading.
        
        Args:
            prompt: Text to synthesize
            fade_duration: Duration of fade-in and fade-out in milliseconds
            start_silence_duration: Duration of silence at the beginning in milliseconds
            end_silence_duration: Duration of silence at the end in milliseconds
            
        Returns:
            AudioSegment with the generated audio
        """
        
        # Generate raw audio
        audio = self.generate_with_context(prompt, speaker=1, max_audio_length_ms=10000)

        # Normalize audio
        audio = audio.to(torch.float32)
        if audio.dim() > 1:
            audio = audio.squeeze()
        audio = audio / max(audio.abs().max(), 1e-6)

        # Convert to 16-bit PCM
        audio_np = (audio.cpu().numpy() * 32767).astype("int16")
        audio_segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=self.generator.sample_rate,
            sample_width=2,
            channels=1,
        )

        # Add silence padding and fade-in/out
        start_silence = AudioSegment.silent(duration=start_silence_duration)
        end_silence = AudioSegment.silent(duration=end_silence_duration)
        audio_segment = start_silence + audio_segment + end_silence
        audio_segment = audio_segment.fade_in(fade_duration).fade_out(fade_duration)

        return audio_segment

    def _generate_audio_segment_wrapper(self, sentence, fade_duration, start_silence_duration, end_silence_duration):
        return self.generate_audio_segment(sentence, fade_duration, start_silence_duration, end_silence_duration)

    def say(
        self, 
        text: str, 
        output_filename: Optional[str] = "combined_output.wav", 
        fallback_duration: int = 1000, 
        fade_duration: int = 500, 
        start_silence_duration: int = 500, 
        end_silence_duration: int = 100
    ) -> None:
        """
        Generate and play audio for a given text, splitting into sentences for better quality.
        
        Args:
            text: Text to synthesize
            output_filename: Optional filename to save the combined audio
            fallback_duration: Duration of silence to use if generation fails
            fade_duration: Duration of fade-in and fade-out in milliseconds
            start_silence_duration: Duration of silence at the beginning in milliseconds
            end_silence_duration: Duration of silence at the end in milliseconds
        """
        # Normalize and split text into sentences
        text = textwrap.dedent(text).strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            print("No valid text to process")
            return

        segments = []
        
        # Import threading for parallel playback
        import threading
        
        for sentence in sentences:
            try:
                start_time = time.time()
                print(f"> {sentence} ... ", end='', flush=True)
                
                # Generate audio segment
                seg = self.generate_audio_segment(
                    sentence, 
                    fade_duration=fade_duration, 
                    start_silence_duration=start_silence_duration, 
                    end_silence_duration=end_silence_duration
                )
                end_time = time.time()
                # Compute metrics
                duration = seg.duration_seconds
try:
    proc_time = end_time - start_time
    rtt_ratio = proc_time / duration
    rtf = 1 / rtt_ratio
except ZeroDivisionError:
    print("Error: Division by zero.")
                print(f"[Audio: {duration:.2f}s in {proc_time:.2f}s, RTF: {rtf:.2f}x]")
                segments.append(seg)
                
                # Play audio in a separate thread so it doesn't block the next generation
                audio_thread = threading.Thread(target=play, args=(seg,))
                audio_thread.daemon = True  # Allow program to exit even if thread is still running
                audio_thread.start()
                
            except KeyboardInterrupt:
                print("\nExiting due to KeyboardInterrupt")
                return
            except Exception as e:
                print(f"Error generating audio for sentence: {sentence}: {e}")
                seg = AudioSegment.silent(duration=fallback_duration)
                seg = seg.fade_in(fade_duration).fade_out(fade_duration)
                segments.append(seg)

        # Export combined audio if requested
        if output_filename and segments:
            combined = segments[0]
            for seg in segments[1:]:
                combined += seg
            output_path = Path(output_filename)
            logger.debug(f"\nExporting combined audio to {output_path.absolute()}...")
            combined.export(output_filename, format="wav")
            print(f"Export complete: {len(combined) / 1000:.2f} seconds of audio")

    def export_wav(
        self, 
        text: str, 
        output_filename: str, 
        fallback_duration: int = 1000, 
        max_retries: int = 2
    ) -> None:
        """
        Generate audio for a text and export it to a WAV file without playing.
        
        Args:
            text: Text to synthesize
            output_filename: Filename to save the combined audio
            fallback_duration: Duration of silence to use if generation fails
            max_retries: Maximum number of retries if generation fails
        """
        # Split text into sentences
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        segments = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue

            retries = 0
            seg = None
            while retries <= max_retries:
                print(f"Export: Generating audio for sentence: {sentence} (Attempt {retries + 1})")
                try:
                    seg = self.generate_audio_segment(sentence)
                except Exception
                    break
                except Exception as e:
                    retries += 1
                    print(f"Export: Error for sentence: {sentence} (Attempt {retries}): {e}")
            
            if seg is None:
                print(f"Export: Using fallback for sentence: {sentence}")
                seg = AudioSegment.silent(duration=fallback_duration)
            segments.append(seg)

        if segments:
            # Concatenate segments
            combined = segments[0]
            for seg in segments[1:]:
                combined += seg
            print(f"Exporting to {output_filename}...")
            combined.export(output_filename, format="wav")
            print(f"Export complete: {len(combined) / 1000:.2f} seconds of audio")
        else:
            print("No audio segments to export")


def main():
    """Main entry point for the script."""
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="SesameAI Text-to-Speech Runner")
    parser.add_argument(
        "--voice-dir",
        type=str,
        required=True,
        help="Path to the directory containing voice reference audio files (e.g., wav, mp3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
         "--output-wav",
         type=str,
         default=None, # Default to None, 'say' will use its own default if not overridden
         help="Optional: Filename to save the combined audio output from the 'say' command. If not provided, audio is played but not saved by default from the loop."
    )
    parser.add_argument(
         "--llm-model",
         type=str,
         default="Qwen/Qwen2.5-7B-Instruct-AWQ",
         help="Hugging Face model ID for the text generation pipeline."
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="local",
        choices=["local", "groq"],
        help="Specify the LLM provider ('local' for Hugging Face model, 'groq' for Groq API)."
    )
    parser.add_argument(
        "--groq-model",
        type=str,
        default="llama3-8b-8192",
        help="Model to use with the Groq provider (e.g., llama3-8b-8192, mixtral-8x7b-32768)."
    )


    args = parser.parse_args()

    # Determine output filename for the say loop
    # If --output-wav is provided, use it. Otherwise, 'say' will use its internal default if saving is desired later.
    # For clarity, we can explicitly pass None if the argument wasn't given.
    output_filename_for_say = args.output_wav 


    print(f"Using device: {args.device}")
    print(f"Using voice directory: {args.voice_dir}")
    print(f"Using LLM model: {args.llm_model}")
    print(f"Using LLM provider: {args.llm_provider}") # Added print
    if args.llm_provider == "groq":
        print(f"Using Groq model: {args.groq_model}") # Added print

    tts = TTS(device=args.device, voice_dir=args.voice_dir) # Pass voice_dir

    # --- Initialize LLM Handler ---
    # The handler itself will print loading messages and handle errors
    llm_handler = None # Initialize as None
    if args.llm_provider == "local":
        llm_handler = LLMHandler(model_name=args.llm_model, device=args.device)
    elif args.llm_provider == "groq":
        # GroqHandler uses GROQ_API_KEY env var and takes groq model name
        llm_handler = GroqHandler(model_name=args.groq_model)
        if not llm_handler.client:
             print("Groq handler failed to initialize. Falling back to no LLM mode.")
             # No need to set llm_handler to None explicitly, GroqHandler handles its state
    else:
         print(f"Error: Unknown LLM provider '{args.llm_provider}'. No LLM will be used.")
    # -----------------------------

    try:
        tts.load_model()
        print("Performing initial warmup generation...")
        try:
             warmup = tts.generate_audio_segment("All warmed up baby!")
             play(warmup)
             print("Warmup complete.")
        except Exception as e:
             print(f"Warning: Warmup generation/playback failed: {e}")
             # Not fatal, TTS might still work

        print("\nSesameAI Conversational System")
        print("===============================")
        while True:
            try:
                user_input = input("\nEnter your message (or press Ctrl+C to exit): ")
                if not user_input.strip():
                     print("Please enter some text.")
                     continue

                # --- Get LLM Response (using LLMHandler) ---
                if llm_handler and llm_handler.model: # Check if handler initialized and model loaded
                    print("\nLLM Thinking...")
                    # Add system prompt if desired, or modify get_response signature
                    llm_response_text = llm_handler.get_response(user_input)
                    print(f"\nLLM Response:\n{textwrap.fill(llm_response_text, width=80)}")
                    print("-" * 80)
                else:
                    # Fallback if LLM failed to load - just echo user input for TTS
                    print("\n(LLM not loaded, using user input directly for TTS)")
                    llm_response_text = user_input 
                # ------------------------------------
                
                # --- Generate Speech for LLM Response ---
                if llm_response_text.strip():
                    print("\nGenerating speech...")
                    # Pass the output filename from args to the say method
                    tts.say(llm_response_text, output_filename=output_filename_for_say) 
                else:
                    print("No text content from LLM to synthesize.")
                # ---------------------------------------
                    
            except Exception as e:
                print(f"Error processing input: {e}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Example LLM integration (from user) - for reference
# # Use a pipeline as a high-level helper
# from transformers import pipeline
# 
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct")
# pipe(messages)