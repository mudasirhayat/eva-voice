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
import queue
import threading
import sounddevice as sd
from pathlib import Path
from typing import Optional, Generator
from pydub import AudioSegment
from pydub.playback import play
from sesameai.generator import Segment, load_csm_1b
from sesameai.watermarking import CSM_1B_GH_WATERMARK, watermark
from samples import voice1
import numpy as np


# Suppress unnecessary warnings and configure environment
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism for better stability

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class StreamingAudioPlayer:
    """Handles streaming audio playback with a buffer."""
    
    def __init__(self, sample_rate: int = 24000, buffer_size: int = 4096):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stop_flag = False
        self._stream = None
        self._overlap_samples = int(0.01 * sample_rate)  # 10ms overlap for crossfading
        self._previous_chunk_end = None
        
    def _crossfade(self, chunk1, chunk2, overlap_samples):
        """Apply linear crossfade between two chunks."""
        if chunk1 is None or chunk2 is None or overlap_samples == 0:
            return chunk2
        
        fade_in = np.linspace(0., 1., overlap_samples)
        fade_out = 1. - fade_in
        
        result = chunk2.copy()
        result[:overlap_samples] = (
            chunk1[-overlap_samples:] * fade_out +
            chunk2[:overlap_samples] * fade_in
        )
        return result
        
    def _audio_callback(self, outdata, frames, time, status):
        """Callback for sounddevice to get audio data."""
        try:
            data = self.audio_queue.get_nowait()
            
            # Apply crossfading if we have previous data
            if self._previous_chunk_end is not None:
                data = self._crossfade(self._previous_chunk_end, data, self._overlap_samples)
            
            if len(data) < frames:
                # Store end of current chunk for next crossfade
                self._previous_chunk_end = data
                outdata[:len(data), 0] = data
                outdata[len(data):, 0] = 0
                raise sd.CallbackStop()
            else:
                # Store end of current chunk for next crossfade
                self._previous_chunk_end = data[:frames]
                outdata[:, 0] = data[:frames]
                # Put remaining data back in queue
                if len(data) > frames:
                    self.audio_queue.put(data[frames:])
        except queue.Empty:
            outdata.fill(0)
            raise sd.CallbackStop()
            
    def start(self):
        """Start audio playback stream."""
        if not self.is_playing:
            self.stop_flag = False
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                dtype=np.float32,
                latency='low',
                blocksize=self.buffer_size
            )
            self._stream.start()
            self.is_playing = True
            
    def queue_audio(self, audio_chunk: torch.Tensor):
        """Add audio chunk to playback queue."""
        if not self.stop_flag:
            # Ensure audio is float32 and normalize carefully
            audio_chunk = audio_chunk.to(torch.float32)
            if audio_chunk.dim() > 1:
                audio_chunk = audio_chunk.squeeze()
            
            # Normalize with epsilon to prevent division by zero
            audio_chunk = audio_chunk / max(audio_chunk.abs().max().item(), 1e-6)
            
            # Convert to numpy and ensure proper range
            audio_np = audio_chunk.cpu().numpy()
            audio_np = np.clip(audio_np, -1.0, 1.0)
            
            self.audio_queue.put(audio_np)
            
    def stop(self):
        """Stop audio playback."""
        self.stop_flag = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.is_playing = False
        self._previous_chunk_end = None
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

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
            
        # If voice_dir is provided, use dynamic loading from directory
        if self.voice_dir:
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
                    cached_data = torch.load(cache_path, map_location='cpu')
                    if 'tokens' in cached_data and 'masks' in cached_data:
                        self.cached_context_tokens = [t.to(self.device) for t in cached_data['tokens']]
                        self.cached_context_masks = [m.to(self.device) for m in cached_data['masks']]
                        if self.cached_context_tokens and self.cached_context_masks:
                            print(f"Successfully loaded cached context for {len(self.cached_context_tokens)} segments.")
                            return
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
            
            # Load Whisper model
            try:
                import whisper
                whisper_device = "cuda" if "cuda" in self.device else "cpu"
                print(f"Loading Whisper ASR model (base) onto device: {whisper_device}...")
                asr_model = whisper.load_model("base", device=whisper_device)
                print("Whisper model loaded.")
            except Exception as e:
                print(f"Fatal Error: Failed to load Whisper ASR model: {e}")
                print("Please ensure 'openai-whisper' is installed ('pip install -U openai-whisper')")
                raise

            segments_to_process = []
            generated_metadata = []
            
            for audio_path in audio_files:
                print(f"Processing: {audio_path.name} ...", end=" ", flush=True)
                try:
                    # Transcribe
                    result = asr_model.transcribe(str(audio_path), fp16=torch.cuda.is_available() and whisper_device=="cuda")
                    transcript = result["text"].strip()
                    print(f"Transcript: '{transcript[:50]}...'", end=" ", flush=True)
                    if not transcript:
                        print("[Skipped: Empty Transcript]")
                        continue
                    
                    # Load Audio Tensor
                    audio_tensor = self._load_audio(str(audio_path))
                    print("[Loaded Audio]", end=" ", flush=True)

                    segments_to_process.append({
                        'text': transcript,
                        'audio': audio_tensor,
                        'path': str(audio_path)
                    })
                    generated_metadata.append({
                        'filename': audio_path.name,
                        'transcript': transcript
                    })
                    print("[OK]")

                except Exception as e:
                    print(f"[Error: {e}]")

            if not segments_to_process:
                raise ValueError(f"Could not successfully process any audio files from {self.voice_dir}")

            # Tokenize Segments for TTS Context
            print(f"\nTokenizing {len(segments_to_process)} processed segments for TTS context...")
            for segment_data in segments_to_process:
                segment = Segment(text=segment_data['text'], speaker=1, audio=segment_data['audio'])
                try:
                    tokens, masks = self.generator._tokenize_segment(segment)
                    self.cached_context_tokens.append(tokens.cpu())
                    self.cached_context_masks.append(masks.cpu())
                except Exception as e:
                    print(f"Failed to tokenize segment from {Path(segment_data['path']).name}: {e}")

            if not self.cached_context_tokens:
                raise ValueError("Failed to tokenize any reference segments after processing")

            # Save Cache
            print(f"Saving context cache for {len(self.cached_context_tokens)} segments to: {cache_path}")
            try:
                cache_data = {
                    'tokens': self.cached_context_tokens,
                    'masks': self.cached_context_masks
                }
                torch.save(cache_data, cache_path)
                print("Context cache saved successfully.")
                
                # Save metadata
                import csv
                try:
                    with open(metadata_auto_path, 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = ['filename', 'transcript']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(generated_metadata)
                    print(f"Auto-generated transcripts saved to: {metadata_auto_path}")
                except Exception as meta_e:
                    print(f"Could not save auto-generated metadata csv: {meta_e}")
                    
            except Exception as e:
                print(f"Failed to save context cache to {cache_path}: {e}")
                print("Warning: Could not save context cache. Context will be regenerated next time.")
                
            # Move cached tensors to the target device
            self.cached_context_tokens = [t.to(self.device) for t in self.cached_context_tokens]
            self.cached_context_masks = [m.to(self.device) for m in self.cached_context_masks]

            print(f"Reference audio context prepared and cached using {len(self.cached_context_tokens)} segments.")
            
        # If no voice_dir, use the default voice1 dictionary from samples.py
        else:
            print("Using default voice samples from samples.py...")
            from samples import voice1
            segments = [
                Segment(text=text, speaker=1, audio=self._load_audio(audio_path))
                for audio_path, text in voice1.items()
            ]
            
            # Cache tokenized representations for fixed context segments
            for segment in segments:
                tokens, masks = self.generator._tokenize_segment(segment)
                self.cached_context_tokens.append(tokens)
                self.cached_context_masks.append(masks)
            print("Reference audio context prepared")

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
        max_audio_length_ms: int = 30000, 
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
                    .to(self.device)
                )

                max_audio_frames = int(max_audio_length_ms / 80)
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
        start_time = time.time()
        logger.info(f"Generating audio for text: '{prompt}'")
        
        # Generate raw audio
        audio = self.generate_with_context(prompt)
        
        # Convert to float32 and normalize
        audio = audio.to(torch.float32)
        audio = audio.squeeze() if audio.dim() > 1 else audio
        audio = audio / max(audio.abs().max().item(), 1e-6)
        
        # Convert to 16-bit PCM
        audio_np = (audio.cpu().numpy() * 32767).astype("int16")
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=self.generator.sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Add silence padding and fading
        audio_segment = (
            AudioSegment.silent(duration=start_silence_duration) +
            audio_segment.fade_in(fade_duration).fade_out(fade_duration) +
            AudioSegment.silent(duration=end_silence_duration)
        )
        
        # Log audio properties
        generation_time = time.time() - start_time
        logger.info("Audio segment properties:")
        logger.info(f"  Duration: {len(audio_segment)/1000:.2f}s")
        logger.info(f"  Sample rate: {audio_segment.frame_rate}Hz")
        logger.info(f"  Sample width: {audio_segment.sample_width} bytes")
        logger.info(f"  Channels: {audio_segment.channels}")
        logger.info(f"  Generation time: {generation_time:.2f}s")
        logger.info(f"  Max amplitude: {float(audio.abs().max()):.4f}")
        
        # Save to file with timestamp for comparison
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"runme_audio_{timestamp}.wav"
        audio_segment.export(filename, format="wav")
        logger.info(f"Saved audio segment to: {filename}")
        
        return audio_segment

    def _generate_audio_segment_wrapper(self, sentence, fade_duration, start_silence_duration, end_silence_duration):
        return self.generate_audio_segment(sentence, fade_duration, start_silence_duration, end_silence_duration)

    def say(
        self, 
        text: str, 
        output_filename: Optional[str] = "combined_output.wav", 
        fallback_duration: int = 1000, 
        fade_duration: int = 50, 
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
                proc_time = end_time - start_time
                rtt_ratio = proc_time / duration
                rtf = 1 / rtt_ratio
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
                try:
                    print(f"Export: Generating audio for sentence: {sentence} (Attempt {retries + 1})")
                    seg = self.generate_audio_segment(sentence)
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

    async def generate_streaming(
        self,
        prompt: str,
        chunk_size: int = 20,
        speaker: int = 1,
        temperature: float = 0.9,
        topk: int = 50,
        accumulate_size: int = 3  # Number of chunks to accumulate before processing
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generate audio in streaming chunks.
        
        Args:
            prompt: Text to synthesize
            chunk_size: Number of frames to generate per chunk
            speaker: Speaker ID
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            accumulate_size: Number of chunks to accumulate before processing
            
        Yields:
            Audio tensor chunks
        """
        self.generator._model.reset_caches()
        
        with torch.inference_mode():
            with torch.autocast(self.device, dtype=torch.bfloat16):
                # Tokenize the prompt
                gen_tokens, gen_masks = self.generator._tokenize_text_segment(prompt, speaker)
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

                curr_tokens = prompt_tokens.unsqueeze(0)
                curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
                curr_pos = (
                    torch.arange(0, prompt_tokens.size(0))
                    .unsqueeze(0)
                    .long()
                    .to(self.device)
                )

                chunk_samples = []
                accumulated_samples = []
                max_audio_frames = int(30000 / 80)  # 30 seconds max
                max_seq_len = 2048 - max_audio_frames
                
                if curr_tokens.size(1) >= max_seq_len:
                    raise ValueError(f"Input too long ({curr_tokens.size(1)} tokens). Maximum is {max_seq_len} tokens.")

                for _ in range(max_audio_frames):
                    sample = self.generator._model.generate_frame(
                        curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                    )
                    if torch.all(sample == 0):
                        break
                        
                    chunk_samples.append(sample)
                    
                    # When we have enough samples for a chunk
                    if len(chunk_samples) >= chunk_size:
                        # Add to accumulated samples
                        accumulated_samples.extend(chunk_samples)
                        chunk_samples = []
                        
                        # When we have enough accumulated chunks, process and yield
                        if len(accumulated_samples) >= chunk_size * accumulate_size:
                            # Process accumulated samples
                            audio_chunk = (
                                self.generator._audio_tokenizer.decode(
                                    torch.stack(accumulated_samples).permute(1, 2, 0)
                                )
                                .squeeze(0)
                                .squeeze(0)
                            )
                            
                            # Normalize audio chunk before watermarking
                            audio_chunk = audio_chunk.to(torch.float32)
                            audio_chunk = audio_chunk.squeeze() if audio_chunk.dim() > 1 else audio_chunk
                            audio_chunk = audio_chunk / max(audio_chunk.abs().max().item(), 1e-6)
                            
                            # Apply watermarking to accumulated chunk
                            audio_chunk, wm_sample_rate = watermark(
                                self.generator._watermarker,
                                audio_chunk,
                                self.generator.sample_rate,
                                CSM_1B_GH_WATERMARK
                            )
                            if wm_sample_rate != self.generator.sample_rate:
                                audio_chunk = torchaudio.functional.resample(
                                    audio_chunk,
                                    orig_freq=wm_sample_rate,
                                    new_freq=self.generator.sample_rate
                                )
                                
                            yield audio_chunk
                            accumulated_samples = []
                    
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
                
                # Handle any remaining samples
                remaining_samples = chunk_samples + accumulated_samples
                if remaining_samples:
                    audio_chunk = (
                        self.generator._audio_tokenizer.decode(
                            torch.stack(remaining_samples).permute(1, 2, 0)
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    
                    # Normalize audio chunk before watermarking
                    audio_chunk = audio_chunk.to(torch.float32)
                    audio_chunk = audio_chunk.squeeze() if audio_chunk.dim() > 1 else audio_chunk
                    audio_chunk = audio_chunk / max(audio_chunk.abs().max().item(), 1e-6)
                    
                    # Apply watermarking to final chunk
                    audio_chunk, wm_sample_rate = watermark(
                        self.generator._watermarker,
                        audio_chunk,
                        self.generator.sample_rate,
                        CSM_1B_GH_WATERMARK
                    )
                    if wm_sample_rate != self.generator.sample_rate:
                        audio_chunk = torchaudio.functional.resample(
                            audio_chunk,
                            orig_freq=wm_sample_rate,
                            new_freq=self.generator.sample_rate
                        )
                        
                    yield audio_chunk

    async def streaming_say(
        self,
        text: str,
        chunk_size: int = 20,
        temperature: float = 0.9,
        topk: int = 50
    ) -> None:
        """
        Generate and play audio in a streaming fashion.
        
        Args:
            text: Text to synthesize
            chunk_size: Number of frames to generate per chunk
            temperature: Sampling temperature
            topk: Top-k sampling parameter
        """
        # Split text into sentences
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            print("No valid text to process")
            return
            
        # Create audio player
        player = StreamingAudioPlayer(sample_rate=self.generator.sample_rate)
        
        try:
            for sentence in sentences:
                print(f"> {sentence} ... ", end="", flush=True)
                start_time = time.time()
                
                # Start playback
                player.start()
                
                # Generate and play chunks
                async for audio_chunk in self.generate_streaming(
                    sentence,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    topk=topk
                ):
                    # Process audio chunk similar to generate_audio_segment
                    audio_chunk = audio_chunk.to(torch.float32)
                    audio_chunk = audio_chunk.squeeze() if audio_chunk.dim() > 1 else audio_chunk
                    audio_chunk = audio_chunk / max(audio_chunk.abs().max().item(), 1e-6)
                    player.queue_audio(audio_chunk)
                    
                # Wait for queue to empty
                while not player.audio_queue.empty() and player.is_playing:
                    await asyncio.sleep(0.1)
                    
                player.stop()
                
                end_time = time.time()
                print(f"[Generated in {end_time - start_time:.2f}s]")
                
        except KeyboardInterrupt:
            print("\nStopping audio generation...")
            player.stop()
        except Exception as e:
            print(f"Error generating audio: {e}")
            player.stop()
            raise


def main():
    """Main entry point for the script."""
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="SesameAI Text-to-Speech Runner")
    parser.add_argument(
        "--voice-dir",
        type=str,
        help="Optional: Path to the directory containing voice reference audio files (e.g., wav, mp3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for audio generation",
    )
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    if args.voice_dir:
        print(f"Using voice directory: {args.voice_dir}")
    else:
        print("Using default voice samples from samples.py")
    
    tts = TTS(device=args.device, voice_dir=args.voice_dir)
    
    async def async_main():
        try:
            tts.load_model()
            if args.streaming:
                await tts.streaming_say("All warmed up baby!")
            else:
                warmup = tts.generate_audio_segment("All warmed up baby!")
                play(warmup)
            
            print("\nSesameAI TTS System")
            print("====================")
            while True:
                try:
                    user_input = input("\nEnter text (or press Ctrl+C to exit): ")
                    if user_input.strip():
                        if args.streaming:
                            await tts.streaming_say(user_input)
                        else:
                            tts.say(user_input)
                    else:
                        print("Please enter some text to generate audio.")
                except Exception as e:
                    print(f"Error processing input: {e}")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Fatal error: {e}")
            sys.exit(1)
    
    asyncio.run(async_main())


if __name__ == "__main__":
    main()