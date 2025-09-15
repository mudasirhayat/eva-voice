#!/usr/bin/env python3
"""
Test script for the SesameAI TTS API streaming endpoint.
Streams audio from the API and plays it in real-time.
Supports interactive conversation mode with context memory.
"""

import requests
import io
import os
import tempfile
import subprocess
import platform
from pydub import AudioSegment
from pydub.playback import play as pydub_play
import logging
import argparse
from dotenv import load_dotenv

# Handle readline based on platform
if platform.system() != 'Windows':
try:
    import readline  # Unix systems
except ImportError:
    try:
        import pyreadline3 as readline  # Windows alternative
    except ImportError:
        print("Error: Unable to import readline library")
        # Fallback for Windows if pyreadline3 is not available
        readline = None
        print("Note: Install 'pyreadline3' package for better input handling on Windows")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def play_with_logging(audio_segment):
    """Wrapper around pydub.playback.play that logs which backend is being used."""
    try:
        import simpleaudio
        logger.info("Using simpleaudio backend for playback (best quality)")
        playback = simpleaudio.play_buffer(
            audio_segment.raw_data,
            num_channels=audio_segment.channels,
            bytes_per_sample=audio_segment.sample_width,
            sample_rate=audio_segment.frame_rate
        )
        try:
            playback.wait_done()
        except KeyboardInterrupt:
            playback.stop()
        return
    except ImportError:
        logger.info("simpleaudio not available, trying pyaudio...")
    
    try:
        import pyaudio
        logger.info("Using pyaudio backend for playback")
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(audio_segment.sample_width),
                       channels=audio_segment.channels,
                       rate=audio_segment.frame_rate,
                       output=True)
        try:
            # break audio into half-second chunks (to allows keyboard interrupts)
            from pydub.utils import make_chunks
            for chunk in make_chunks(audio_segment, 500):
                stream.write(chunk._data)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        return
    except ImportError:
        logger.info("pyaudio not available, falling back to ffplay...")
    
    logger.info("Using ffplay backend for playback (fallback method)")
    from pydub import playback
    playback._play_with_ffplay(audio_segment)

def patch_audio_playback():
    """Patch pydub playback to properly clean up temp files (from runme.py)."""
    from pydub import playback
    
    def patched_play_with_ffplay(seg):
        """Enhanced playback function that properly cleans up temporary files."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        seg.export(path, format="wav")
        command = ["ffplay", path, "-nodisp", "-autoexit", "-loglevel", "quiet"]
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(path)  # Clean up temporary file
            
    playback._play_with_ffplay = patched_play_with_ffplay

def stream_and_play(api_url: str, api_key: str, prompt: str, use_streaming: bool = False):
    """
    Stream audio from the API and play it in real-time.
    
    Args:
        api_url: Base URL of the API
        api_key: API key for authentication
        prompt: Text prompt to send to the API
        use_streaming: Whether to use streaming endpoints
    """
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    data = {"prompt": prompt}
    
    logger.info(f"Sending prompt to API: '{prompt}'")
    
    try:
        # Choose endpoint based on streaming mode
        endpoint = "/generate/stream" if use_streaming else "/generate_stream"
        
        # Stream the response
        with requests.post(
            f"{api_url}{endpoint}",
            headers=headers,
            json=data,
            stream=True
        ) as response:
            response.raise_for_status()
            
            # Process the WAV stream
            wav_data = io.BytesIO()
            wav_size = None
            header_size = 44  # Standard WAV header size
            
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                
                wav_data.write(chunk)
                current_size = wav_data.tell()
                
                # If we don't have the WAV size yet and we have enough data for the header
                if wav_size is None and current_size >= header_size:
                    wav_data.seek(0)
                    if wav_data.read(4) == b'RIFF':
                        # Read the total file size from the WAV header
                        wav_size = int.from_bytes(wav_data.read(4), 'little') + 8
                        wav_data.seek(current_size)  # Return to end of buffer
                
                # If we have a complete WAV file
                if wav_size is not None and current_size >= wav_size:
                    try:
                        # Extract the complete WAV file
                        wav_data.seek(0)
                        complete_wav = wav_data.read(wav_size)
                        
                        # Create a new buffer for the complete WAV
                        wav_buffer = io.BytesIO(complete_wav)
                        audio_segment = AudioSegment.from_wav(wav_buffer)
                        logger.info(f"Playing audio segment (duration: {len(audio_segment)/1000:.2f}s)")
                        play_with_logging(audio_segment)
                        
                        # Keep any remaining data for the next WAV file
                        remaining_data = wav_data.read()
                        wav_data = io.BytesIO(remaining_data)
                        wav_size = None
                        
                    except Exception as e:
                        logger.warning(f"Error processing WAV chunk: {e}")
                        wav_data = io.BytesIO()
                        wav_size = None
            
            # Process any remaining data
            if wav_data.tell() > 0:
                try:
                    wav_data.seek(0)
                    audio_segment = AudioSegment.from_wav(wav_data)
                    logger.info(f"Playing final audio segment (duration: {len(audio_segment)/1000:.2f}s)")
                    play_with_logging(audio_segment)
                except Exception as e:
                    logger.warning(f"Error processing final WAV chunk: {e}")
                
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"API response: {e.response.text}")
        raise

def conversational_mode(api_url: str, api_key: str, use_streaming: bool = False):
    """
    Run an interactive conversation with the API.
    Maintains conversation context within the session.
    
    Args:
        api_url: Base URL of the API
        api_key: API key for authentication
        use_streaming: Whether to use streaming endpoints
    """
    print("\nEntering conversational mode.")
    print("Type your message and press Enter to send.")
    print("Type 'exit' to end the conversation.")
    print("Press Ctrl+C to skip current response.\n")
    
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    print("\nEnding conversation.")
                    break
                
                # Stream response from API
                stream_and_play(api_url, api_key, user_input, use_streaming)
                
            except KeyboardInterrupt:
                print("\nSkipping current response...")
                continue
                
    except Exception as e:
        logger.error(f"Conversation error: {e}")
        raise

def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test SesameAI TTS API streaming")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8001",
        help="Base URL of the API (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("API_KEY"),
        help="API key for authentication (default: from API_KEY env var)"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive conversation mode"
    )
    parser.add_argument(
        "--streaming",
        "-s",
        action="store_true",
        help="Use streaming endpoints for lower latency"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Hello! This is a test of the streaming TTS API.",
        help="Text prompt to send to the API (ignored in interactive mode)"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        parser.error("API key is required. Provide with --api-key or set API_KEY environment variable.")
    
    # Patch audio playback
    patch_audio_playback()
    
    try:
        if args.interactive:
            conversational_mode(args.api_url, args.api_key, args.streaming)
        else:
            stream_and_play(args.api_url, args.api_key, args.prompt, args.streaming)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 