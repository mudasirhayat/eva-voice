from pydub import AudioSegment
import os
import sys
import argparse

def convert_audio(input_file, output_file=None, output_format=None, bitrate=None, sample_rate=None, channels=None):
    """
    Convert an audio file to another format with optional audio parameters
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str, optional): Path to the output audio file
        output_format (str, optional): Format for the output file (wav, mp3, ogg, flac, etc.)
        bitrate (str, optional): Bitrate for the output file (e.g., "192k")
        sample_rate (int, optional): Sample rate for the output file (e.g., 44100)
        channels (int, optional): Number of audio channels (1 for mono, 2 for stereo)
    
    Returns:
        str: Path to the output audio file
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return None
    
    # Automatically detect input format from file extension
    input_format = os.path.splitext(input_file)[1][1:].lower()
    
    # Generate output details if not provided
    if output_file is None:
        if output_format is None:
            output_format = "wav"  # Default output format
        output_file = os.path.splitext(input_file)[0] + f".{output_format}"
    else:
        # Extract format from output file if not explicitly provided
        if output_format is None:
            output_format = os.path.splitext(output_file)[1][1:].lower()
            if not output_format:
                output_format = "wav"  # Default if no extension
                output_file += f".{output_format}"
    
    try:
        # Load the audio file with automatic format detection
        print(f"Loading {input_file}...")
        try:
            audio = AudioSegment.from_file(input_file, format=input_format)
        except:
            # If format detection fails, try without specifying format
            print(f"Format detection failed, trying automatic detection...")
            audio = AudioSegment.from_file(input_file)
        
        # Apply audio modifications if specified
        if sample_rate:
            print(f"Setting sample rate to {sample_rate} Hz...")
            audio = audio.set_frame_rate(sample_rate)
        
        if channels:
            print(f"Setting channels to {channels}...")
            audio = audio.set_channels(channels)
        
        # Prepare export parameters
        export_params = {
            "format": output_format
        }
        
        if bitrate:
            export_params["bitrate"] = bitrate
        
        # Export to output format
        print(f"Converting to {output_format.upper()} and saving as {output_file}...")
        audio.export(output_file, **export_params)
        
        print(f"Conversion completed successfully!")
        return output_file
    
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

def list_supported_formats():
    """Display a list of commonly supported formats by pydub/ffmpeg"""
    formats = [
        "wav - Waveform Audio File Format",
        "mp3 - MPEG Audio Layer III",
"ogg - Ogg Vorbis",
"flac - Free Lossless Audio Codec",
"aac - Advanced Audio Coding",
        "m4a - MPEG-4 Audio",
        "wma - Windows Media Audio",
        "aiff - Audio Interchange File Format"
    ]
    
    print("\nCommonly supported formats:")
    for fmt in formats:
        print(f"  • {fmt}")
    print("\nNote: Actual format support depends on your FFmpeg installation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio files between formats with optional modifications")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("-o", "--output", help="Path to the output audio file")
    parser.add_argument("-f", "--format", help="Output format (wav, mp3, ogg, flac, etc.)")
    parser.add_argument("-b", "--bitrate", help="Bitrate for the output file (e.g., '192k')")
    parser.add_argument("-sr", "--sample-rate", type=int, help="Sample rate for the output file (e.g., 44100)")
    parser.add_argument("-c", "--channels", type=int, choices=[1, 2], help="Number of audio channels (1=mono, 2=stereo)")
    parser.add_argument("-l", "--list-formats", action="store_true", help="List commonly supported formats")
    
    args = parser.parse_args()
    
    if args.list_formats:
        list_supported_formats()
        if args.input_file == "formats":  # Special case to only show formats
            sys.exit(0)
    
    convert_audio(
        args.input_file,
        args.output,
        args.format,
        args.bitrate,
        args.sample_rate,
        args.channels
    )