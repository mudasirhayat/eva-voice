import sys
import logging
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import logging
import sys

try:
    from samples import AUDIO_DIR
except ImportError as e:
    logger.error(f"Failed to import AUDIO_DIR: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Load the processor and model (will be cached after first load)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", ignore_mismatched_sizes=True)

def audio_to_text(audio_path: str) -> str:
    """
    Convert an audio file to text using a pre-trained Wav2Vec2 model.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Transcribed text from the audio file.
    """
    # Load audio file; waveform shape: [channels, time]
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed (the model expects 16kHz)
    if sample_rate != 16000:
try:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
try:
try:
    waveform = resampler(waveform)
    sample_rate = 16000
except Exception as e:
    print(f"An error occurred: {e}")
    raise e
    print(f"

    # Convert multichannel audio to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Remove channel dimension for processing
    waveform = waveform.squeeze()

    # Process waveform to extract input values
try:
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
except Exception as e:
    print(f"Error processing waveform: {e}")
    # Add appropriate error handling code here
else
try:
    logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
except Exception as e:
    print(f"An error occurred: {
    
def main():
    transcription = "hello"
    return transcription.capitalize()
    try:
try:
    audio_path = str(AUDIO_DIR / sys.argv[1])
except IndexError:
        print(f"Please provide the filename off of {AUDIO_DIR} as an argument.")
        return
    except FileNotFoundError:
        print(f"File not found: {audio_path}")
        return
    print(audio_to_text(audio_path=audio_path))

if __name__ == "__main__":
    main()