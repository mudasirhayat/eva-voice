# Streaming Audio Implementation

This document details the implementation of real-time audio streaming in the TTS system, focusing on how we achieved high-quality audio output while maintaining low latency.

## Architecture Overview

The streaming implementation consists of three main components:

1. `generate_streaming()` - Generates audio chunks from text
2. `streaming_say()` - Coordinates the streaming process
3. `StreamingAudioPlayer` - Handles real-time audio playback with high-quality output

## Audio Quality Optimizations

### Chunk Size and Latency
- Default chunk size: 20 frames (1.6 seconds of audio)
- Each frame is 80ms, providing a good balance between:
  - Low latency (first audio plays quickly)
  - Sufficient context for quality generation
  - Smooth playback

### Audio Processing Pipeline

The audio processing happens at two levels:

1. **Generator Level** (`generate_streaming`)
```python
# Normalize audio chunk before watermarking
audio_chunk = audio_chunk.to(torch.float32)
audio_chunk = audio_chunk.squeeze() if audio_chunk.dim() > 1 else audio_chunk
audio_chunk = audio_chunk / max(audio_chunk.abs().max().item(), 1e-6)
```

2. **Playback Level** (`StreamingAudioPlayer`)
```python
def queue_audio(self, audio_chunk: torch.Tensor):
    # Careful normalization with epsilon
    audio_chunk = audio_chunk / max(audio_chunk.abs().max().item(), 1e-6)
    # Range clipping for clean audio
    audio_np = np.clip(audio_np, -1.0, 1.0)
```

Key optimizations in `StreamingAudioPlayer`:

1. **Crossfading Between Chunks**
```python
def _crossfade(self, chunk1, chunk2, overlap_samples):
    fade_in = np.linspace(0., 1., overlap_samples)
    fade_out = 1. - fade_in
    result = chunk2.copy()
    result[:overlap_samples] = (
        chunk1[-overlap_samples:] * fade_out +
        chunk2[:overlap_samples] * fade_in
    )
    return result
```
- 10ms overlap between chunks (calculated as `int(0.01 * sample_rate)`)
- Linear crossfading for smooth transitions
- Eliminates audible gaps or clicks between chunks

2. **Audio Format Handling**
- Proper dimension handling (squeezing extra dimensions)
- Consistent float32 format throughout the pipeline
- Range clipping to prevent audio artifacts

3. **Stream Configuration**
```python
self._stream = sd.OutputStream(
    samplerate=self.sample_rate,
    channels=1,
    callback=self._audio_callback,
    dtype=np.float32,
    latency='low',
    blocksize=self.buffer_size
)
```
- Low latency mode for responsive playback
- Explicit blocksize for consistent buffering
- Proper dtype specification for audio quality

## Comparison with Non-Streaming Mode

The streaming implementation matches the audio quality of non-streaming mode (`say()`) while providing real-time playback:

1. **Audio Processing**
   - Both modes use the same normalization approach
   - Both ensure proper audio format handling
   - Streaming adds crossfading for smooth chunk transitions

2. **Latency vs Quality**
   - Streaming: Starts playing after first chunk (~1.6s)
   - Non-streaming: Must generate entire audio before playback

3. **Memory Efficiency**
   - Streaming: Only holds current chunk in memory
   - Non-streaming: Holds entire audio in memory

## Usage Example

```python
# Initialize TTS with streaming
tts = TTS(device="cuda")
tts.load_model()

# Stream with default parameters (20 frames per chunk)
await tts.streaming_say("Hello, world!")

# Stream with custom parameters
await tts.streaming_say(
    "Custom streaming parameters.",
    chunk_size=20,
    temperature=0.9,
    topk=50
)
```

## Implementation Details

### Audio Chunk Processing
1. Generation (`generate_streaming`):
   - Text is split into sentences
   - Each sentence is processed in 20-frame chunks
   - Each chunk undergoes:
     - Normalization
     - Watermarking
     - Resampling if needed

2. Playback (`StreamingAudioPlayer`):
   - Chunks are normalized and clipped
   - Crossfading is applied between chunks
   - Real-time playback with buffer management

### Error Handling
- Graceful stopping on keyboard interrupt
- Buffer underrun prevention
- Proper cleanup of audio resources

## Future Improvements

Potential areas for further enhancement:
1. Adaptive chunk sizes based on system performance
2. Dynamic crossfade duration based on audio characteristics
3. Advanced buffering strategies for network streaming
4. Multi-speaker support in streaming mode 