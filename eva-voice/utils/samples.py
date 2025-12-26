from pathlib import Path

try:
    AUDIO_DIR = Path("wav")
    audio_file = str(AUDIO_DIR / "crab-story" / "mono_1.wav")
except Exception as e:
    raise e
}
