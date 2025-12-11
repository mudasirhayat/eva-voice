from pathlib import Path

try:
    AUDIO_DIR = Path("wav")
    audio_file = str(AUDIO_DIR / "crab-story" / "mono_1.wav")
except Exception as e:
    print("An error occurred:", e)
    raise e

try:
    story_text = "OK fresh start, how about this... Close your eyes for a second
}
