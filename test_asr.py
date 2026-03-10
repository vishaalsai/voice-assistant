"""
Test script for the ASR pipeline stage.
Runs transcribe_audio() on test_input.wav and prints the resulting TranscriptionEvent.
"""

import asyncio
import sys
from pathlib import Path

# Ensure `pipeline` package is importable when running from this directory.
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.asr import transcribe_audio

WAV_PATH = Path(__file__).parent / "test_input.wav"


async def main() -> None:
    if not WAV_PATH.exists():
        print(f"[ERROR] Test file not found: {WAV_PATH}")
        print("Generate it first:  python _gen_test_wav.py")
        sys.exit(1)

    print(f"Transcribing: {WAV_PATH}\n")
    event = await transcribe_audio(str(WAV_PATH))

    print("--- TranscriptionEvent ---")
    print(f"  text       : {event.text}")
    print(f"  audio_path : {event.audio_path}")
    print(f"  timestamp  : {event.timestamp}")


if __name__ == "__main__":
    asyncio.run(main())
