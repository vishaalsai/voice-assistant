"""
Test script for the TTS pipeline stage.
Synthesises a short sentence to output_test.mp3 and prints the AudioOutputEvent.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.tts import synthesize_speech

TEXT = (
    "Light travels at about 186,000 miles per second. "
    "It matters because it limits how quickly information travels, "
    "which is why GPS and fiber-optic internet work."
)
OUTPUT_PATH = str(Path(__file__).parent / "output_test.mp3")


async def main() -> None:
    print(f"Synthesising: \"{TEXT}\"\n")

    event = await synthesize_speech(TEXT, OUTPUT_PATH)

    print("--- AudioOutputEvent ---")
    print(f"  audio_path : {event.audio_path}")
    print(f"  timestamp  : {event.timestamp}")
    print()
    print("Open output_test.mp3 to verify audio quality.")


if __name__ == "__main__":
    asyncio.run(main())
