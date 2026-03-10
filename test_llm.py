"""
Test script for the LLM pipeline stage.
Streams a Claude response to a fake TranscriptionEvent and prints each chunk
as it arrives, so streaming behaviour is immediately visible.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.events import TranscriptionEvent
from pipeline.llm import stream_response

FAKE_TRANSCRIPTION = TranscriptionEvent(
    text="What is the speed of light and why does it matter in everyday life?",
    audio_path="test_input.wav",
)


async def main() -> None:
    print(f"User: {FAKE_TRANSCRIPTION.text}\n")
    print("Assistant: ", end="", flush=True)

    final_event = None

    async for event in stream_response(FAKE_TRANSCRIPTION):
        if not event.is_final:
            # Print each chunk immediately without a newline so the text
            # appears to stream in real time, just like a terminal typewriter.
            print(event.chunk_text, end="", flush=True)
        else:
            final_event = event

    print("\n")  # newline after streaming finishes

    if final_event:
        print("--- Final LLMResponseChunkEvent ---")
        print(f"  is_final             : {final_event.is_final}")
        print(f"  chunk_text           : {repr(final_event.chunk_text)}")
        print(f"  full_response_so_far : {final_event.full_response_so_far}")
        print(f"  timestamp            : {final_event.timestamp}")


if __name__ == "__main__":
    asyncio.run(main())
