"""
Text-to-Speech (TTS) stage.
Converts LLM response text to audio and plays it back to the user,
using the OpenAI TTS API or a compatible speech synthesis backend.
"""

import os
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.events import AudioOutputEvent

load_dotenv()

# We use tts-1 rather than tts-1-hd deliberately: in a live voice assistant
# latency matters more than audio fidelity. tts-1-hd produces slightly richer
# output but at the cost of noticeably higher time-to-first-byte, which makes
# the conversation feel sluggish. tts-1 keeps the turn-around tight.
MODEL = "tts-1"
VOICE = "alloy"


async def synthesize_speech(text: str, output_path: str) -> AudioOutputEvent:
    """
    Convert text to speech via the OpenAI TTS API and stream the result
    directly to disk, then return an AudioOutputEvent.

    Uses the non-deprecated streaming context manager
    (``with_streaming_response.create``) so the HTTP response body is consumed
    incrementally rather than buffered entirely in memory before writing.

    Args:
        text:        The text to synthesise.
        output_path: Destination file path (should end in .mp3).

    Returns:
        AudioOutputEvent with the saved path and a current timestamp.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _stream_to_file() -> None:
        with client.audio.speech.with_streaming_response.create(
            model=MODEL,
            voice=VOICE,
            input=text,
        ) as response:
            response.stream_to_file(output_path)

    try:
        # Offload the blocking HTTP stream to a thread so the event loop stays
        # responsive while audio data is being written to disk.
        await asyncio.to_thread(_stream_to_file)
    except Exception as exc:
        print(f"[TTS] OpenAI TTS API call failed: {exc}")
        raise

    return AudioOutputEvent(audio_path=output_path)
