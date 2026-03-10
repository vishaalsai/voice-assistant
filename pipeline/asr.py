"""
Automatic Speech Recognition (ASR) stage.
Captures audio input and transcribes it to text using the OpenAI Whisper API
or a compatible speech-to-text backend.
"""

import os
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.events import TranscriptionEvent

load_dotenv()


# Why async?
# The Whisper API call is a single blocking HTTP request — it doesn't natively
# support streaming or cooperative yielding. We wrap it in asyncio.to_thread()
# so the event loop can continue driving other coroutines (e.g. a UI heartbeat,
# a timeout watchdog, or concurrent pipeline runs) while the network I/O blocks.
async def transcribe_audio(audio_path: str) -> TranscriptionEvent:
    """
    Transcribe a .wav file via the OpenAI Whisper API and return a
    TranscriptionEvent containing the transcript and a current timestamp.

    Args:
        audio_path: Absolute or relative path to the .wav file to transcribe.

    Returns:
        TranscriptionEvent with the transcribed text, source path, and timestamp.

    Raises:
        Prints a human-readable error and re-raises on API failure.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call_whisper() -> str:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return response.text

    try:
        text = await asyncio.to_thread(_call_whisper)
    except Exception as exc:
        print(f"[ASR] Whisper API call failed for '{audio_path}': {exc}")
        raise

    return TranscriptionEvent(text=text, audio_path=audio_path)
