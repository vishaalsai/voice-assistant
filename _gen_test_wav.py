"""
One-off utility: generates test_input.wav via the OpenAI TTS API.
Run once before using test_asr.py.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TEXT = "What is the speed of light and why does it matter in everyday life?"
OUT_PATH = Path(__file__).parent / "test_input.wav"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(f"Generating TTS audio for: \"{TEXT}\"")
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=TEXT,
    response_format="wav",
)
response.stream_to_file(str(OUT_PATH))
print(f"Saved: {OUT_PATH}")
