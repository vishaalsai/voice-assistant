"""
Large Language Model (LLM) stage.
Sends transcribed text to the configured LLM (OpenAI GPT or Anthropic Claude)
and returns the generated response, managing conversation history and context.
"""

import os
from typing import AsyncGenerator

import anthropic
from dotenv import load_dotenv

from pipeline.events import LLMResponseChunkEvent, TranscriptionEvent

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise — "
    "2-3 sentences max. You are being spoken aloud."
)
MODEL = "claude-haiku-4-5"


async def stream_response(
    transcription: TranscriptionEvent,
) -> AsyncGenerator[LLMResponseChunkEvent, None]:
    """
    Stream a Claude response to the transcribed user input.

    Yields one LLMResponseChunkEvent per text delta from the API, with
    is_final=False, then a single closing event with is_final=True that
    carries the complete accumulated response.

    Args:
        transcription: The TranscriptionEvent produced by the ASR stage.

    Yields:
        LLMResponseChunkEvent for each streamed chunk, then a final summary event.
    """
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    accumulated = ""

    try:
        async with client.messages.stream(
            model=MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": transcription.text}],
        ) as stream:
            async for text_delta in stream.text_stream:
                accumulated += text_delta
                yield LLMResponseChunkEvent(
                    chunk_text=text_delta,
                    is_final=False,
                    full_response_so_far=accumulated,
                )

    except Exception as exc:
        print(f"[LLM] Claude API streaming failed: {exc}")
        raise

    yield LLMResponseChunkEvent(
        chunk_text="",
        is_final=True,
        full_response_so_far=accumulated,
    )
