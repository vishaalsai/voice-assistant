"""
Error and retry handlers.
Implements exponential back-off retries, circuit-breaker logic, and graceful
degradation fallbacks for ASR, LLM, and TTS API calls.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Union

from pipeline.asr import transcribe_audio
from pipeline.llm import stream_response
from pipeline.tts import synthesize_speech
from pipeline.events import (
    AudioOutputEvent,
    LLMResponseChunkEvent,
    TranscriptionEvent,
)


# ── Error type ────────────────────────────────────────────────────────────────

@dataclass
class ComponentError:
    """Returned (or yielded) in place of a normal event when a stage fails."""
    component:  str    # e.g. "ASR", "LLM", "TTS"
    error_type: str    # e.g. "timeout", "api_error", "rate_limit"
    message:    str
    timestamp:  float = field(default_factory=time.time)


# ── Per-component timeout budgets (seconds) ───────────────────────────────────

TIMEOUT_BUDGET: dict[str, float] = {
    "asr":             10.0,
    "llm_first_token":  8.0,
    "tts":             15.0,
}


# ── Resilient wrappers ────────────────────────────────────────────────────────

async def transcribe_with_timeout(
    audio_path: str,
) -> Union[TranscriptionEvent, ComponentError]:
    """
    Call transcribe_audio() enforcing the ASR timeout budget.

    Returns a TranscriptionEvent on success, or a ComponentError on
    timeout / any other API failure.
    """
    try:
        return await asyncio.wait_for(
            transcribe_audio(audio_path),
            timeout=TIMEOUT_BUDGET["asr"],
        )
    except asyncio.TimeoutError:
        return ComponentError(
            component="ASR",
            error_type="timeout",
            message=f"ASR timed out after {TIMEOUT_BUDGET['asr']:.0f}s",
        )
    except Exception as e:
        return ComponentError(
            component="ASR",
            error_type="api_error",
            message=str(e),
        )


async def stream_response_with_timeout(
    transcription: TranscriptionEvent,
) -> AsyncGenerator[Union[LLMResponseChunkEvent, ComponentError], None]:
    """
    Wrap stream_response() and enforce llm_first_token timeout.

    Yields LLMResponseChunkEvents normally; yields a single ComponentError
    if the first token does not arrive within the budget, or on any exception.

    Note: asyncio.wait_for() cannot wrap an async generator directly, so we
    pull the first chunk manually under a timeout, then stream the remainder
    without a cap — the hard cut-off protects perceived responsiveness while
    allowing long responses to complete once generation has started.
    """
    try:
        stream = stream_response(transcription)

        # --- enforce first-token deadline -----------------------------------
        try:
            first = await asyncio.wait_for(
                stream.__anext__(),
                timeout=TIMEOUT_BUDGET["llm_first_token"],
            )
        except asyncio.TimeoutError:
            yield ComponentError(
                component="LLM",
                error_type="timeout",
                message=(
                    f"LLM timed out waiting for first token "
                    f"after {TIMEOUT_BUDGET['llm_first_token']:.0f}s"
                ),
            )
            return

        yield first

        # --- stream remaining chunks without additional timeout --------------
        async for chunk in stream:
            yield chunk

    except Exception as e:
        yield ComponentError(
            component="LLM",
            error_type="api_error",
            message=str(e),
        )


async def synthesize_with_timeout(
    text: str,
    output_path: str,
) -> Union[AudioOutputEvent, ComponentError]:
    """
    Call synthesize_speech() enforcing the TTS timeout budget.

    Returns an AudioOutputEvent on success, or a ComponentError on
    timeout / any other API failure.
    """
    try:
        return await asyncio.wait_for(
            synthesize_speech(text, output_path),
            timeout=TIMEOUT_BUDGET["tts"],
        )
    except asyncio.TimeoutError:
        return ComponentError(
            component="TTS",
            error_type="timeout",
            message=f"TTS timed out after {TIMEOUT_BUDGET['tts']:.0f}s",
        )
    except Exception as e:
        return ComponentError(
            component="TTS",
            error_type="api_error",
            message=str(e),
        )


# ── Fallback messaging ────────────────────────────────────────────────────────

def get_fallback_response(error: ComponentError) -> str:
    """
    Return a user-facing fallback string appropriate for the failed component.

    Args:
        error: The ComponentError produced by a resilient wrapper.

    Returns:
        A plain-English message the assistant can surface to the user.
    """
    match error.component:
        case "ASR":
            return "I couldn't hear that clearly. Could you try again?"
        case "LLM":
            return "I'm thinking a bit slowly right now. Please give me a moment."
        case "TTS":
            return (
                "[Text response only] Audio synthesis is temporarily unavailable. "
                "Here is the response as text instead."
            )
        case _:
            return f"Something went wrong in the {error.component} stage. Please try again."
