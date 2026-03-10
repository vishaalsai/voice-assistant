"""
Entry point for the voice assistant application.
Initializes the pipeline, loads environment variables, and drives the main
conversation loop by coordinating ASR, LLM, and TTS stages.
"""

import asyncio
import sys
import uuid
from pathlib import Path

# Force UTF-8 output so emoji progress labels render correctly on Windows,
# where the default console codec (cp1252) cannot encode them.
sys.stdout.reconfigure(encoding="utf-8")

from pipeline.events import AudioInputEvent, PipelineCompleteEvent, PipelineRun
from instrumentation.tracker import analyze_latency, pretty_print_latency
from resilience.handlers import (
    ComponentError,
    TIMEOUT_BUDGET,
    get_fallback_response,
    stream_response_with_timeout,
    synthesize_with_timeout,
    transcribe_with_timeout,
)
from replay.replay import save_replay


async def run_pipeline(audio_path: str) -> PipelineRun:
    """
    Run one full ASR → LLM → TTS pass for a given audio file.
    All three component calls use resilient wrappers that enforce per-stage
    timeouts and return ComponentErrors on failure instead of raising.

    Special test mode: pass audio_path="test_timeout" to force an ASR timeout
    and verify graceful degradation without a real audio file.

    Args:
        audio_path: Path to the input .wav file, or "test_timeout".

    Returns:
        PipelineRun containing every event (and any errors) from this pass.
    """
    run = PipelineRun()

    # ── Test mode: force an ASR timeout to exercise degradation paths ──────
    if audio_path == "test_timeout":
        print("⚠️  Test-timeout mode: setting ASR budget to 0.001s to force timeout")
        TIMEOUT_BUDGET["asr"] = 0.001

    # ── Step 1 — Audio input ───────────────────────────────────────────────
    print("🎤 Step 1/4: Reading audio input...")
    audio_event = AudioInputEvent(audio_path=audio_path, duration_seconds=0.0)
    run.add(audio_event)

    # ── Step 2 — Transcription ─────────────────────────────────────────────
    print("📝 Step 2/4: Transcribing audio...")
    asr_result = await transcribe_with_timeout(audio_path)

    if isinstance(asr_result, ComponentError):
        print(f"   [ASR ERROR] {asr_result.error_type}: {asr_result.message}")
        print(f"   Fallback: {get_fallback_response(asr_result)}")
        # Cannot continue without a transcript — abort the pipeline here.
        return run

    transcription_event = asr_result
    run.add(transcription_event)
    print(f"   Transcription: \"{transcription_event.text}\"")

    # Save a replay snapshot immediately after successful transcription so
    # any run can be reproduced even if LLM or TTS fail later.
    if audio_path != "test_timeout":
        run_id = uuid.uuid4().hex[:6]
        save_replay(audio_path, run_id)
        print(f"💾 Replay saved as run_id: {run_id}")

    # ── Step 3 — LLM streaming response ───────────────────────────────────
    print("🤖 Step 3/4: Getting AI response...")
    print("   Response: ", end="", flush=True)

    full_response = ""
    llm_failed = False

    async for chunk in stream_response_with_timeout(transcription_event):
        if isinstance(chunk, ComponentError):
            print()  # newline after partial streamed output (if any)
            print(f"   [LLM ERROR] {chunk.error_type}: {chunk.message}")
            print(f"   Fallback: {get_fallback_response(chunk)}")
            llm_failed = True
            break

        run.add(chunk)
        if not chunk.is_final:
            print(chunk.chunk_text, end="", flush=True)
        else:
            full_response = chunk.full_response_so_far

    if llm_failed:
        # Cannot synthesise without a response — abort.
        return run

    # ── Step 4 — Speech synthesis ──────────────────────────────────────────
    print("\n🔊 Step 4/4: Synthesising speech...")
    output_path = "output_response.mp3"
    tts_result = await synthesize_with_timeout(full_response, output_path)

    if isinstance(tts_result, ComponentError):
        print(f"   [TTS ERROR] {tts_result.error_type}: {tts_result.message}")
        print(f"   Fallback: {get_fallback_response(tts_result)}")
        # TTS failure is a degraded-but-valid outcome: we have a text response,
        # so we continue to summary and latency reporting rather than aborting.
        output_audio_path = ""
    else:
        audio_output_event = tts_result
        run.add(audio_output_event)
        output_audio_path = audio_output_event.audio_path
        print(f"✅ Done! Audio saved to {output_path}")

    # ── Final summary event ────────────────────────────────────────────────
    complete_event = PipelineCompleteEvent(
        input_audio_path=audio_path,
        transcription=transcription_event.text,
        full_response=full_response,
        output_audio_path=output_audio_path,
    )
    run.add(complete_event)

    print(f"\nPipeline complete. Events captured: {len(run.events)}")

    # ── Latency analysis — runs regardless of TTS success ─────────────────
    try:
        breakdown = analyze_latency(run)
        print()
        pretty_print_latency(breakdown)
    except ValueError as e:
        print(f"\n[Tracker] Skipping latency breakdown: {e}")

    return run


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_input.wav"
    asyncio.run(run_pipeline(path))
