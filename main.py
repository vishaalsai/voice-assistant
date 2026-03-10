"""
Entry point for the voice assistant application.
Initializes the pipeline, loads environment variables, and drives the main
conversation loop by coordinating ASR, LLM, and TTS stages.
"""

import asyncio
import sys
from pathlib import Path

# Force UTF-8 output so emoji progress labels render correctly on Windows,
# where the default console codec (cp1252) cannot encode them.
sys.stdout.reconfigure(encoding="utf-8")

from pipeline.asr import transcribe_audio
from pipeline.llm import stream_response
from pipeline.tts import synthesize_speech
from pipeline.events import AudioInputEvent, PipelineCompleteEvent, PipelineRun
from instrumentation.tracker import analyze_latency, pretty_print_latency


async def run_pipeline(audio_path: str) -> PipelineRun:
    """
    Run one full ASR → LLM → TTS pass for a given audio file.

    Args:
        audio_path: Path to the input .wav file to transcribe.

    Returns:
        PipelineRun containing every event produced during this pass.
    """
    run = PipelineRun()

    # ------------------------------------------------------------------
    # Step 1 — Audio input
    # ------------------------------------------------------------------
    print("🎤 Step 1/4: Reading audio input...")
    audio_event = AudioInputEvent(audio_path=audio_path, duration_seconds=0.0)
    run.add(audio_event)

    # ------------------------------------------------------------------
    # Step 2 — Transcription
    # ------------------------------------------------------------------
    print("📝 Step 2/4: Transcribing audio...")
    transcription_event = await transcribe_audio(audio_path)
    run.add(transcription_event)
    print(f"   Transcription: \"{transcription_event.text}\"")

    # ------------------------------------------------------------------
    # Step 3 — LLM streaming response
    # ------------------------------------------------------------------
    print("🤖 Step 3/4: Getting AI response...")
    print("   Response: ", end="", flush=True)

    full_response = ""
    async for chunk_event in stream_response(transcription_event):
        run.add(chunk_event)
        if not chunk_event.is_final:
            print(chunk_event.chunk_text, end="", flush=True)
        else:
            full_response = chunk_event.full_response_so_far

    # ------------------------------------------------------------------
    # Step 4 — Speech synthesis
    # ------------------------------------------------------------------
    print("\n🔊 Step 4/4: Synthesising speech...")
    output_path = "output_response.mp3"
    audio_output_event = await synthesize_speech(full_response, output_path)
    run.add(audio_output_event)
    print(f"✅ Done! Audio saved to {output_path}")

    # ------------------------------------------------------------------
    # Final summary event
    # ------------------------------------------------------------------
    complete_event = PipelineCompleteEvent(
        input_audio_path=audio_path,
        transcription=transcription_event.text,
        full_response=full_response,
        output_audio_path=audio_output_event.audio_path,
    )
    run.add(complete_event)

    print(f"\nPipeline complete. Events captured: {len(run.events)}")

    # ------------------------------------------------------------------
    # Latency analysis
    # ------------------------------------------------------------------
    breakdown = analyze_latency(run)
    print()
    pretty_print_latency(breakdown)

    return run


if __name__ == "__main__":
    wav = sys.argv[1] if len(sys.argv) > 1 else "test_input.wav"
    if not Path(wav).exists():
        print(f"[ERROR] Audio file not found: {wav}")
        print("Generate it first:  python _gen_test_wav.py")
        sys.exit(1)
    asyncio.run(run_pipeline(wav))
