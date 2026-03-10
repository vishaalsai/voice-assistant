"""
Metrics and latency tracker.
Records per-stage timing, token counts, and error rates, and exposes
aggregated stats for display in the dashboard or export to monitoring systems.
"""

from pipeline.events import (
    AudioInputEvent,
    AudioOutputEvent,
    LLMResponseChunkEvent,
    PipelineRun,
    TranscriptionEvent,
)


def analyze_latency(run: PipelineRun) -> dict:
    """
    Derive per-stage latency values (in milliseconds) from the timestamps
    embedded in a completed PipelineRun's event list.

    Returns a dict with keys:
        asr_latency_ms              – Whisper network + decode time
        llm_time_to_first_token_ms  – LLM scheduling + prefill latency
        llm_total_streaming_ms      – Full token generation window
        tts_latency_ms              – TTS synthesis + write-to-disk time
        total_pipeline_ms           – Wall-clock from audio-in to audio-out
        pipeline_overhead_ms        – Unmeasured gaps (network, Python, I/O)

    Raises:
        ValueError if any required event type is missing from the run.
    """
    def _ms(start: float, end: float) -> float:
        return round((end - start) * 1000, 2)

    # --- Locate required events ----------------------------------------

    audio_in = next(
        (e for e in run.events if isinstance(e, AudioInputEvent)), None
    )
    transcription = next(
        (e for e in run.events if isinstance(e, TranscriptionEvent)), None
    )
    # First non-final chunk marks when the first token arrived.
    first_chunk = next(
        (e for e in run.events
         if isinstance(e, LLMResponseChunkEvent) and not e.is_final),
        None,
    )
    # The sentinel chunk (is_final=True) closes the streaming window.
    final_chunk = next(
        (e for e in reversed(run.events)
         if isinstance(e, LLMResponseChunkEvent) and e.is_final),
        None,
    )
    audio_out = next(
        (e for e in run.events if isinstance(e, AudioOutputEvent)), None
    )

    missing = [
        name for name, val in [
            ("AudioInputEvent", audio_in),
            ("TranscriptionEvent", transcription),
            ("first LLMResponseChunkEvent (is_final=False)", first_chunk),
            ("final LLMResponseChunkEvent (is_final=True)", final_chunk),
            ("AudioOutputEvent", audio_out),
        ]
        if val is None
    ]
    if missing:
        raise ValueError(f"analyze_latency: run is missing events: {missing}")

    # --- Calculate latencies -------------------------------------------

    asr_ms          = _ms(audio_in.timestamp,    transcription.timestamp)
    first_token_ms  = _ms(transcription.timestamp, first_chunk.timestamp)
    streaming_ms    = _ms(first_chunk.timestamp,  final_chunk.timestamp)
    tts_ms          = _ms(final_chunk.timestamp,  audio_out.timestamp)
    total_ms        = _ms(audio_in.timestamp,     audio_out.timestamp)
    component_sum   = asr_ms + first_token_ms + streaming_ms + tts_ms
    overhead_ms     = round(total_ms - component_sum, 2)

    return {
        "asr_latency_ms":             asr_ms,
        "llm_time_to_first_token_ms": first_token_ms,
        "llm_total_streaming_ms":     streaming_ms,
        "tts_latency_ms":             tts_ms,
        "total_pipeline_ms":          total_ms,
        "pipeline_overhead_ms":       overhead_ms,
    }


def pretty_print_latency(breakdown: dict) -> None:
    """
    Print a formatted latency breakdown table to stdout.

    Args:
        breakdown: Dict returned by analyze_latency().
    """
    total = breakdown["total_pipeline_ms"]

    def pct(value: float) -> str:
        return f"{round(value / total * 100):>3}%" if total else "  N/A"

    def fmt_ms(value: float) -> str:
        return f"{value:>6.0f}ms"

    rows = [
        ("ASR (Whisper)",     breakdown["asr_latency_ms"]),
        ("LLM First Token",   breakdown["llm_time_to_first_token_ms"]),
        ("LLM Streaming",     breakdown["llm_total_streaming_ms"]),
        ("TTS Synthesis",     breakdown["tts_latency_ms"]),
        ("Pipeline Overhead", breakdown["pipeline_overhead_ms"]),
    ]

    # Dynamically size the inner width to the widest row label.
    col_w = max(len(label) for label, _ in rows) + 2  # label column width
    inner_w = col_w + 16  # label + ms value + pct value + padding
    bar = "═" * inner_w

    print(f"╔{bar}╗")
    print(f"║{'  LATENCY BREAKDOWN':<{inner_w}}║")
    print(f"╠{bar}╣")
    for label, value in rows:
        cell = f"  {label:<{col_w}}{fmt_ms(value)}  {pct(value)}"
        print(f"║{cell:<{inner_w}}║")
    print(f"╠{bar}╣")
    total_cell = f"  {'TOTAL':<{col_w}}{fmt_ms(total)}  100%"
    print(f"║{total_cell:<{inner_w}}║")
    print(f"╚{bar}╝")
