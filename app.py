"""
Streamlit dashboard for the Voice Assistant pipeline.
Tabs: Live Demo · Latency Breakdown · Run History · Resilience Demo
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# ── API key loading ────────────────────────────────────────────────────────────
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
except:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Import guard: pipeline modules may fail if API keys are absent ─────────────
try:
    from main import run_pipeline
    _pipeline_available = True
except Exception as _pipeline_err:
    _pipeline_available = False
    _pipeline_err_msg = str(_pipeline_err)

try:
    from instrumentation.tracker import analyze_latency
    _tracker_available = True
except Exception:
    _tracker_available = False

try:
    import openai as _openai_mod
    _openai_available = True
except Exception:
    _openai_available = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Assistant — Real-Time AI Pipeline",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Voice Assistant — Real-Time AI Pipeline")
st.markdown(
    "_End-to-end latency decomposition · Graceful degradation · Replay mode_"
)

# ── Hardcoded showcase data ────────────────────────────────────────────────────
SHOWCASE = {
    "What is the speed of light?": {
        "transcription": "What is the speed of light and why does it matter in everyday life?",
        "response": (
            "Light travels at about 186,000 miles per second, or 300,000 kilometers "
            "per second. While that's impossibly fast for everyday experience, it matters because "
            "it limits how quickly information travels — it's why there's a tiny delay in video "
            "calls, and it's the foundation for GPS, fiber-optic internet, and basically all "
            "modern communication technology."
        ),
        "latency": {
            "asr_latency_ms": 2840,
            "llm_time_to_first_token_ms": 1055,
            "llm_total_streaming_ms": 697,
            "tts_latency_ms": 6338,
            "total_pipeline_ms": 10931,
            "pipeline_overhead_ms": 1,
        },
    },
    "How does GPS work?": {
        "transcription": "How does GPS work and why does it need Einstein's theory of relativity?",
        "response": (
            "GPS works by triangulating your position using signals from at least 4 "
            "satellites. Each satellite broadcasts its position and the exact time — your device "
            "calculates distance from each satellite by measuring signal delay, then solves for "
            "your location. It needs Einstein's relativity corrections because satellite clocks "
            "run slightly faster in orbit due to weaker gravity and slightly slower due to their "
            "speed — without correcting for both effects, GPS would drift by about 10km per day."
        ),
        "latency": {
            "asr_latency_ms": 3100,
            "llm_time_to_first_token_ms": 890,
            "llm_total_streaming_ms": 720,
            "tts_latency_ms": 6200,
            "total_pipeline_ms": 10910,
            "pipeline_overhead_ms": 0,
        },
    },
    "What is machine learning?": {
        "transcription": "What is machine learning and how is it different from traditional programming?",
        "response": (
            "Machine learning is a way of building software where instead of writing "
            "explicit rules, you show the system thousands of examples and it figures out the "
            "patterns itself. Traditional programming is like writing a recipe — you specify every "
            "step. Machine learning is like hiring someone who learns by eating thousands of dishes "
            "and eventually develops their own intuition for what tastes good."
        ),
        "latency": {
            "asr_latency_ms": 2950,
            "llm_time_to_first_token_ms": 1100,
            "llm_total_streaming_ms": 680,
            "tts_latency_ms": 6500,
            "total_pipeline_ms": 11230,
            "pipeline_overhead_ms": 0,
        },
    },
}

# ── Color palette (consistent across tabs) ────────────────────────────────────
COLORS = {
    "ASR":              "#3b82f6",   # blue
    "LLM First Token":  "#22c55e",   # green
    "LLM Streaming":    "#14b8a6",   # teal
    "TTS":              "#ef4444",   # red
}

LATENCY_LOG = Path("latency_log.jsonl")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_log() -> list[dict]:
    """Return all runs from latency_log.jsonl, or [] if absent."""
    if not LATENCY_LOG.exists():
        return []
    runs = []
    with LATENCY_LOG.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return runs


def _breakdown_to_components(b: dict) -> list[tuple[str, float]]:
    """Convert a latency breakdown dict to (label, ms) list for charting."""
    return [
        ("ASR",             b.get("asr_latency_ms", 0)),
        ("LLM First Token", b.get("llm_time_to_first_token_ms", 0)),
        ("LLM Streaming",   b.get("llm_total_streaming_ms", 0)),
        ("TTS",             b.get("tts_latency_ms", 0)),
    ]


def _stacked_bar(components: list[tuple[str, float]], title: str = "") -> go.Figure:
    """Return a horizontal stacked bar Plotly figure."""
    fig = go.Figure()
    for label, ms in components:
        fig.add_trace(go.Bar(
            name=label,
            x=[ms],
            y=["Pipeline"],
            orientation="h",
            marker_color=COLORS.get(label, "#888"),
            text=f"{label}<br>{ms:.0f} ms",
            textposition="inside",
            insidetextanchor="middle",
        ))
    fig.update_layout(
        barmode="stack",
        title=title,
        height=160,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(orientation="h", y=-0.3),
        xaxis_title="Latency (ms)",
        yaxis=dict(visible=False),
    )
    return fig


def _show_latency_results(breakdown: dict) -> None:
    """Render chart + table for a breakdown dict."""
    components = _breakdown_to_components(breakdown)
    total = breakdown.get("total_pipeline_ms", 1) or 1
    st.plotly_chart(_stacked_bar(components), use_container_width=True)
    rows = [(lbl, ms, ms / total * 100) for lbl, ms in components]
    st.table(
        {
            "Component": [r[0] for r in rows],
            "Latency (ms)": [f"{r[1]:.0f}" for r in rows],
            "Share": [f"{r[2]:.1f}%" for r in rows],
        }
    )


def _get_api_key(name: str) -> str | None:
    """Read an API key from st.secrets then os.environ."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎤 Live Demo", "📊 Latency Breakdown", "📈 Run History", "🛡️ Resilience Demo"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns(2)

    # ── Left: Run Live ─────────────────────────────────────────────────────────
    with left:
        st.subheader("Run Live")
        question = st.text_input("Type your question", placeholder="Ask anything…")
        run_btn = st.button("▶ Run Pipeline", key="run_live")

        if run_btn:
            if not question.strip():
                st.warning("Please type a question first.")
            elif not _pipeline_available:
                st.error(f"Pipeline unavailable: {_pipeline_err_msg}")
            elif not _openai_available:
                st.error("openai package not installed — cannot run live pipeline.")
            else:
                openai_key = _get_api_key("OPENAI_API_KEY")
                if not openai_key:
                    st.error("OPENAI_API_KEY not set. Add it to your environment or Hugging Face secrets.")
                else:
                    progress = st.progress(0, text="Transcribing…")

                    try:
                        import openai as _oa

                        # Step 0 — TTS of typed question → temp .wav
                        progress.progress(10, text="Synthesising input audio…")
                        client = _oa.OpenAI(api_key=openai_key)

                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp_path = tmp.name

                        with client.audio.speech.with_streaming_response.create(
                            model="tts-1",
                            voice="alloy",
                            input=question,
                            response_format="wav",
                        ) as resp:
                            resp.stream_to_file(tmp_path)

                        # Step 1-4 — real pipeline
                        progress.progress(25, text="Transcribing…")

                        run = asyncio.run(run_pipeline(tmp_path))

                        progress.progress(60, text="Thinking…")
                        time.sleep(0.2)
                        progress.progress(85, text="Synthesising…")
                        time.sleep(0.2)
                        progress.progress(100, text="Done ✓")

                        # Extract results from run events
                        from pipeline.events import (
                            TranscriptionEvent,
                            LLMResponseChunkEvent,
                            AudioOutputEvent,
                            PipelineCompleteEvent,
                        )

                        transcription_text = next(
                            (e.text for e in run.events if isinstance(e, TranscriptionEvent)), ""
                        )
                        complete_evt = next(
                            (e for e in run.events if isinstance(e, PipelineCompleteEvent)), None
                        )
                        ai_response = complete_evt.full_response if complete_evt else ""
                        output_audio = complete_evt.output_audio_path if complete_evt else ""

                        st.markdown("**Transcription**")
                        st.info(transcription_text or question)
                        st.markdown("**AI Response**")
                        st.success(ai_response)

                        if output_audio and Path(output_audio).exists():
                            st.audio(output_audio)

                        if _tracker_available:
                            try:
                                breakdown = analyze_latency(run)
                                st.markdown("**Latency breakdown**")
                                _show_latency_results(breakdown)
                            except ValueError:
                                pass

                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

    # ── Right: Showcase ────────────────────────────────────────────────────────
    with right:
        st.subheader("Showcase (Pre-recorded)")
        chosen = st.selectbox(
            "Select an example question",
            list(SHOWCASE.keys()),
            key="showcase_select",
        )
        show_btn = st.button("▶ Show Example", key="show_example")

        if show_btn:
            data = SHOWCASE[chosen]
            st.caption("Pre-recorded run — results shown instantly")

            st.markdown("**Transcription**")
            st.info(data["transcription"])
            st.markdown("**AI Response**")
            st.success(data["response"])

            st.markdown("**Latency breakdown**")
            _show_latency_results(data["latency"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LATENCY BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Latency Breakdown — Most Recent Run")

    runs = _load_log()
    if runs:
        latest = runs[-1]
    else:
        # Fall back to first showcase example
        latest = SHOWCASE["What is the speed of light?"]["latency"]
        st.info("No latency_log.jsonl found — showing sample data from a pre-recorded run.")

    components = _breakdown_to_components(latest)
    total_ms = latest.get("total_pipeline_ms", 1) or 1

    st.plotly_chart(_stacked_bar(components, title="Pipeline Latency Breakdown"), use_container_width=True)

    # Styled table
    rows = [(lbl, ms, ms / total_ms * 100) for lbl, ms in components]
    st.table({
        "Component":    [r[0] for r in rows],
        "Latency (ms)": [f"{r[1]:.0f}" for r in rows],
        "% of Total":   [f"{r[2]:.1f}%" for r in rows],
    })

    # Bottleneck analysis
    st.markdown("### Bottleneck Analysis")
    biggest_label, biggest_ms = max(components, key=lambda x: x[1])
    biggest_pct = biggest_ms / total_ms * 100
    llm_ttft = latest.get("llm_time_to_first_token_ms", 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Biggest Bottleneck", biggest_label, f"{biggest_pct:.1f}% of total")
    m2.metric("Total Pipeline Time", f"{total_ms:.0f} ms", f"{total_ms/1000:.2f} s")
    m3.metric("LLM First Token (AI Thinking)", f"{llm_ttft:.0f} ms")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RUN HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Run History")

    all_runs = _load_log()

    if len(all_runs) < 2:
        st.info("Run the live demo a few times to see latency trends here.")
    else:
        n = len(all_runs)
        run_labels = [r.get("run_id", str(i + 1)) for i, r in enumerate(all_runs)]

        fig = go.Figure()
        component_keys = [
            ("ASR",             "asr_latency_ms"),
            ("LLM First Token", "llm_time_to_first_token_ms"),
            ("LLM Streaming",   "llm_total_streaming_ms"),
            ("TTS",             "tts_latency_ms"),
        ]
        for label, key in component_keys:
            values = [r.get(key, 0) for r in all_runs]
            fig.add_trace(go.Scatter(
                x=run_labels,
                y=values,
                mode="lines+markers",
                name=label,
                line=dict(color=COLORS[label]),
            ))

        fig.update_layout(
            title=f"Latency Over Time — {n} runs",
            xaxis_title="Run ID",
            yaxis_title="Latency (ms)",
            height=400,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        totals = [r.get("total_pipeline_ms", 0) for r in all_runs]
        c1, c2, c3 = st.columns(3)
        c1.metric("Min Total Latency", f"{min(totals):.0f} ms")
        c2.metric("Max Total Latency", f"{max(totals):.0f} ms")
        c3.metric("Avg Total Latency", f"{sum(totals)/len(totals):.0f} ms")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESILIENCE DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Resilience Demo — Simulated Failures")

    COMPONENTS = [
        {
            "name": "ASR",
            "timeout": "10 s",
            "fallback": "I couldn't hear that clearly. Could you try again?",
            "still_runs": [],
            "aborts": ["LLM", "TTS"],
            "note": "Without a transcript the pipeline cannot continue — LLM and TTS are skipped.",
        },
        {
            "name": "LLM",
            "timeout": "8 s",
            "fallback": "I'm thinking a bit slowly right now. Please give me a moment.",
            "still_runs": ["ASR"],
            "aborts": ["TTS"],
            "note": "ASR completed successfully. Without an LLM response TTS is skipped.",
        },
        {
            "name": "TTS",
            "timeout": "15 s",
            "fallback": (
                "[Text response only] Audio synthesis is temporarily unavailable. "
                "Here is the response as text instead."
            ),
            "still_runs": ["ASR", "LLM"],
            "aborts": [],
            "note": "ASR and LLM completed. TTS failure is a degraded-but-valid outcome — text response is returned.",
        },
    ]

    cols = st.columns(3)
    for col, comp in zip(cols, COMPONENTS):
        with col:
            st.markdown(f"### {comp['name']}")
            st.caption(f"Timeout budget: **{comp['timeout']}**")

            if st.button(f"Simulate {comp['name']} Failure", key=f"sim_{comp['name']}"):
                with st.spinner(f"Simulating {comp['name']} timeout…"):
                    time.sleep(1)

                st.warning(f"**Fallback triggered:**\n\n{comp['fallback']}")
                st.success("✅ Pipeline handled failure gracefully — no crash")

                if comp["still_runs"]:
                    st.markdown(
                        "**Completed:** " + " · ".join(f"✅ {c}" for c in comp["still_runs"])
                    )
                st.markdown(f"**Skipped:** ❌ {comp['name']}" + (
                    " · " + " · ".join(f"❌ {c}" for c in comp["aborts"])
                    if comp["aborts"] else ""
                ))
                st.caption(comp["note"])

    st.divider()
    with st.expander("How resilience works in this pipeline"):
        st.markdown(
            """
- **Per-stage timeout budgets** — each component (ASR 10 s, LLM 8 s first token, TTS 15 s) \
runs inside `asyncio.wait_for()`. If it exceeds its budget a `ComponentError` is returned \
instead of raising, so the caller always gets a typed result it can inspect.

- **Graceful degradation over hard failure** — the pipeline treats each stage's outcome \
independently. A TTS failure still delivers a text response; only an ASR failure (no \
transcript) forces a full abort, because there is literally nothing to pass forward.

- **No silent crashes** — every `ComponentError` carries a `component`, `error_type`, and \
`message` field. The app surfaces the appropriate fallback string to the user and continues \
tracking latency for stages that did complete, preserving observability even in degraded runs.
"""
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with OpenAI Whisper · Anthropic Claude · OpenAI TTS")
