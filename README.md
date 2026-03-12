---
title: Voice Assistant Real-Time AI Pipeline
emoji: 🎙️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Voice Assistant — Real-Time Multimodal AI Pipeline

A real-time voice assistant pipeline built in Python demonstrating end-to-end streaming AI architecture, latency decomposition, resilience engineering, and deterministic replay debugging.

```
Audio Input → ASR (Whisper) → LLM (Claude Haiku) → TTS (OpenAI) → Audio Output
```

## Quick Reference

| | |
|---|---|
| Live Demo | huggingface.co/spaces/vishaalsai29/voice-assistant |

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Audio .wav │───▶│  Whisper    │───▶│  Claude     │───▶│  OpenAI TTS │
│  Input      │    │  ASR        │    │  Haiku      │    │  tts-1      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                        │                   │                   │
                 TranscriptionEvent  LLMChunkEvent       AudioOutputEvent
                  (timestamped)       (timestamped)       (timestamped)
                        └───────────────────┴───────────────────┘
                                      PipelineRun
                               (collects all events for
                                latency analysis)
```

Each stage emits a typed, timestamped dataclass event into a `PipelineRun`. After the run completes, `analyze_latency()` walks the event list and derives per-component timing with millisecond precision — no instrumentation wrappers, no monkey-patching.

---

## Latency Breakdown (Real Measurements)

Averaged across multiple runs on a standard consumer internet connection:

| Component         | Latency   | % of Total |
|-------------------|-----------|------------|
| ASR (Whisper)     | ~3,200ms  | ~29%       |
| LLM First Token   | ~935ms    | ~8%        |
| LLM Streaming     | ~698ms    | ~6%        |
| TTS Synthesis     | ~6,400ms  | ~57%       |
| Pipeline Overhead | ~0ms      | ~0%        |
| **TOTAL**         | **~11.2s**| **100%**   |

### Bottleneck Analysis

**TTS synthesis is the dominant bottleneck at ~57% of total latency.**

The architectural reason is that `tts-1` synthesises the complete response text before returning any audio. The entire LLM output must finish streaming before the TTS call can even begin, and then the full audio file must be written to disk before playback can start. This means two full sequential blocking operations before the user hears anything.

**What would fix it:** Chunk-level TTS streaming — synthesise each sentence as Claude produces it rather than waiting for the full response. With this approach, the first audio chunk could begin playing within ~1–2 seconds of the user finishing their question, reducing perceived latency by 4–5 seconds. This is the single highest-leverage optimisation available.

**LLM first token at ~935ms** is reasonable for a cloud API call to Claude Haiku and reflects network round-trip plus model prefill time. This could be reduced further with a locally-hosted model (e.g. Ollama with Llama 3) to eliminate the network hop entirely.

---

## Resilience Engineering

Every component call is wrapped in an explicit timeout budget. On failure, the pipeline degrades gracefully rather than crashing.

| Component | Failure Mode | Timeout | Degradation Strategy |
|-----------|--------------|---------|----------------------|
| ASR | API error / timeout | 10s | Returns `"I couldn't hear that clearly. Could you try again?"` and aborts the run |
| LLM | Timeout waiting for first token | 8s | Returns `"I'm thinking a bit slowly right now. Please give me a moment."` and aborts the run |
| TTS | API error / timeout | 15s | Returns a text-only fallback; pipeline continues to completion |

### Degradation Philosophy

- **ASR and LLM failures abort the pipeline.** There is no meaningful output to produce without a transcript or a response — silence would be worse than an error message.
- **TTS failure degrades gracefully.** A text response is a valid partial output. The `PipelineCompleteEvent` is still recorded and latency analysis still runs, so the failure is observable without losing the run's data.
- **Nothing ever blocks indefinitely.** Every component has an explicit `asyncio.wait_for()` timeout budget defined in `resilience/handlers.py`. The system is safe to run in production without a separate watchdog process.

To test the degradation path without breaking anything:

```bash
python main.py test_timeout   # forces ASR timeout in <1ms, prints fallback
```

---

## Replay Mode

Every successful pipeline run automatically snapshots its input audio:

```
replay/
└── recorded_inputs/
    ├── a3f9c1.wav       ← copy of the original input
    └── a3f9c1.json      ← metadata: run_id, timestamp, original path
```

Any past run can be re-executed deterministically:

```bash
python replay/run_replay.py list            # list all saved runs
python replay/run_replay.py a3f9c1          # re-execute a specific run
python replay/run_replay.py                 # print usage
```

**Why this matters:**

- Reproduces exact failure conditions without recreating live inputs — if a user reports a bug, you replay their audio rather than guessing what they said.
- Standard pattern in production real-time systems engineering (event sourcing, deterministic simulation).
- Lets you verify a fix against the exact input that triggered the bug, not a paraphrase of it.

---

## Project Structure

```
voice-assistant/
│
├── main.py                        Entry point; orchestrates the full pipeline
│
├── pipeline/
│   ├── events.py                  Dataclass event schema (AudioInputEvent,
│   │                              TranscriptionEvent, LLMResponseChunkEvent,
│   │                              AudioOutputEvent, PipelineCompleteEvent, PipelineRun)
│   ├── asr.py                     Whisper API wrapper — async, thread-offloaded
│   ├── llm.py                     Claude streaming async generator
│   └── tts.py                     OpenAI TTS with streaming HTTP response
│
├── resilience/
│   └── handlers.py                Timeout wrappers, ComponentError dataclass,
│                                  per-component budgets, fallback messages
│
├── instrumentation/
│   └── tracker.py                 analyze_latency() + pretty_print_latency()
│
├── replay/
│   ├── replay.py                  save_replay(), run_replay(), list_replays()
│   ├── run_replay.py              CLI entry point for the replay system
│   └── recorded_inputs/           Auto-populated on each run (.wav + .json)
│
├── dashboard/
│   └── index.html                 Self-contained latency dashboard (Chart.js);
│                                  open directly in a browser, no server needed
│
├── _gen_test_wav.py               One-off utility: generate test_input.wav via TTS
├── test_asr.py                    ASR stage smoke test
├── test_llm.py                    LLM streaming smoke test
├── test_tts.py                    TTS stage smoke test
│
├── .env.example                   API key template
├── requirements.txt               Python dependencies
└── README.md
```

---

## Setup

**Requirements:** Python 3.12+, ffmpeg

```bash
# 1. Clone the repo
git clone https://github.com/vishaalsai/voice-assistant.git
cd voice-assistant

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install ffmpeg (required by pydub for audio processing)
winget install ffmpeg          # Windows
brew install ffmpeg            # macOS
sudo apt install ffmpeg        # Ubuntu/Debian

# 4. Configure API keys
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...

# 5. Generate a test audio file and run the pipeline
python _gen_test_wav.py        # creates test_input.wav via OpenAI TTS
python main.py test_input.wav
```

To re-run a saved session:

```bash
python replay/run_replay.py list
python replay/run_replay.py {run_id}
```

To view the latency dashboard, open `dashboard/index.html` in any browser.

---

## What I Would Optimise Next

1. **Chunk-level TTS streaming** — Synthesise per sentence as Claude streams output rather than waiting for the full response. This would cut TTS latency from ~6.4s to near real-time and is the single highest-leverage change available given current measurements.

2. **Local ASR** — Replace the Whisper API with a local `whisper.cpp` instance. This eliminates the ~3.2s network round trip and removes the per-request cost, at the expense of GPU/CPU resources on the host machine.

3. **Live microphone input** — Replace `.wav` file input with real-time `PyAudio` capture so the assistant responds to live speech rather than pre-recorded files. This is the remaining gap between the current implementation and a true voice UX.

4. **Latency SLA alerts** — Track a rolling baseline for each component and trigger an alert when any stage exceeds 2× its baseline latency. This would surface model degradation, network issues, or API slowdowns before users notice them, making the system production-observable.

---

## Stack

| Layer | Technology |
|-------|-----------|
| ASR | OpenAI Whisper (`whisper-1`) |
| LLM | Anthropic Claude Haiku (`claude-haiku-4-5`) via streaming API |
| TTS | OpenAI TTS (`tts-1`, voice: `alloy`) |
| Async runtime | Python `asyncio` + `asyncio.to_thread` for blocking I/O |
| Event schema | Python `dataclasses` |
| Resilience | `asyncio.wait_for` timeout budgets per component |
| Dashboard | Chart.js (CDN), vanilla HTML/CSS/JS |
