# Voice Assistant

A modular, observable voice assistant built on OpenAI and Anthropic APIs.

## Structure

| Path | Purpose |
|---|---|
| `main.py` | Application entry point |
| `pipeline/` | ASR → LLM → TTS processing stages |
| `instrumentation/` | Latency and token usage tracking |
| `resilience/` | Retry and circuit-breaker logic |
| `replay/` | Recorded inputs for offline testing |
| `dashboard/` | Browser-based metrics dashboard |

## Setup

```bash
cp .env.example .env
# Fill in your API keys in .env

pip install -r requirements.txt
python main.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (Whisper + GPT + TTS) |
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude) |
