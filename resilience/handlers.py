"""
Error and retry handlers.
Implements exponential back-off retries, circuit-breaker logic, and graceful
degradation fallbacks for ASR, LLM, and TTS API calls.
"""
