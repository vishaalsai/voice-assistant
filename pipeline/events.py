"""
Event bus for the voice assistant pipeline.
Defines event types and provides a lightweight publish/subscribe mechanism
for decoupled communication between ASR, LLM, and TTS stages.
"""

import time
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class AudioInputEvent:
    """Fired when raw audio is received and ready for transcription."""
    audio_path: str
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TranscriptionEvent:
    """Fired when ASR produces a transcript from an audio input."""
    text: str
    audio_path: str  # reference back to the source AudioInputEvent
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMResponseChunkEvent:
    """Fired for each streamed chunk from the LLM; is_final marks the last chunk."""
    chunk_text: str
    is_final: bool
    full_response_so_far: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioOutputEvent:
    """Fired when TTS has synthesised audio and written it to disk."""
    audio_path: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineCompleteEvent:
    """Fired once per turn when all stages have finished successfully."""
    input_audio_path: str
    transcription: str
    full_response: str
    output_audio_path: str
    timestamp: float = field(default_factory=time.time)


# Union of all event types — useful for type annotations in the event bus.
PipelineEvent = Union[
    AudioInputEvent,
    TranscriptionEvent,
    LLMResponseChunkEvent,
    AudioOutputEvent,
    PipelineCompleteEvent,
]


@dataclass
class PipelineRun:
    """
    Collects every event emitted during a single end-to-end pipeline turn.
    Used by instrumentation/tracker.py for latency and token analysis.
    """
    events: List[PipelineEvent] = field(default_factory=list)

    def add(self, event: PipelineEvent) -> None:
        self.events.append(event)

    def events_of_type(self, event_type: type) -> List[PipelineEvent]:
        return [e for e in self.events if isinstance(e, event_type)]

    @property
    def complete_event(self) -> PipelineCompleteEvent | None:
        matches = self.events_of_type(PipelineCompleteEvent)
        return matches[-1] if matches else None
