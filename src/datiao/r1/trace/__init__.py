"""Evidence provenance resolution for R1 process events."""

from .resolver import (
    EvidenceTrace,
    TraceResolutionError,
    resolve_event_trace,
    resolve_event_traces,
)

__all__ = [
    "EvidenceTrace",
    "TraceResolutionError",
    "resolve_event_trace",
    "resolve_event_traces",
]
