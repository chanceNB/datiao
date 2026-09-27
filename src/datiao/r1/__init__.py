"""R1 point-matrix process foundation."""

from .event import detect_student_process_events
from .mapper import StrokeMapping, map_strokes_to_regions
from .models import Point, QuestionRegion, StudentProcessEvent, Stroke
from .trace import EvidenceTrace, resolve_event_trace, resolve_event_traces

__all__ = [
    "EvidenceTrace",
    "Point",
    "QuestionRegion",
    "StudentProcessEvent",
    "Stroke",
    "StrokeMapping",
    "detect_student_process_events",
    "map_strokes_to_regions",
    "resolve_event_trace",
    "resolve_event_traces",
]
