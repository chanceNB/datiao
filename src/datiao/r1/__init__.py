"""R1 point-matrix process foundation."""

from .event import EventDetectionConfig, detect_student_process_events
from .mapper import QuestionMappingConfig, StrokeMapping, map_strokes_to_regions
from .models import Point, QuestionRegion, StudentProcessEvent, Stroke
from .pipeline import R1ProcessResult, run_r1_pipeline
from .replay import PageReplay, ReplayFrame, build_page_replay
from .trace import EvidenceTrace, resolve_event_trace, resolve_event_traces
from .verification import (
    ManualVerificationRecord,
    ManualVerificationSummary,
    apply_verification_label,
    build_verification_template,
    summarize_verification,
)

__all__ = [
    "EvidenceTrace",
    "Point",
    "PageReplay",
    "QuestionRegion",
    "R1ProcessResult",
    "ReplayFrame",
    "StudentProcessEvent",
    "Stroke",
    "StrokeMapping",
    "EventDetectionConfig",
    "QuestionMappingConfig",
    "detect_student_process_events",
    "map_strokes_to_regions",
    "run_r1_pipeline",
    "resolve_event_trace",
    "resolve_event_traces",
    "ManualVerificationRecord",
    "ManualVerificationSummary",
    "apply_verification_label",
    "build_page_replay",
    "build_verification_template",
    "summarize_verification",
]
