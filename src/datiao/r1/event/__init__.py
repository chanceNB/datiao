"""Student process event detection for R1."""

from .detector import EVENT_ALGORITHM_VERSION, PROCESS_SCHEMA_VERSION, EventDetectionConfig, detect_student_process_events

__all__ = ["EVENT_ALGORITHM_VERSION", "PROCESS_SCHEMA_VERSION", "EventDetectionConfig", "detect_student_process_events"]
