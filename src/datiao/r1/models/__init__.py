"""Immutable R1 data models."""

from .event import EventType, QualityStatus, StudentProcessEvent
from .legacy import legacy_event, legacy_point, legacy_rectangle, legacy_stroke
from .point import NormalizedPoint, PenState, Point
from .question_region import QuestionRegion
from .stroke import BoundingBox, Stroke

__all__ = [
    "BoundingBox",
    "EventType",
    "QualityStatus",
    "NormalizedPoint",
    "PenState",
    "Point",
    "QuestionRegion",
    "StudentProcessEvent",
    "Stroke",
    "legacy_event",
    "legacy_point",
    "legacy_rectangle",
    "legacy_stroke",
]
