"""Immutable R1 data models."""

from .event import EventType, QualityStatus, StudentProcessEvent
from .legacy import LegacyQuestionRegion, legacy_event, legacy_point, legacy_rectangle, legacy_stroke
from .point import NormalizedPoint, PenState, Point, is_canonical_v1_eligible, validate_canonical_point_for_export
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
    "LegacyQuestionRegion",
    "StudentProcessEvent",
    "Stroke",
    "legacy_event",
    "legacy_point",
    "legacy_rectangle",
    "legacy_stroke",
    "is_canonical_v1_eligible",
    "validate_canonical_point_for_export",
]
