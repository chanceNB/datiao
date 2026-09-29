"""Explicit adapters for pre-V1 internal objects.

These helpers are intentionally separate from V1 model validation. They are
migration tools only and never change the fields emitted by standard models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .event import StudentProcessEvent
from .point import Point
from .question_region import QuestionRegion
from .stroke import BoundingBox, Stroke


def legacy_point(data: Mapping[str, Any]) -> Point:
    normalized = data.get("normalized", {})
    if hasattr(normalized, "model_dump"):
        normalized = normalized.model_dump()
    return Point(
        schema_version="1.0.0", point_id=str(data["point_id"]), session_id=str(data["session_id"]),
        participant_id=data.get("participant_id"), task_segment_id=data.get("task_segment_id"),
        device_id=data.get("device_id"), page_id=data.get("page_id"), sequence=int(data.get("sequence", data.get("raw_index", 0))),
        timestamp_ms=normalized.get("timestamp_ms"), x_raw=normalized.get("x"), y_raw=normalized.get("y"),
        x_mm=None, y_mm=None, x_norm=None, y_norm=None, pressure_raw=normalized.get("pressure"), pressure_norm=None,
        pen_state_raw=None, pen_state="UNKNOWN", source_file=data.get("source_file"),
        source_index=int(data.get("raw_index", 0)), raw_order=int(data.get("raw_index", 0)), processed_order=int(data.get("raw_index", 0)),
        source_payload=dict(data), quality_flags=tuple(data.get("quality_flags", ())),
    )


def legacy_stroke(data: Mapping[str, Any]) -> Stroke:
    start = data.get("start_timestamp_ms")
    end = data.get("end_timestamp_ms")
    raw_order = tuple(data.get("raw_order", ()))
    return Stroke(
        schema_version="1.0.0", stroke_id=str(data["stroke_id"]), session_id=str(data["session_id"]),
        participant_id=data.get("participant_id"), task_segment_id=data.get("task_segment_id"), page_id=data.get("page_id"),
        point_refs=raw_order, raw_order=raw_order, processed_order=tuple(data.get("processed_order", raw_order)),
        start_time_ms=start, end_time_ms=end, duration_ms=(end - start if start is not None and end is not None else None),
        bbox=BoundingBox.model_validate(data["bbox"]), quality_flags=tuple(data.get("quality_flags", ())),
        algorithm_version="r1-stroke-rule-v1", provenance={"legacy_adapter": True},
    )


def legacy_rectangle(*, region_id: str, page_id: str, question_id: str, x: float, y: float, width: float, height: float, priority: int = 0) -> QuestionRegion:
    if width < 0 or height < 0:
        raise ValueError("rectangle width and height must be non-negative")
    return QuestionRegion(
        schema_version="1.0.0", region_id=region_id, page_id=page_id, question_id=question_id,
        region_type="rectangle", polygon_norm=((x, y), (x + width, y), (x + width, y + height), (x, y + height)),
        priority=priority, coordinate_space="legacy",
    )


def legacy_event(data: Mapping[str, Any]) -> StudentProcessEvent:
    occurred = data.get("occurred_at_ms")
    return StudentProcessEvent(
        schema_version="1.0.0", event_id=str(data["event_id"]), event_type=data["event_type"], session_id=str(data["session_id"]),
        task_segment_id=data.get("task_segment_id"), participant_id=data.get("participant_id"), question_id=data.get("question_id"),
        page_id=data.get("page_id"), sequence=int(data.get("sequence", 0)), start_time_ms=occurred, end_time_ms=occurred,
        point_refs=tuple(data.get("source_point_ids", ())), stroke_refs=tuple(data.get("source_stroke_ids", ())),
        quality_status=data.get("quality_status", "VALID"), quality_flags=tuple(data.get("quality_flags", ())),
        algorithm_version="r1-event-rule-v0.1", provenance=data.get("source_provenance", {}), metadata=data.get("metadata", {}),
    )
