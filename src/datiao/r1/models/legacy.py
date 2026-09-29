"""Explicit adapters for pre-V1 internal objects.

These helpers are intentionally separate from V1 model validation. They are
migration tools only and never change the fields emitted by standard models.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .event import StudentProcessEvent
from .point import Point
from .question_region import QuestionRegion, _point_on_segment
from .stroke import BoundingBox, Stroke


def legacy_point(data: Mapping[str, Any]) -> Point:
    normalized = data.get("normalized", {})
    if hasattr(normalized, "model_dump"):
        normalized = normalized.model_dump(mode="json")
    source_payload = dict(data)
    if "normalized" in source_payload:
        source_payload["normalized"] = normalized
    return Point(
        schema_version="1.0.0", point_id=str(data["point_id"]), session_id=str(data["session_id"]),
        participant_id=data.get("participant_id"), task_segment_id=data.get("task_segment_id"),
        device_id=data.get("device_id"), page_id=data.get("page_id"), sequence=int(data.get("sequence", data.get("raw_index", 0))),
        timestamp_ms=normalized.get("timestamp_ms"), x_raw=normalized.get("x"), y_raw=normalized.get("y"),
        x_mm=None, y_mm=None, x_norm=None, y_norm=None, pressure_raw=normalized.get("pressure"), pressure_norm=None,
        pen_state_raw=None, pen_state="UNKNOWN", source_file=data.get("source_file"),
        source_index=int(data.get("raw_index", 0)), raw_order=int(data.get("raw_index", 0)), processed_order=int(data.get("raw_index", 0)),
        source_payload=source_payload, quality_flags=tuple(data.get("quality_flags", ())),
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


class LegacyQuestionRegion(BaseModel):
    """Pre-V1 rectangle region; never serializes as a standard V1 region."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: str = "1.0.0"
    region_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    region_type: str = "rectangle"
    polygon_norm: tuple[tuple[float, float], ...]
    priority: int = 0
    coordinate_space: str = "legacy"

    @model_validator(mode="before")
    @classmethod
    def prepare_polygon(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if isinstance(values.get("polygon_norm"), list):
            values["polygon_norm"] = tuple(tuple(point) for point in values["polygon_norm"])
        return values

    @model_validator(mode="after")
    def validate_geometry(self) -> "LegacyQuestionRegion":
        if self.region_type != "rectangle" or len(self.polygon_norm) != 4:
            raise ValueError("legacy regions must be four-point rectangles")
        for point in self.polygon_norm:
            if len(point) != 2 or any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) for value in point):
                raise ValueError("legacy polygon points must contain finite numbers")
        return self

    def contains(self, x: float, y: float) -> bool:
        vertices = self.polygon_norm
        inside = False
        for index, (x1, y1) in enumerate(vertices):
            x2, y2 = vertices[(index + 1) % len(vertices)]
            if _point_on_segment(x, y, x1, y1, x2, y2):
                return True
            if (y1 > y) != (y2 > y):
                crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < crossing_x:
                    inside = not inside
        return inside


def legacy_rectangle(*, region_id: str, page_id: str, question_id: str, x: float, y: float, width: float, height: float, priority: int = 0) -> LegacyQuestionRegion:
    if width < 0 or height < 0:
        raise ValueError("rectangle width and height must be non-negative")
    return LegacyQuestionRegion(
        schema_version="1.0.0", region_id=region_id, page_id=page_id, question_id=question_id,
        region_type="rectangle", polygon_norm=((x, y), (x + width, y), (x + width, y + height), (x, y + height)),
        priority=priority, coordinate_space="legacy",
    )


def legacy_event(data: Mapping[str, Any]) -> StudentProcessEvent:
    occurred = data.get("occurred_at_ms")
    question_id = data.get("question_id")
    event_type = data["event_type"]
    if event_type in {"QUESTION_VISIT", "QUESTION_LEAVE", "RETURN", "REVISION_CANDIDATE"} and question_id is None:
        question_id = "UNKNOWN"
    quality_flags = list(data.get("quality_flags", ()))
    quality_status = data.get("quality_status", "VALID")
    if occurred is None:
        quality_status = "DEGRADED" if quality_status == "VALID" else quality_status
        quality_flags.append("TIME_UNAVAILABLE")
    return StudentProcessEvent(
        schema_version="1.0.0", event_id=str(data["event_id"]), event_type=event_type, session_id=str(data["session_id"]),
        task_segment_id=data.get("task_segment_id"), participant_id=data.get("participant_id"), question_id=question_id,
        page_id=data.get("page_id"), sequence=int(data.get("sequence", 0)), start_time_ms=occurred, end_time_ms=occurred,
        point_refs=tuple(data.get("source_point_ids", ())), stroke_refs=tuple(data.get("source_stroke_ids", ())),
        quality_status=quality_status, quality_flags=tuple(dict.fromkeys(quality_flags)),
        algorithm_version="r1-event-rule-v0.2", provenance=data.get("source_provenance", {}), metadata=data.get("metadata", {}),
    )
