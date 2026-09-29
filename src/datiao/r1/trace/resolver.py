"""Resolve process-event provenance back to strokes and raw points."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ..models import Point, Stroke, StudentProcessEvent


class TraceResolutionError(ValueError):
    """Raised when an event cannot be resolved to its declared sources."""


class EvidenceTrace(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    source_strokes: tuple[Stroke, ...] = ()
    source_points: tuple[Point, ...] = ()


def resolve_event_trace(
    event: StudentProcessEvent,
    strokes: Iterable[Stroke],
    points: Iterable[Point],
) -> EvidenceTrace:
    """Resolve all source IDs named by one event or raise an explicit error."""

    stroke_index = _unique_index(strokes, "stroke_id")
    point_index = _unique_index(points, "point_id")
    resolved_strokes = _resolve_ids(
        event.source_stroke_ids,
        stroke_index,
        f"event {event.event_id} source stroke",
    )
    resolved_points = _resolve_ids(
        event.source_point_ids,
        point_index,
        f"event {event.event_id} source point",
    )
    for stroke in resolved_strokes:
        if stroke.session_id != event.session_id:
            raise TraceResolutionError(
                f"stroke {stroke.stroke_id} belongs to another session"
            )
    for point in resolved_points:
        if point.session_id != event.session_id:
            raise TraceResolutionError(
                f"point {point.point_id} belongs to another session"
            )
    declared_point_ids = set(event.source_point_ids)
    for stroke in resolved_strokes:
        stroke_point_ids = set(stroke.point_refs)
        if stroke_point_ids and not declared_point_ids:
            raise TraceResolutionError(
                f"event {event.event_id} declares stroke {stroke.stroke_id} without its point refs"
            )
        missing = stroke_point_ids.difference(declared_point_ids)
        if missing:
            raise TraceResolutionError(
                f"event {event.event_id} point refs do not cover stroke {stroke.stroke_id}: {sorted(missing)}"
            )
    return EvidenceTrace(
        event_id=event.event_id,
        session_id=event.session_id,
        source_strokes=tuple(resolved_strokes),
        source_points=tuple(resolved_points),
    )


def resolve_event_traces(
    events: Iterable[StudentProcessEvent],
    strokes: Iterable[Stroke],
    points: Iterable[Point],
) -> tuple[EvidenceTrace, ...]:
    stroke_list = tuple(strokes)
    point_list = tuple(points)
    return tuple(
        resolve_event_trace(event, stroke_list, point_list) for event in events
    )


def _unique_index(items: Iterable[object], field_name: str) -> dict[str, object]:
    index: dict[str, object] = {}
    for item in items:
        item_id = getattr(item, field_name)
        if item_id in index:
            raise TraceResolutionError(f"duplicate {field_name}: {item_id}")
        index[item_id] = item
    return index


def _resolve_ids(
    ids: tuple[str, ...],
    index: dict[str, object],
    label: str,
) -> tuple[object, ...]:
    resolved: list[object] = []
    for item_id in ids:
        if item_id not in index:
            raise TraceResolutionError(f"{label} not found: {item_id}")
        resolved.append(index[item_id])
    return tuple(resolved)
