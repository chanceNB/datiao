"""Public offline entry point for the R1 process reconstruction chain."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .event import EventDetectionConfig, detect_student_process_events
from .mapper import QuestionMappingConfig, StrokeMapping, map_strokes_to_regions
from .models import Point, QuestionRegion, StudentProcessEvent, Stroke
from .models.immutable import freeze_json
from .parser import CanonicalPointAdapter, PointRecordAdapter, parse_raw_points
from .replay import PageReplay, build_page_replay
from .stroke import StrokeBuildConfig, build_strokes
from .trace import EvidenceTrace, TraceResolutionError, resolve_event_trace

QualityStatus = Literal["OK", "DEGRADED", "INVALID"]


class R1ProcessResult(BaseModel):
    """Stable transport contract consumed by R3 and reviewed by R4."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    task_segment_id: str | None = None
    data_version: str | None = None
    points: tuple[Point, ...] = ()
    strokes: tuple[Stroke, ...] = ()
    stroke_mappings: tuple[StrokeMapping, ...] = ()
    student_process_events: tuple[StudentProcessEvent, ...] = ()
    page_replay: PageReplay
    event_traces: tuple[EvidenceTrace, ...] = ()
    trace_errors: tuple[str, ...] = ()
    quality_status: QualityStatus
    quality_flags: tuple[str, ...] = ()
    source_provenance: Any = Field(default_factory=lambda: freeze_json({}))

    @field_validator("source_provenance", mode="before")
    @classmethod
    def freeze_source_provenance(cls, value: Any) -> Any:
        return freeze_json(value)


def run_r1_pipeline(
    canonical_records: Iterable[Mapping[str, Any]],
    question_regions: Iterable[QuestionRegion],
    session_id: str,
    task_segment_id: str | None = None,
    data_version: str | None = None,
    *,
    adapter: PointRecordAdapter | None = None,
    stroke_config: StrokeBuildConfig | None = None,
    min_coverage: float = 0.5,
    mapping_config: QuestionMappingConfig | None = None,
    event_config: EventDetectionConfig | None = None,
    process_end_signal: bool = False,
    session_end_ms: int | None = None,
    source_provenance: Mapping[str, Any] | None = None,
) -> R1ProcessResult:
    """Run canonicalization, parsing, strokes, mapping, events and trace checks.

    The default adapter accepts only the frozen canonical field names. A device
    adapter must be passed explicitly once its field semantics are confirmed.
    """

    if not session_id:
        raise ValueError("session_id must not be empty")
    if task_segment_id is not None and not task_segment_id:
        raise ValueError("task_segment_id must not be empty")
    if data_version is not None and not data_version:
        raise ValueError("data_version must not be empty")

    active_adapter = adapter or CanonicalPointAdapter()
    adapted_records = tuple(active_adapter.adapt(canonical_records))
    points = parse_raw_points(adapted_records)
    if any(point.session_id != session_id for point in points):
        raise ValueError("all records must belong to the requested session_id")
    if task_segment_id is not None:
        conflicting_segments = sorted({point.task_segment_id for point in points if point.task_segment_id is not None and point.task_segment_id != task_segment_id})
        if conflicting_segments:
            raise ValueError(f"task_segment_id context mismatch: caller={task_segment_id!r}, points={conflicting_segments!r}")

    regions = tuple(question_regions)
    strokes = build_strokes(points, stroke_config)
    mappings = map_strokes_to_regions(
        strokes,
        points,
        regions,
        min_coverage=min_coverage,
        config=mapping_config,
    )
    events = detect_student_process_events(
        mappings,
        strokes=strokes,
        points=points,
        config=event_config,
        process_end_signal=process_end_signal,
        session_end_ms=session_end_ms,
        task_segment_id=task_segment_id,
        data_version=data_version,
        source_provenance=source_provenance,
    )
    replay = (
        build_page_replay(points, strokes, session_id=session_id)
        if points
        else PageReplay(session_id=session_id)
    )

    traces: list[EvidenceTrace] = []
    trace_errors: list[str] = []
    for event in events:
        try:
            traces.append(resolve_event_trace(event, strokes, points))
        except TraceResolutionError as error:
            trace_errors.append(f"{event.event_id}: {error}")

    quality_flags = _collect_quality_flags(points, strokes, mappings, events)
    if trace_errors:
        quality_flags.append("TRACE_ERROR")
    quality_flags = list(dict.fromkeys(quality_flags))
    if not adapted_records:
        status: QualityStatus = "INVALID"
        quality_flags.append("EMPTY_INPUT")
    elif quality_flags:
        status = "DEGRADED"
    else:
        status = "OK"

    return R1ProcessResult(
        session_id=session_id,
        task_segment_id=task_segment_id,
        data_version=data_version,
        points=points,
        strokes=strokes,
        stroke_mappings=mappings,
        student_process_events=events,
        page_replay=replay,
        event_traces=tuple(traces),
        trace_errors=tuple(trace_errors),
        quality_status=status,
        quality_flags=tuple(dict.fromkeys(quality_flags)),
        source_provenance=source_provenance or {},
    )


def _collect_quality_flags(
    points: Iterable[Point],
    strokes: Iterable[Stroke],
    mappings: Iterable[StrokeMapping],
    events: Iterable[StudentProcessEvent],
) -> list[str]:
    point_items = tuple(points)
    stroke_items = tuple(strokes)
    mapping_items = tuple(mappings)
    event_items = tuple(events)
    flags: list[str] = []
    for item in (*point_items, *stroke_items, *mapping_items, *event_items):
        flags.extend(getattr(item, "quality_flags", ()))
    for mapping in mapping_items:
        if mapping.status != "MAPPED":
            flags.append(f"MAPPING_{mapping.status}")
    for event in event_items:
        if event.event_type == "UNKNOWN":
            flags.append("UNKNOWN_EVENT")
    return list(dict.fromkeys(flags))
