"""Deterministic observable Student Process event state machine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import hypot
from typing import Any

from ..mapper import StrokeMapping
from ..models import Point, Stroke, StudentProcessEvent

PROCESS_SCHEMA_VERSION = "1.0.0"
EVENT_ALGORITHM_VERSION = "r1-event-rule-v0.2"


@dataclass(frozen=True, slots=True)
class EventDetectionConfig:
    revision_pause_threshold_ms: int = 3_000
    revision_overlap_threshold: float = 0.10
    process_end_timeout_ms: int | None = None
    emit_process_end: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.revision_pause_threshold_ms, bool) or not isinstance(self.revision_pause_threshold_ms, int) or self.revision_pause_threshold_ms < 0:
            raise ValueError("revision_pause_threshold_ms must be a non-negative integer")
        if isinstance(self.revision_overlap_threshold, bool) or not isinstance(self.revision_overlap_threshold, (int, float)) or not 0 <= self.revision_overlap_threshold <= 1:
            raise ValueError("revision_overlap_threshold must be in [0, 1]")
        if self.process_end_timeout_ms is not None and (isinstance(self.process_end_timeout_ms, bool) or not isinstance(self.process_end_timeout_ms, int) or self.process_end_timeout_ms < 0):
            raise ValueError("process_end_timeout_ms must be a non-negative integer or None")


def detect_student_process_events(
    mappings: Iterable[StrokeMapping],
    *,
    strokes: Iterable[Stroke] | None = None,
    points: Iterable[Point] | None = None,
    config: EventDetectionConfig | None = None,
    process_end_signal: bool = False,
    session_end_ms: int | None = None,
    schema_version: str = PROCESS_SCHEMA_VERSION,
    task_segment_id: str | None = None,
    data_version: str | None = None,
    source_provenance: Mapping[str, Any] | None = None,
) -> tuple[StudentProcessEvent, ...]:
    """Convert ordered Stroke mappings into observable process events.

    EOF alone is never a process end.  ``process_end_signal`` or a caller
    supplied ``session_end_ms`` together with a configured timeout is required.
    """

    active_config = config or EventDetectionConfig()
    ordered = tuple(sorted(tuple(mappings), key=lambda item: (item.start_time_ms is None, item.start_time_ms if item.start_time_ms is not None else 0, item.stroke_id)))
    if not ordered:
        return ()
    session_ids = {mapping.session_id for mapping in ordered}
    if len(session_ids) != 1:
        raise ValueError("all mappings must belong to one session")
    session_id = ordered[0].session_id
    stroke_index = {stroke.stroke_id: stroke for stroke in (strokes or ())}
    point_index = {point.point_id: point for point in (points or ())}
    events: list[StudentProcessEvent] = []
    visited_questions: set[str] = set()
    active_question: str | None = None
    active_mapping: StrokeMapping | None = None
    previous_page_id: str | None = None
    question_ink_history: dict[str, list[tuple[float, float, float, float]]] = {}

    def emit(
        event_type: str,
        mapping: StrokeMapping,
        *,
        occurred_at_ms: int | None = None,
        question_id: str | None = None,
        previous_question_id: str | None = None,
        next_question_id: str | None = None,
        quality_flags: tuple[str, ...] = (),
        metadata: dict[str, object] | None = None,
        confidence: float | None = None,
        source_mapping: StrokeMapping | None = None,
    ) -> None:
        source = source_mapping or mapping
        start_time_ms = occurred_at_ms if occurred_at_ms is not None else mapping.start_time_ms
        end_time_ms = occurred_at_ms if occurred_at_ms is not None else mapping.end_time_ms
        event_quality_flags = list(mapping.quality_flags) + list(quality_flags)
        if start_time_ms is None or end_time_ms is None:
            event_quality_flags.append("TIME_UNAVAILABLE")
        events.append(
            StudentProcessEvent(
                event_id=f"{session_id}:event:{len(events):04d}",
                schema_version=schema_version,
                event_type=event_type,
                session_id=session_id,
                task_segment_id=task_segment_id,
                participant_id=source.participant_id,
                page_id=mapping.page_id,
                data_version=data_version,
                sequence=len(events),
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                question_id=question_id,
                previous_question_id=previous_question_id,
                next_question_id=next_question_id,
                stroke_refs=(source.stroke_id,),
                point_refs=source.point_refs,
                quality_status="DEGRADED" if (event_quality_flags or mapping.status != "MAPPED") else "VALID",
                algorithm_version=EVENT_ALGORITHM_VERSION,
                provenance=source_provenance or {},
                mapping_confidence=confidence,
                quality_flags=tuple(dict.fromkeys(event_quality_flags)),
                metadata=metadata or {},
            )
        )

    for mapping in ordered:
        stroke = stroke_index.get(mapping.stroke_id)
        page_changed = _known_page(previous_page_id) and _known_page(mapping.page_id) and mapping.page_id != previous_page_id
        if page_changed:
            emit("PAGE_CHANGE", mapping, metadata={"from_page_id": previous_page_id, "to_page_id": mapping.page_id}, confidence=mapping.confidence)
        if _known_page(mapping.page_id):
            previous_page_id = mapping.page_id

        writing_evidence = _has_writing_evidence(mapping, stroke, point_index)
        current_bbox = _stroke_bbox(stroke, point_index)
        prior_bboxes = tuple(question_ink_history.get(mapping.question_id, ())) if mapping.question_id is not None else ()
        if writing_evidence:
            emit("WRITING", mapping, question_id=mapping.question_id if mapping.status == "MAPPED" else None, confidence=mapping.confidence, metadata={"evidence": "observable_geometry_or_pen_contact"})

        if mapping.status != "MAPPED" or mapping.question_id is None:
            if active_question is not None and active_mapping is not None:
                emit("QUESTION_LEAVE", active_mapping, occurred_at_ms=active_mapping.end_time_ms, question_id=active_question, next_question_id=None, source_mapping=active_mapping, metadata={"reason": "mapping_unavailable"})
                active_question = None
                active_mapping = None
            emit("UNKNOWN", mapping, question_id="UNKNOWN", quality_flags=mapping.quality_flags or ("UNKNOWN_MAPPING",), metadata={"mapping_status": mapping.status})
            continue

        question_id = mapping.question_id
        revisited = question_id in visited_questions and (active_question != question_id or page_changed)
        paused = _has_pause(active_mapping, mapping, active_config.revision_pause_threshold_ms) if active_question == question_id else False
        overlaps_prior = current_bbox is not None and any(_bbox_iou(current_bbox, previous) >= active_config.revision_overlap_threshold for previous in prior_bboxes)

        if active_question is None:
            emit("RETURN" if revisited else "QUESTION_VISIT", mapping, question_id=question_id, confidence=mapping.confidence)
        elif active_question == question_id and not page_changed:
            if prior_bboxes and (paused or revisited) and overlaps_prior:
                emit("REVISION_CANDIDATE", mapping, question_id=question_id, confidence=mapping.confidence, metadata={"reason": "prior_ink_pause_or_revisit_overlap"})
        else:
            emit("QUESTION_LEAVE", active_mapping or mapping, occurred_at_ms=active_mapping.end_time_ms if active_mapping else None, question_id=active_question, next_question_id=question_id, source_mapping=active_mapping or mapping)
            emit("RETURN" if revisited else "QUESTION_VISIT", mapping, question_id=question_id, previous_question_id=active_question, confidence=mapping.confidence)
            if prior_bboxes and overlaps_prior:
                emit("REVISION_CANDIDATE", mapping, question_id=question_id, confidence=mapping.confidence, metadata={"reason": "reentry_overlap"})

        visited_questions.add(question_id)
        active_question = question_id
        active_mapping = mapping
        if writing_evidence and current_bbox is not None:
            question_ink_history.setdefault(question_id, []).append(current_bbox)

    last_mapping = ordered[-1]
    should_end = process_end_signal or active_config.emit_process_end
    end_reason = "explicit_signal" if process_end_signal or active_config.emit_process_end else None
    if session_end_ms is not None and active_config.process_end_timeout_ms is not None and last_mapping.end_time_ms is not None:
        if session_end_ms - last_mapping.end_time_ms >= active_config.process_end_timeout_ms:
            should_end = True
            end_reason = "session_timeout"
    if should_end:
        emit("PROCESS_END", last_mapping, occurred_at_ms=last_mapping.end_time_ms, question_id=active_question, source_mapping=last_mapping, metadata={"process_end_reason": end_reason or "configured"})
    return tuple(events)


def _has_writing_evidence(mapping: StrokeMapping, stroke: Stroke | None, points: dict[str, Point]) -> bool:
    if stroke is None:
        return len(mapping.point_refs) >= 2
    referenced = [points[point_id] for point_id in stroke.processed_order if point_id in points]
    valid = [point for point in referenced if (point.x_norm is not None and point.y_norm is not None) or (point.x_raw is not None and point.y_raw is not None)]
    if len(valid) < 2:
        return False
    for coordinate_space in ("norm", "raw"):
        pairs = [_point_pair(point, coordinate_space) for point in valid]
        if all(pair is not None for pair in pairs):
            length = sum(hypot(second[0] - first[0], second[1] - first[1]) for first, second in zip(pairs, pairs[1:]))
            if length > 1e-12:
                return True
    return False


def _point_pair(point: Point, space: str) -> tuple[float, float] | None:
    return (point.x_norm, point.y_norm) if space == "norm" and point.x_norm is not None and point.y_norm is not None else ((point.x_raw, point.y_raw) if space == "raw" and point.x_raw is not None and point.y_raw is not None else None)


def _stroke_bbox(stroke: Stroke | None, points: dict[str, Point]) -> tuple[float, float, float, float] | None:
    if stroke is None:
        return None
    referenced = [points[point_id] for point_id in stroke.processed_order if point_id in points]
    norm = [_point_pair(point, "norm") for point in referenced]
    coordinates = [value for value in norm if value is not None]
    if not coordinates:
        coordinates = [value for value in (_point_pair(point, "raw") for point in referenced) if value is not None]
    if not coordinates:
        return None
    xs = [value[0] for value in coordinates]
    ys = [value[1] for value in coordinates]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    if union <= 1e-12:
        return 1.0 if first == second else 0.0
    return intersection / union


def _has_pause(previous: StrokeMapping | None, current: StrokeMapping, threshold_ms: int) -> bool:
    if previous is None or previous.end_time_ms is None or current.start_time_ms is None:
        return False
    return current.start_time_ms - previous.end_time_ms >= threshold_ms


def _known_page(page_id: str | None) -> bool:
    if page_id is None:
        return False
    return page_id.strip().upper() not in {"UNKNOWN", "PAGE_UNKNOWN", "PAGE-UNKNOWN"}
