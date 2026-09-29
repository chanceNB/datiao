"""Deterministic observable Student Process event state machine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import hypot
from typing import Any

from ..mapper import StrokeMapping
from ..models import Point, Stroke, StudentProcessEvent

PROCESS_SCHEMA_VERSION = "1.0.0"
EVENT_ALGORITHM_VERSION = "r1-event-rule-v0.2.2"
UNKNOWN_PARTICIPANT_BUCKET = "UNKNOWN_PARTICIPANT"
UNKNOWN_TASK_SEGMENT_BUCKET = "UNKNOWN_TASK_SEGMENT"


@dataclass(frozen=True, slots=True)
class EventDetectionConfig:
    revision_pause_threshold_ms: int = 3_000
    revision_overlap_threshold: float = 0.10
    process_end_timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.revision_pause_threshold_ms, bool) or not isinstance(self.revision_pause_threshold_ms, int) or self.revision_pause_threshold_ms < 0:
            raise ValueError("revision_pause_threshold_ms must be a non-negative integer")
        if isinstance(self.revision_overlap_threshold, bool) or not isinstance(self.revision_overlap_threshold, (int, float)) or not 0 <= self.revision_overlap_threshold <= 1:
            raise ValueError("revision_overlap_threshold must be in [0, 1]")
        if self.process_end_timeout_ms is not None and (isinstance(self.process_end_timeout_ms, bool) or not isinstance(self.process_end_timeout_ms, int) or self.process_end_timeout_ms < 0):
            raise ValueError("process_end_timeout_ms must be a non-negative integer or None")


@dataclass
class ParticipantProcessState:
    """State for one observed participant (or the explicit unknown bucket)."""

    visited_questions: set[str] = field(default_factory=set)
    active_question: str | None = None
    active_mapping: StrokeMapping | None = None
    previous_page_id: str | None = None
    question_ink_history: dict[tuple[str, str, str, str, str, str], list["_StrokeBBox"]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _StrokeBBox:
    left: float
    top: float
    right: float
    bottom: float
    coordinate_space: str
    page_id: str | None
    participant_id: str | None
    coordinate_domain: str


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    sort_key: tuple[object, ...]
    event: StudentProcessEvent


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
    """Convert ordered Stroke mappings into participant-isolated events.

    EOF alone is never a process end.  A process end comes only from the
    explicit signal or from a caller-supplied session end plus timeout.
    """

    active_config = config or EventDetectionConfig()
    ordered = tuple(sorted(tuple(mappings), key=lambda item: (item.start_time_ms is None, item.start_time_ms if item.start_time_ms is not None else 0, _participant_bucket(item.participant_id), _task_segment_bucket(item.task_segment_id), item.stroke_id)))
    if not ordered:
        return ()
    session_ids = {mapping.session_id for mapping in ordered}
    if len(session_ids) != 1:
        raise ValueError("all mappings must belong to one session")
    if task_segment_id is not None:
        conflicts = sorted({mapping.task_segment_id for mapping in ordered if mapping.task_segment_id is not None and mapping.task_segment_id != task_segment_id})
        if conflicts:
            raise ValueError(f"task_segment_id context mismatch: caller={task_segment_id!r}, mappings={conflicts!r}")
    session_id = ordered[0].session_id
    stroke_index = {stroke.stroke_id: stroke for stroke in (strokes or ())}
    point_index = {point.point_id: point for point in (points or ())}
    grouped: dict[tuple[str, str], list[StrokeMapping]] = {}
    for mapping in ordered:
        key = (_participant_bucket(mapping.participant_id), _task_segment_bucket(mapping.task_segment_id))
        grouped.setdefault(key, []).append(mapping)

    pending: list[_PendingEvent] = []
    for state_key in sorted(grouped):
        pending.extend(
            _detect_participant_events(
                grouped[state_key],
                state_key=state_key,
                session_id=session_id,
                stroke_index=stroke_index,
                point_index=point_index,
                config=active_config,
                process_end_signal=process_end_signal,
                session_end_ms=session_end_ms,
                schema_version=schema_version,
                caller_task_segment_id=task_segment_id,
                data_version=data_version,
                source_provenance=source_provenance,
            )
        )

    pending.sort(key=lambda item: item.sort_key)
    result: list[StudentProcessEvent] = []
    for sequence, item in enumerate(pending):
        result.append(item.event.model_copy(update={"event_id": f"{session_id}:event:{sequence:04d}", "sequence": sequence}))
    return tuple(result)


def _detect_participant_events(
    mappings: list[StrokeMapping],
    *,
    state_key: tuple[str, str],
    session_id: str,
    stroke_index: dict[str, Stroke],
    point_index: dict[str, Point],
    config: EventDetectionConfig,
    process_end_signal: bool,
    session_end_ms: int | None,
    schema_version: str,
    caller_task_segment_id: str | None,
    data_version: str | None,
    source_provenance: Mapping[str, Any] | None,
) -> list[_PendingEvent]:
    state = ParticipantProcessState()
    pending: list[_PendingEvent] = []

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
        order_mapping: StrokeMapping | None = None,
    ) -> None:
        source = source_mapping or mapping
        start_time_ms = occurred_at_ms if occurred_at_ms is not None else mapping.start_time_ms
        end_time_ms = occurred_at_ms if occurred_at_ms is not None else mapping.end_time_ms
        event_quality_flags = list(mapping.quality_flags) + list(quality_flags)
        if start_time_ms is None or end_time_ms is None:
            event_quality_flags.append("TIME_UNAVAILABLE")
        local_sequence = len(pending)
        event = StudentProcessEvent(
            event_id=f"{session_id}:{state_key[0]}:{state_key[1]}:event:{local_sequence:04d}",
            schema_version=schema_version,
            event_type=event_type,
            session_id=session_id,
            task_segment_id=source.task_segment_id or mapping.task_segment_id or caller_task_segment_id,
            participant_id=source.participant_id,
            page_id=mapping.page_id,
            data_version=data_version,
            sequence=local_sequence,
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
        anchor = order_mapping or mapping
        semantic_order = {"PAGE_CHANGE": 0, "WRITING": 1, "QUESTION_LEAVE": 2, "QUESTION_VISIT": 3, "RETURN": 3, "REVISION_CANDIDATE": 4, "UNKNOWN": 5, "PROCESS_END": 6}.get(event_type, 9)
        pending.append(_PendingEvent((anchor.start_time_ms is None, anchor.start_time_ms if anchor.start_time_ms is not None else 0, state_key[0], state_key[1], anchor.stroke_id, semantic_order, local_sequence), event))

    for mapping in mappings:
        stroke = stroke_index.get(mapping.stroke_id)
        page_changed = _known_page(state.previous_page_id) and _known_page(mapping.page_id) and mapping.page_id != state.previous_page_id
        if page_changed:
            emit("PAGE_CHANGE", mapping, metadata={"from_page_id": state.previous_page_id, "to_page_id": mapping.page_id}, confidence=mapping.confidence)
        if _known_page(mapping.page_id):
            state.previous_page_id = mapping.page_id

        writing_evidence = _has_writing_evidence(stroke, point_index)
        current_bbox = _stroke_bbox(stroke, point_index)
        if writing_evidence:
            emit("WRITING", mapping, question_id=mapping.question_id if mapping.status == "MAPPED" else None, confidence=mapping.confidence, metadata={"evidence": "observable_geometry_or_pen_contact"})

        if mapping.status != "MAPPED" or mapping.question_id is None:
            if state.active_question is not None and state.active_mapping is not None:
                emit("QUESTION_LEAVE", state.active_mapping, occurred_at_ms=state.active_mapping.end_time_ms, question_id=state.active_question, next_question_id=None, source_mapping=state.active_mapping, metadata={"reason": "mapping_unavailable"}, order_mapping=mapping)
                state.active_question = None
                state.active_mapping = None
            emit("UNKNOWN", mapping, question_id="UNKNOWN", quality_flags=mapping.quality_flags or ("UNKNOWN_MAPPING",), metadata={"mapping_status": mapping.status})
            continue

        question_id = mapping.question_id
        revisited = question_id in state.visited_questions and (state.active_question != question_id or page_changed)
        paused = _has_pause(state.active_mapping, mapping, config.revision_pause_threshold_ms) if state.active_question == question_id and not page_changed else False
        history_key = _history_key(state_key, question_id, mapping.page_id, current_bbox)
        prior_bboxes = tuple(state.question_ink_history.get(history_key, ())) if history_key is not None else ()
        overlaps_prior = current_bbox is not None and any(_bbox_iou(current_bbox, previous) is not None and _bbox_iou(current_bbox, previous) >= config.revision_overlap_threshold for previous in prior_bboxes)

        if state.active_question is None:
            emit("RETURN" if revisited else "QUESTION_VISIT", mapping, question_id=question_id, confidence=mapping.confidence)
        elif state.active_question == question_id and not page_changed:
            if prior_bboxes and (paused or revisited) and overlaps_prior:
                emit("REVISION_CANDIDATE", mapping, question_id=question_id, confidence=mapping.confidence, metadata={"reason": "prior_ink_pause_or_revisit_overlap"})
        else:
            emit("QUESTION_LEAVE", state.active_mapping or mapping, occurred_at_ms=state.active_mapping.end_time_ms if state.active_mapping else None, question_id=state.active_question, next_question_id=question_id, source_mapping=state.active_mapping or mapping, order_mapping=mapping)
            emit("RETURN" if revisited else "QUESTION_VISIT", mapping, question_id=question_id, previous_question_id=state.active_question, confidence=mapping.confidence)
            if prior_bboxes and overlaps_prior:
                emit("REVISION_CANDIDATE", mapping, question_id=question_id, confidence=mapping.confidence, metadata={"reason": "reentry_overlap"})

        state.visited_questions.add(question_id)
        state.active_question = question_id
        state.active_mapping = mapping
        if writing_evidence and history_key is not None and current_bbox is not None:
            state.question_ink_history.setdefault(history_key, []).append(current_bbox)

    last_mapping = mappings[-1]
    should_end = process_end_signal
    end_reason = "explicit_signal" if process_end_signal else None
    if session_end_ms is not None and config.process_end_timeout_ms is not None and last_mapping.end_time_ms is not None:
        if session_end_ms - last_mapping.end_time_ms >= config.process_end_timeout_ms:
            should_end = True
            end_reason = "session_timeout"
    if should_end:
        emit("PROCESS_END", last_mapping, occurred_at_ms=last_mapping.end_time_ms, question_id=state.active_question, source_mapping=last_mapping, metadata={"process_end_reason": end_reason or "configured"})
    return pending


def _participant_bucket(participant_id: str | None) -> str:
    return participant_id if participant_id is not None else UNKNOWN_PARTICIPANT_BUCKET


def _task_segment_bucket(task_segment_id: str | None) -> str:
    return task_segment_id if task_segment_id is not None else UNKNOWN_TASK_SEGMENT_BUCKET


def _has_writing_evidence(stroke: Stroke | None, points: dict[str, Point]) -> bool:
    if stroke is None:
        return False
    referenced = [points[point_id] for point_id in stroke.processed_order if point_id in points]
    if len(referenced) < 2:
        return False
    for coordinate_space in ("norm", "raw"):
        pairs = [_point_pair(point, coordinate_space) for point in referenced]
        if all(pair is not None for pair in pairs):
            length = sum(hypot(second[0] - first[0], second[1] - first[1]) for first, second in zip(pairs, pairs[1:]))
            if length > 1e-12:
                return True
    return False


def _point_pair(point: Point, space: str) -> tuple[float, float] | None:
    if space == "norm" and point.x_norm is not None and point.y_norm is not None:
        return point.x_norm, point.y_norm
    if space == "raw" and point.x_raw is not None and point.y_raw is not None:
        return point.x_raw, point.y_raw
    return None


def _stroke_bbox(stroke: Stroke | None, points: dict[str, Point]) -> _StrokeBBox | None:
    if stroke is None:
        return None
    referenced = [points[point_id] for point_id in stroke.processed_order if point_id in points]
    if not referenced:
        return None
    norm = [_point_pair(point, "norm") for point in referenced]
    raw = [_point_pair(point, "raw") for point in referenced]
    if all(value is not None for value in norm):
        coordinates = [value for value in norm if value is not None]
        space = "norm"
    elif all(value is not None for value in raw):
        coordinates = [value for value in raw if value is not None]
        space = "raw"
    else:
        return None
    xs = [value[0] for value in coordinates]
    ys = [value[1] for value in coordinates]
    raw_domain = stroke.provenance.get("raw_coordinate_domain", "unknown") if isinstance(stroke.provenance, dict) else "unknown"
    coordinate_domain = "norm" if space == "norm" else str(raw_domain)
    return _StrokeBBox(min(xs), min(ys), max(xs), max(ys), space, stroke.page_id, stroke.participant_id, coordinate_domain)


def _history_key(state_key: tuple[str, str], question_id: str, page_id: str | None, bbox: _StrokeBBox | None) -> tuple[str, str, str, str, str, str] | None:
    if bbox is None or page_id is None or bbox.coordinate_space == "unavailable":
        return None
    if bbox.coordinate_space == "raw" and bbox.coordinate_domain == "unknown":
        return None
    return state_key[0], state_key[1], question_id, page_id, bbox.coordinate_space, bbox.coordinate_domain


def _bbox_iou(first: _StrokeBBox, second: _StrokeBBox) -> float | None:
    if first.coordinate_space != second.coordinate_space or first.coordinate_domain != second.coordinate_domain or first.page_id != second.page_id or first.participant_id != second.participant_id:
        return None
    left, top = max(first.left, second.left), max(first.top, second.top)
    right, bottom = min(first.right, second.right), min(first.bottom, second.bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first.right - first.left) * max(0.0, first.bottom - first.top)
    area_second = max(0.0, second.right - second.left) * max(0.0, second.bottom - second.top)
    union = area_first + area_second - intersection
    if union <= 1e-12:
        return 1.0 if (first.left, first.top, first.right, first.bottom) == (second.left, second.top, second.right, second.bottom) else 0.0
    return intersection / union


def _has_pause(previous: StrokeMapping | None, current: StrokeMapping, threshold_ms: int) -> bool:
    if previous is None or previous.end_time_ms is None or current.start_time_ms is None:
        return False
    return current.start_time_ms - previous.end_time_ms >= threshold_ms


def _known_page(page_id: str | None) -> bool:
    if page_id is None:
        return False
    return page_id.strip().upper() not in {"UNKNOWN", "PAGE_UNKNOWN", "PAGE-UNKNOWN"}
