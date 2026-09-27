"""Deterministic Student Process event state machine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..mapper import StrokeMapping
from ..models import StudentProcessEvent

PROCESS_SCHEMA_VERSION = "r1.process.v1"


def detect_student_process_events(
    mappings: Iterable[StrokeMapping],
    *,
    schema_version: str = PROCESS_SCHEMA_VERSION,
    task_segment_id: str | None = None,
    data_version: str | None = None,
    source_provenance: Mapping[str, Any] | None = None,
) -> tuple[StudentProcessEvent, ...]:
    """Convert ordered stroke mappings into observable process events."""

    ordered = tuple(
        sorted(
            tuple(mappings),
            key=lambda item: (
                item.start_timestamp_ms is None,
                item.start_timestamp_ms if item.start_timestamp_ms is not None else 0,
                item.stroke_id,
            ),
        )
    )
    if not ordered:
        return ()
    session_ids = {mapping.session_id for mapping in ordered}
    if len(session_ids) != 1:
        raise ValueError("all mappings must belong to one session")
    session_id = ordered[0].session_id
    events: list[StudentProcessEvent] = []
    visited_questions: set[str] = set()
    active_question: str | None = None
    active_mapping: StrokeMapping | None = None
    previous_page_id: str | None = None

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
        sequence = len(events)
        events.append(
            StudentProcessEvent(
                event_id=f"{session_id}:event:{sequence:04d}",
                schema_version=schema_version,
                session_id=session_id,
                task_segment_id=task_segment_id,
                data_version=data_version,
                source_provenance=source_provenance or {},
                event_type=event_type,
                sequence=sequence,
                occurred_at_ms=(
                    occurred_at_ms
                    if occurred_at_ms is not None
                    else mapping.start_timestamp_ms
                ),
                page_id=mapping.page_id,
                question_id=question_id,
                previous_question_id=previous_question_id,
                next_question_id=next_question_id,
                source_stroke_ids=(source.stroke_id,),
                source_point_ids=source.source_point_ids,
                mapping_confidence=confidence,
                quality_flags=quality_flags,
                metadata=metadata or {},
            )
        )

    for mapping in ordered:
        if (
            previous_page_id is not None
            and mapping.page_id != previous_page_id
        ):
            emit(
                "PAGE_CHANGE",
                mapping,
                metadata={
                    "from_page_id": previous_page_id,
                    "to_page_id": mapping.page_id,
                },
                confidence=mapping.confidence,
            )
        previous_page_id = mapping.page_id

        if mapping.status != "MAPPED" or mapping.question_id is None:
            if active_question is not None and active_mapping is not None:
                emit(
                    "QUESTION_LEAVE",
                    mapping,
                    occurred_at_ms=active_mapping.end_timestamp_ms,
                    question_id=active_question,
                    next_question_id=None,
                    source_mapping=active_mapping,
                    metadata={"reason": "mapping_unavailable"},
                )
                active_question = None
                active_mapping = None
            emit(
                "UNKNOWN",
                mapping,
                quality_flags=mapping.quality_flags or ("UNKNOWN_MAPPING",),
                metadata={"mapping_status": mapping.status},
            )
            continue

        question_id = mapping.question_id
        if active_question is None:
            event_type = "RETURN" if question_id in visited_questions else "QUESTION_VISIT"
            emit(
                event_type,
                mapping,
                question_id=question_id,
                confidence=mapping.confidence,
            )
        elif active_question == question_id:
            emit(
                "REVISION_CANDIDATE",
                mapping,
                question_id=question_id,
                confidence=mapping.confidence,
                metadata={"reason": "separate_stroke_same_question"},
            )
        else:
            emit(
                "QUESTION_LEAVE",
                mapping,
                occurred_at_ms=active_mapping.end_timestamp_ms if active_mapping else None,
                question_id=active_question,
                next_question_id=question_id,
                source_mapping=active_mapping or mapping,
            )
            event_type = "RETURN" if question_id in visited_questions else "QUESTION_VISIT"
            emit(
                event_type,
                mapping,
                question_id=question_id,
                previous_question_id=active_question,
                confidence=mapping.confidence,
            )
        visited_questions.add(question_id)
        active_question = question_id
        active_mapping = mapping

    last_mapping = ordered[-1]
    emit(
        "PROCESS_END",
        last_mapping,
        occurred_at_ms=last_mapping.end_timestamp_ms,
        question_id=active_question,
        source_mapping=last_mapping,
        metadata={"processed_stroke_count": len(ordered)},
    )
    return tuple(events)
