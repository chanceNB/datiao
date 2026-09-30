"""Strict R1 to R3 transport projection."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..models import StudentProcessEvent

EVENT_TYPES = (
    "WRITING",
    "QUESTION_VISIT",
    "QUESTION_LEAVE",
    "RETURN",
    "REVISION_CANDIDATE",
    "PAGE_CHANGE",
    "PROCESS_END",
    "UNKNOWN",
)
_SYNTHETIC_REQUIRED_PROVENANCE = frozenset(
    {
        "dataset_type",
        "generator_version",
        "seed",
        "scenario_id",
        "ground_truth_source",
        "manifest_hash",
    }
)
_FORBIDDEN_PROVENANCE_KEYS = frozenset({"emotion", "ability", "attention", "psychological_labels"})


class R1R3EventV01(BaseModel):
    """R1 StudentProcessEvent projected onto the R3 wire contract."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    event_type: Literal[
        "WRITING",
        "QUESTION_VISIT",
        "QUESTION_LEAVE",
        "RETURN",
        "REVISION_CANDIDATE",
        "PAGE_CHANGE",
        "PROCESS_END",
        "UNKNOWN",
    ]
    session_id: str = Field(min_length=1)
    task_segment_id: str | None = None
    participant_id: str | None = None
    question_id: str = Field(min_length=1)
    start_time_ms: int | None
    end_time_ms: int | None
    point_refs: tuple[str, ...] = ()
    stroke_refs: tuple[str, ...] = ()
    quality_status: Literal["VALID", "DEGRADED", "INVALID"]
    algorithm_version: str = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)


class R1R3EventBatchV01(BaseModel):
    """Proposal envelope for a consistent collection of R3 event projections."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    contract_version: Literal["r1-event-batch-v0.1"] = "r1-event-batch-v0.1"
    batch_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task_segment_id: str | None = None
    dataset_version: str = Field(min_length=1)
    algorithm_version: str = Field(min_length=1)
    events: tuple[R1R3EventV01, ...] = Field(min_length=1)


def export_event_to_r3(
    event: StudentProcessEvent,
    *,
    point_ids: Iterable[str],
    stroke_ids: Iterable[str],
) -> R1R3EventV01:
    """Validate traceability and project one immutable R1 event."""

    point_index = set(point_ids)
    stroke_index = set(stroke_ids)
    missing_points = sorted(set(event.point_refs).difference(point_index))
    missing_strokes = sorted(set(event.stroke_refs).difference(stroke_index))
    if missing_points:
        raise ValueError(f"point refs cannot be resolved: {missing_points}")
    if missing_strokes:
        raise ValueError(f"stroke refs cannot be resolved: {missing_strokes}")

    provenance = _project_provenance(event.provenance)
    _validate_synthetic_identity(event, provenance)
    return R1R3EventV01(
        event_id=event.event_id,
        event_type=event.event_type,
        session_id=event.session_id,
        task_segment_id=event.task_segment_id,
        participant_id=event.participant_id,
        question_id=event.question_id if event.question_id is not None else "UNKNOWN",
        start_time_ms=event.start_time_ms,
        end_time_ms=event.end_time_ms,
        point_refs=event.point_refs,
        stroke_refs=event.stroke_refs,
        quality_status=event.quality_status,
        algorithm_version=event.algorithm_version,
        provenance=provenance,
    )


def export_batch_to_r3(
    events: Iterable[StudentProcessEvent],
    *,
    batch_id: str,
    dataset_version: str,
    point_ids: Iterable[str],
    stroke_ids: Iterable[str],
) -> R1R3EventBatchV01:
    """Project events and enforce batch-wide identity/version consistency."""

    point_ids = tuple(point_ids)
    stroke_ids = tuple(stroke_ids)
    projected = tuple(
        export_event_to_r3(event, point_ids=point_ids, stroke_ids=stroke_ids)
        for event in events
    )
    if not projected:
        raise ValueError("batch must contain at least one event")
    first = projected[0]
    for event in projected[1:]:
        if event.session_id != first.session_id:
            raise ValueError("batch session_id must be consistent")
        if event.task_segment_id != first.task_segment_id:
            raise ValueError("batch task_segment_id must be consistent")
        if event.algorithm_version != first.algorithm_version:
            raise ValueError("batch algorithm_version must be consistent")
    return R1R3EventBatchV01(
        batch_id=batch_id,
        session_id=first.session_id,
        task_segment_id=first.task_segment_id,
        dataset_version=dataset_version,
        algorithm_version=first.algorithm_version,
        events=projected,
    )


def _project_provenance(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("event provenance must be an object")
    provenance = dict(value)
    forbidden = sorted(_FORBIDDEN_PROVENANCE_KEYS.intersection(provenance))
    if forbidden:
        raise ValueError(f"provenance contains forbidden fields: {forbidden}")
    if provenance.get("dataset_type") == "synthetic":
        missing = sorted(_SYNTHETIC_REQUIRED_PROVENANCE.difference(provenance))
        if missing:
            raise ValueError(f"synthetic provenance missing fields: {missing}")
        manifest_hash = provenance["manifest_hash"]
        if not isinstance(manifest_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_hash):
            raise ValueError("synthetic provenance manifest_hash must be a real sha256 value")
    return provenance


def _validate_synthetic_identity(
    event: StudentProcessEvent,
    provenance: Mapping[str, Any],
) -> None:
    if provenance.get("dataset_type") != "synthetic":
        return
    if not event.session_id.startswith("sim_"):
        raise ValueError("synthetic session_id must start with sim_")
    if event.participant_id is None or not event.participant_id.startswith("sim_"):
        raise ValueError("synthetic participant_id must start with sim_")
    if event.task_segment_id is not None and not event.task_segment_id.startswith("sim_"):
        raise ValueError("synthetic task_segment_id must start with sim_")

