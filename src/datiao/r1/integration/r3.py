"""Strict R1 to R3 transport projection."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..models import StudentProcessEvent

if TYPE_CHECKING:
    from ..pipeline import R1ProcessResult

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

    @field_validator("point_refs", "stroke_refs")
    @classmethod
    def validate_reference_strings(cls, refs: tuple[str, ...]) -> tuple[str, ...]:
        if any(not reference for reference in refs):
            raise ValueError("reference IDs must not be empty")
        return refs

    @model_validator(mode="after")
    def validate_wire_semantics(self) -> "R1R3EventV01":
        if self.start_time_ms is not None and self.end_time_ms is not None and self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be greater than or equal to start_time_ms")
        if self.quality_status == "VALID" and (self.start_time_ms is None or self.end_time_ms is None):
            raise ValueError("VALID events require start_time_ms and end_time_ms")
        if self.algorithm_version.strip().lower() in {"latest", "current"}:
            raise ValueError("algorithm_version must be a concrete version")
        _validate_provenance_and_identity(
            session_id=self.session_id,
            participant_id=self.participant_id,
            task_segment_id=self.task_segment_id,
            provenance=self.provenance,
        )
        return self


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

    @model_validator(mode="after")
    def validate_batch_consistency(self) -> "R1R3EventBatchV01":
        if self.dataset_version.strip().lower() in {"latest", "current"}:
            raise ValueError("dataset_version must be a concrete version")
        if self.algorithm_version.strip().lower() in {"latest", "current"}:
            raise ValueError("algorithm_version must be a concrete version")
        for event in self.events:
            if event.session_id != self.session_id:
                raise ValueError("batch session_id must be consistent")
            if event.task_segment_id != self.task_segment_id:
                raise ValueError("batch task_segment_id must be consistent")
            if event.algorithm_version != self.algorithm_version:
                raise ValueError("batch algorithm_version must be consistent")
        return self


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
    _validate_provenance_and_identity(
        session_id=event.session_id,
        participant_id=event.participant_id,
        task_segment_id=event.task_segment_id,
        provenance=provenance,
    )
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
    return R1R3EventBatchV01(
        batch_id=batch_id,
        session_id=first.session_id,
        task_segment_id=first.task_segment_id,
        dataset_version=dataset_version,
        algorithm_version=first.algorithm_version,
        events=projected,
    )


def export_process_result_to_r3(
    result: "R1ProcessResult",
    *,
    batch_id: str,
    dataset_version: str,
) -> R1R3EventBatchV01:
    """Export the public R1 pipeline result without replacing its context."""

    if result.data_version is not None and result.data_version != dataset_version:
        raise ValueError("dataset_version must match result.data_version")
    for event in result.student_process_events:
        if event.session_id != result.session_id:
            raise ValueError("event session_id must match R1ProcessResult")
        if event.task_segment_id != result.task_segment_id:
            raise ValueError("event task_segment_id must match R1ProcessResult")
    return export_batch_to_r3(
        result.student_process_events,
        batch_id=batch_id,
        dataset_version=dataset_version,
        point_ids=(point.point_id for point in result.points),
        stroke_ids=(stroke.stroke_id for stroke in result.strokes),
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
    _validate_provenance_and_identity(provenance=provenance)
    return provenance


def _validate_provenance_and_identity(
    *,
    session_id: str | None = None,
    participant_id: str | None = None,
    task_segment_id: str | None = None,
    provenance: Mapping[str, Any],
) -> None:
    if provenance.get("dataset_type") != "synthetic":
        return
    missing = sorted(_SYNTHETIC_REQUIRED_PROVENANCE.difference(provenance))
    if missing:
        raise ValueError(f"synthetic provenance missing fields: {missing}")
    if isinstance(provenance.get("seed"), bool) or not isinstance(provenance.get("seed"), int):
        raise ValueError("synthetic provenance seed must be an integer")
    manifest_hash = provenance["manifest_hash"]
    if not isinstance(manifest_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_hash):
        raise ValueError("synthetic provenance manifest_hash must be a real sha256 value")
    if session_id is None:
        return
    if not session_id.startswith("sim_"):
        raise ValueError("synthetic session_id must start with sim_")
    if participant_id is None or not participant_id.startswith("sim_"):
        raise ValueError("synthetic participant_id must start with sim_")
    if task_segment_id is not None and not task_segment_id.startswith("sim_"):
        raise ValueError("synthetic task_segment_id must start with sim_")

