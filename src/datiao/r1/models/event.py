"""Student process event V1 contract."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .immutable import freeze_json
from .policy import reject_out_of_scope_fields

EventType = Literal[
    "WRITING",
    "QUESTION_VISIT",
    "QUESTION_LEAVE",
    "RETURN",
    "REVISION_CANDIDATE",
    "PAGE_CHANGE",
    "PROCESS_END",
    "UNKNOWN",
]
QualityStatus = Literal["VALID", "DEGRADED", "INVALID"]


class StudentProcessEvent(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    event_id: str = Field(min_length=1)
    event_type: EventType
    session_id: str = Field(min_length=1)
    task_segment_id: str | None = None
    participant_id: str | None = None
    question_id: str | None = None
    page_id: str | None = None
    previous_question_id: str | None = None
    next_question_id: str | None = None
    sequence: int = Field(default=0, ge=0)
    start_time_ms: int | None
    end_time_ms: int | None
    point_refs: tuple[str, ...] = ()
    stroke_refs: tuple[str, ...] = ()
    quality_status: QualityStatus = "VALID"
    quality_flags: tuple[str, ...] = ()
    algorithm_version: str = "r1-event-rule-v0.1"
    provenance: Any = Field(default_factory=lambda: freeze_json({}))
    data_version: str | None = None
    mapping_confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: Any = Field(default_factory=lambda: freeze_json({}))

    @model_validator(mode="before")
    @classmethod
    def prepare_json_arrays(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        for key in ("point_refs", "stroke_refs", "quality_flags"):
            if isinstance(values.get(key), list):
                values[key] = tuple(values[key])
        return values

    @model_validator(mode="after")
    def validate_contract(self) -> "StudentProcessEvent":
        if self.start_time_ms is not None and self.end_time_ms is not None and self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be greater than or equal to start_time_ms")
        if self.event_type in {"WRITING", "QUESTION_VISIT", "QUESTION_LEAVE", "RETURN", "REVISION_CANDIDATE"} and not (self.point_refs or self.stroke_refs):
            raise ValueError("this event type requires point_refs or stroke_refs")
        if isinstance(self.provenance, dict) and self.provenance.get("dataset_type") == "synthetic":
            required = {"dataset_type", "generator_version", "seed", "scenario_id", "ground_truth_source", "manifest_hash"}
            missing = sorted(required.difference(self.provenance))
            if missing:
                raise ValueError(f"synthetic provenance missing fields: {', '.join(missing)}")
            if isinstance(self.provenance.get("seed"), bool) or not isinstance(self.provenance.get("seed"), int):
                raise ValueError("synthetic provenance seed must be an integer")
            manifest_hash = self.provenance.get("manifest_hash")
            if not isinstance(manifest_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_hash):
                raise ValueError("synthetic provenance manifest_hash must be a real sha256 value")
        return self

    @field_validator("provenance", "metadata", mode="before")
    @classmethod
    def freeze_json_fields(cls, value: Any, info: Any) -> Any:
        if info.field_name == "metadata":
            reject_out_of_scope_fields(value)
        return freeze_json(value)

    @property
    def occurred_at_ms(self) -> int | None:
        return self.start_time_ms

    @property
    def source_stroke_ids(self) -> tuple[str, ...]:
        return self.__dict__.get("source_stroke_ids", self.stroke_refs)

    @property
    def source_point_ids(self) -> tuple[str, ...]:
        return self.__dict__.get("source_point_ids", self.point_refs)

    @property
    def source_provenance(self) -> Any:
        return self.provenance
