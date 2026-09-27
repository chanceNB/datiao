"""Student process event model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .immutable import freeze_json
from .policy import reject_out_of_scope_fields

EventType = Literal[
    "QUESTION_VISIT",
    "QUESTION_LEAVE",
    "RETURN",
    "REVISION_CANDIDATE",
    "PAGE_CHANGE",
    "PROCESS_END",
    "UNKNOWN",
]


class StudentProcessEvent(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    event_type: EventType
    sequence: int = Field(ge=0)
    occurred_at_ms: int | None = None
    page_id: str | None = None
    question_id: str | None = None
    previous_question_id: str | None = None
    next_question_id: str | None = None
    source_stroke_ids: tuple[str, ...] = ()
    source_point_ids: tuple[str, ...] = ()
    mapping_confidence: float | None = Field(default=None, ge=0, le=1)
    quality_flags: tuple[str, ...] = ()
    metadata: Any = Field(default_factory=lambda: freeze_json({}))

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: Any) -> Any:
        reject_out_of_scope_fields(value)
        return freeze_json(value)
