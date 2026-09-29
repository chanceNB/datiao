"""Contracts for mapping strokes to question regions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MappingStatus = Literal["MAPPED", "UNKNOWN", "AMBIGUOUS"]


class StrokeMapping(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    stroke_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    participant_id: str | None = None
    task_segment_id: str | None = None
    page_id: str | None = None
    question_id: str | None = None
    region_id: str | None = None
    status: MappingStatus
    confidence: float | None = Field(default=None, ge=0, le=1)
    start_time_ms: int | None = None
    end_time_ms: int | None = None
    point_refs: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "start_time_ms" not in values and "start_timestamp_ms" in values:
            values["start_time_ms"] = values.pop("start_timestamp_ms")
        if "end_time_ms" not in values and "end_timestamp_ms" in values:
            values["end_time_ms"] = values.pop("end_timestamp_ms")
        if "point_refs" not in values and "source_point_ids" in values:
            values["point_refs"] = values.pop("source_point_ids")
        values.pop("start_timestamp_ms", None)
        values.pop("end_timestamp_ms", None)
        values.pop("source_point_ids", None)
        return values

    @property
    def start_timestamp_ms(self) -> int | None:
        return self.start_time_ms

    @property
    def end_timestamp_ms(self) -> int | None:
        return self.end_time_ms

    @property
    def source_point_ids(self) -> tuple[str, ...]:
        return self.point_refs
