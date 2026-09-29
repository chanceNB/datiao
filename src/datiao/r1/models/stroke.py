"""Stroke V1 model preserving raw and processed point order."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .immutable import freeze_json


class BoundingBox(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)


class Stroke(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    stroke_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    participant_id: str | None = None
    task_segment_id: str | None = None
    page_id: str | None = None
    point_refs: tuple[str, ...]
    raw_order: tuple[str, ...]
    processed_order: tuple[str, ...]
    start_time_ms: int | None = None
    end_time_ms: int | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    bbox: BoundingBox
    quality_flags: tuple[str, ...] = ()
    algorithm_version: str = "r1-stroke-rule-v1"
    provenance: Any = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def prepare_json_arrays(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        for key in ("point_refs", "raw_order", "processed_order", "quality_flags"):
            if isinstance(values.get(key), list):
                values[key] = tuple(values[key])
        if "duration_ms" not in values and values.get("start_time_ms") is not None and values.get("end_time_ms") is not None:
            values["duration_ms"] = values["end_time_ms"] - values["start_time_ms"]
        return values

    @model_validator(mode="after")
    def validate_time_range(self) -> "Stroke":
        if self.start_time_ms is not None and self.end_time_ms is not None and self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be greater than or equal to start_time_ms")
        if self.duration_ms is not None and self.start_time_ms is not None and self.end_time_ms is not None:
            if self.duration_ms != self.end_time_ms - self.start_time_ms:
                raise ValueError("duration_ms must equal end_time_ms - start_time_ms")
        return self

    @property
    def start_timestamp_ms(self) -> int | None:
        return self.start_time_ms

    @property
    def end_timestamp_ms(self) -> int | None:
        return self.end_time_ms

    @property
    def raw_index(self) -> tuple[str, ...]:
        return self.raw_order

    @field_validator("provenance", mode="before")
    @classmethod
    def _freeze_provenance(cls, value: Any) -> Any:
        return freeze_json(value)
