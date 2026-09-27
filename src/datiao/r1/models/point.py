"""Immutable raw pen point models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .immutable import freeze_json, stable_json_hash


class NormalizedPoint(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    x: float | None
    y: float | None
    timestamp_ms: int | None = None
    pressure: float | None = None
    tilt_x: float | None = None
    tilt_y: float | None = None


class Point(BaseModel):
    """A raw point whose source payload and identity cannot be overwritten."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    point_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    page_id: str | None = None
    raw_index: int = Field(ge=0)
    normalized: NormalizedPoint
    source_payload: Any
    source_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def calculate_source_hash(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "source_payload" not in data:
            return data
        values = dict(data)
        expected_hash = stable_json_hash(values["source_payload"])
        supplied_hash = values.get("source_payload_hash")
        if supplied_hash is not None and supplied_hash != expected_hash:
            raise ValueError("source_payload_hash does not match source_payload")
        values["source_payload_hash"] = expected_hash
        return values

    @field_validator("source_payload", mode="before")
    @classmethod
    def freeze_source_payload(cls, value: Any) -> Any:
        return freeze_json(value)
