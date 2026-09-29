"""Immutable canonical point and raw provenance models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .immutable import freeze_json, stable_json_hash

PenState = Literal["DOWN", "MOVE", "UP", "UNKNOWN"]


class NormalizedPoint(BaseModel):
    """Legacy view retained as a read-only compatibility property."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    x: float | None
    y: float | None
    timestamp_ms: int | None = None
    pressure: float | None = None
    tilt_x: float | None = None
    tilt_y: float | None = None


class Point(BaseModel):
    """Canonical Point V1 with immutable raw provenance."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    point_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    participant_id: str | None = None
    task_segment_id: str | None = None
    device_id: str | None = None
    page_id: str | None = None
    sequence: int = Field(ge=0)
    timestamp_ms: int | None
    x_raw: float | None
    y_raw: float | None
    x_mm: float | None = None
    y_mm: float | None = None
    x_norm: float | None = Field(default=None, ge=0, le=1)
    y_norm: float | None = Field(default=None, ge=0, le=1)
    pressure_raw: float | int | None = None
    pressure_norm: float | None = Field(default=None, ge=0, le=1)
    pen_state_raw: Any = None
    pen_state: PenState = "UNKNOWN"
    source_file: str | None = None
    source_index: int = Field(ge=0)
    raw_order: int = Field(ge=0)
    processed_order: int = Field(ge=0)
    source_payload: Any
    source_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def prepare_payload_hash(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "source_payload" in values and "source_payload_hash" not in values:
            values["source_payload_hash"] = stable_json_hash(values["source_payload"])
        if isinstance(values.get("quality_flags"), list):
            values["quality_flags"] = tuple(values["quality_flags"])
        return values

    @model_validator(mode="after")
    def validate_norm_pair(self) -> "Point":
        if (self.x_norm is None) != (self.y_norm is None):
            raise ValueError("x_norm and y_norm must be provided together")
        return self

    @field_validator("source_payload", mode="before")
    @classmethod
    def freeze_source_payload(cls, value: Any) -> Any:
        return freeze_json(value)

    @field_validator("source_payload_hash", mode="before")
    @classmethod
    def validate_source_hash(cls, value: Any, info: Any) -> Any:
        payload = info.data.get("source_payload")
        expected = stable_json_hash(payload) if payload is not None else value
        if value is None:
            return expected
        if value != expected:
            raise ValueError("source_payload_hash does not match source_payload")
        return value

    @property
    def raw_index(self) -> int:
        return self.source_index

    @property
    def normalized(self) -> NormalizedPoint:
        return NormalizedPoint(x=self.x_raw, y=self.y_raw, timestamp_ms=self.timestamp_ms, pressure=float(self.pressure_raw) if self.pressure_raw is not None else None)
