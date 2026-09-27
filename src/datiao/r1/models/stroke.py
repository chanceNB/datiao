"""Stroke models preserving raw and processed point order."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)


class Stroke(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    stroke_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    page_id: str | None = None
    raw_order: tuple[str, ...]
    processed_order: tuple[str, ...]
    start_timestamp_ms: int | None = None
    end_timestamp_ms: int | None = None
    bbox: BoundingBox
    quality_flags: tuple[str, ...] = ()
