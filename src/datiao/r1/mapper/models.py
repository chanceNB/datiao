"""Contracts for mapping strokes to question regions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MappingStatus = Literal["MAPPED", "UNKNOWN", "AMBIGUOUS"]


class StrokeMapping(BaseModel):
    """A conservative mapping result that keeps the stroke provenance."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    stroke_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    page_id: str | None = None
    question_id: str | None = None
    region_id: str | None = None
    status: MappingStatus
    confidence: float | None = Field(default=None, ge=0, le=1)
    start_timestamp_ms: int | None = None
    end_timestamp_ms: int | None = None
    source_point_ids: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()
