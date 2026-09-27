"""Page replay contracts for point and stroke provenance."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ReplayFrame(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    frame_index: int = Field(ge=0)
    point_id: str = Field(min_length=1)
    stroke_id: str | None = None
    page_id: str | None = None
    timestamp_ms: int | None = None
    x: float | None = None
    y: float | None = None
    raw_index: int = Field(ge=0)
    quality_flags: tuple[str, ...] = ()


class PageReplay(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    page_order: tuple[str, ...] = ()
    frames: tuple[ReplayFrame, ...] = ()
