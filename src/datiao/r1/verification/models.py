"""Manual verification records for R1 event review."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VerificationLabel = Literal["YES", "NO", "UNCERTAIN"]


class ManualVerificationRecord(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    predicted_event_type: str = Field(min_length=1)
    expected_event_type: str | None = None
    label: VerificationLabel = "UNCERTAIN"
    reviewer: str | None = None
    notes: str | None = None
    source_stroke_ids: tuple[str, ...] = ()
    source_point_ids: tuple[str, ...] = ()


class ManualVerificationSummary(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    total: int = Field(ge=0)
    yes: int = Field(ge=0)
    no: int = Field(ge=0)
    uncertain: int = Field(ge=0)
