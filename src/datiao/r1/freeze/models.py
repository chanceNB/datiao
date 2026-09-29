"""Small immutable identity model for the R1 baseline freeze."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FreezeIdentity:
    freeze_id: str
    freeze_version: str
    source_baseline_commit: str


R1_BASELINE_FREEZE_V1 = FreezeIdentity(
    freeze_id="r1-baseline-freeze-v1",
    freeze_version="1.0.0",
    source_baseline_commit="2846c7f7346ebe36f12fe63fb5813e2d230c162a",
)
