"""Minimal, deterministic Synthetic Manifest V1 contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..models.immutable import stable_json_hash


class SyntheticManifest(BaseModel):
    """Frozen metadata describing exactly one generated raw-record set.

    ``manifest_hash`` is intentionally not a field.  It is computed from this
    payload so the hash can never contain a self-reference.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    manifest_version: Literal["1.0.0"] = "1.0.0"
    dataset_type: Literal["synthetic"] = "synthetic"
    dataset_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    seed: int
    session_id: str = Field(min_length=1)
    participant_id: str = Field(min_length=1)
    task_segment_id: str = Field(min_length=1)
    device_id: str | None = Field(default=None, min_length=1)
    record_count: int = Field(ge=0)
    raw_records_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @property
    def source_hash(self) -> str:
        """Compatibility name for the raw-records fingerprint."""

        return self.raw_records_hash


def compute_manifest_hash(manifest: SyntheticManifest | Mapping[str, Any]) -> str:
    """Return the SHA-256 of the canonical Manifest payload."""

    payload = (
        manifest.model_dump(mode="json")
        if isinstance(manifest, SyntheticManifest)
        else dict(manifest)
    )
    payload.pop("manifest_hash", None)
    return f"sha256:{stable_json_hash(payload)}"


def build_synthetic_manifest(
    *,
    dataset_id: str,
    scenario_id: str,
    generator_version: str,
    seed: int,
    session_id: str,
    participant_id: str,
    task_segment_id: str,
    raw_records: Sequence[Mapping[str, Any]],
    device_id: str | None = None,
) -> SyntheticManifest:
    """Build a Manifest from a frozen raw-record collection."""

    return SyntheticManifest(
        dataset_id=dataset_id,
        dataset_name=dataset_id,
        generator_version=generator_version,
        scenario_id=scenario_id,
        seed=seed,
        session_id=session_id,
        participant_id=participant_id,
        task_segment_id=task_segment_id,
        device_id=device_id,
        record_count=len(raw_records),
        raw_records_hash=f"sha256:{stable_json_hash(raw_records)}",
    )
