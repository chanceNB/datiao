"""Versioned structures for the R1 Synthetic Development Dataset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..synthetic.models import ScenarioType


class SplitConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_sum(self) -> "SplitConfig":
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1.0")
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_type: Literal["synthetic"] = "synthetic"
    generator_version: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    participant_count: int = Field(ge=3)
    scenario_types: tuple[ScenarioType, ...] = Field(min_length=1)
    dataset_seed: int
    split_seed: int
    split_ratio: SplitConfig
    default_device_id: str = Field(min_length=1)
    task_family_mapping: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def prepare_arrays(cls, data: Any) -> Any:
        values = dict(data)
        if isinstance(values.get("scenario_types"), list):
            values["scenario_types"] = tuple(values["scenario_types"])
        return values

    @model_validator(mode="after")
    def validate_scenarios(self) -> "DatasetConfig":
        configured = set(self.scenario_types)
        if len(configured) != len(self.scenario_types):
            raise ValueError("scenario_types must not contain duplicates")
        missing = sorted(configured.difference(self.task_family_mapping))
        if missing:
            raise ValueError(f"task_family_mapping missing scenarios: {missing}")
        if any(not family for family in self.task_family_mapping.values()):
            raise ValueError("task family names must not be empty")
        return self


class CaseManifestRecord(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_type: Literal["synthetic"] = "synthetic"
    participant_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task_segment_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_type: ScenarioType
    task_family: str = Field(min_length=1)
    seed: int
    generator_version: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    raw_records_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    case_manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    quality_status: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]


class SplitManifest(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    split_version: str = Field(min_length=1)
    split_name: Literal["train", "validation", "test"]
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split_seed: int
    group_field: Literal["participant_id", "group_id", "session_id"] = "group_id"
    participant_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    count: int = Field(ge=1)

    @model_validator(mode="before")
    @classmethod
    def prepare_arrays(cls, data: Any) -> Any:
        values = dict(data)
        for key in ("participant_ids", "group_ids", "case_ids"):
            if isinstance(values.get(key), list):
                values[key] = tuple(values[key])
        return values


class DatasetManifest(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    manifest_version: Literal["1.0.0"] = "1.0.0"
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_type: Literal["synthetic"] = "synthetic"
    generator_version: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    dataset_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    participant_count: int = Field(ge=1)
    session_count: int = Field(ge=1)
    case_count: int = Field(ge=1)
    point_count: int = Field(ge=0)
    stroke_count: int = Field(ge=0)
    mapping_count: int = Field(ge=0)
    truth_event_count: int = Field(ge=0)
    predicted_event_count: int = Field(ge=0)
    scenario_counts: dict[str, int]
    task_family_counts: dict[str, int]
    quality_counts: dict[str, int]
    split_summary: dict[str, Any]
    files: tuple[str, ...]
    file_hashes: dict[str, str]
    case_manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split_manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def prepare_arrays(cls, data: Any) -> Any:
        values = dict(data)
        if isinstance(values.get("files"), list):
            values["files"] = tuple(values["files"])
        return values


class DatasetSummary(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_type: Literal["synthetic"] = "synthetic"
    participants: int = Field(ge=1)
    cases: int = Field(ge=1)
    points: int = Field(ge=0)
    strokes: int = Field(ge=0)
    mappings: int = Field(ge=0)
    truth_events: int = Field(ge=0)
    predicted_events: int = Field(ge=0)
    scenario_counts: dict[str, int]
    task_family_counts: dict[str, int]
    truth_event_distribution: dict[str, int]
    prediction_event_distribution: dict[str, int]
    quality_summary: dict[str, Any]
    split_counts: dict[str, dict[str, Any]]
    audit_status: dict[str, str]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class DatasetBuildResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True, extra="forbid")

    output_path: str = Field(min_length=1)
    manifest: DatasetManifest
    summary: DatasetSummary
    split_manifests: tuple[SplitManifest, ...]


def as_json_mapping(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    """Return a plain JSON-compatible mapping for audit and hashing helpers."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)
