"""Frozen R1 Feature Builder V1 structures."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FEATURE_BUILDER_VERSION = "r1-feature-v1"
FEATURE_SCHEMA_VERSION = "r1-feature-schema-v1"
TARGET_SPEC_VERSION = "r1-event-multilabel-v1"
TRUTH_ALIGNMENT_VERSION = "r1-truth-align-v1"
FEATURE_DATASET_ID = "r1-synthetic-features-v1"
FEATURE_DATASET_VERSION = "1.0.0"
LABEL_ORDER = (
    "WRITING",
    "QUESTION_VISIT",
    "QUESTION_LEAVE",
    "RETURN",
    "REVISION_CANDIDATE",
    "PAGE_CHANGE",
    "PROCESS_END",
    "UNKNOWN",
)
FEATURE_ORDER = (
    "duration_s",
    "path_length_norm",
    "mean_speed_norm_per_s",
    "pause_before_s",
    "pause_inside_s",
    "bbox_width_norm",
    "bbox_height_norm",
    "question_occupancy",
    "visit_index",
    "return_count",
    "previous_ink_iou",
    "quality_valid",
)


class FeatureConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    feature_dataset_id: str = FEATURE_DATASET_ID
    feature_dataset_version: str = FEATURE_DATASET_VERSION
    feature_builder_version: str = FEATURE_BUILDER_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    target_spec_version: str = TARGET_SPEC_VERSION
    truth_alignment_version: str = TRUTH_ALIGNMENT_VERSION
    inside_pause_threshold_ms: int = Field(default=500, ge=0)


class ObservationEpisode(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    episode_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    participant_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task_segment_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    stroke_id: str = Field(min_length=1)
    point_refs: tuple[str, ...]
    page_id: str | None = None
    question_id: str | None = None
    mapping_status: Literal["MAPPED", "UNKNOWN", "AMBIGUOUS"]
    mapping_method: Literal["ARC_LENGTH", "POINT_FALLBACK", "UNKNOWN"]
    start_time_ms: int | None = None
    end_time_ms: int | None = None
    episode_index: int = Field(ge=0)
    question_visit_index: int | None = Field(default=None, ge=1)
    question_return_count: int | None = Field(default=None, ge=0)
    quality_status: Literal["VALID", "DEGRADED"]
    quality_flags: tuple[str, ...] = ()
    feature_builder_version: str = FEATURE_BUILDER_VERSION

    @model_validator(mode="before")
    @classmethod
    def prepare_arrays(cls, data: Any) -> Any:
        values = dict(data)
        for key in ("point_refs", "quality_flags"):
            if isinstance(values.get(key), list): values[key] = tuple(values[key])
        return values


class FeatureRow(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    episode_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    features: dict[str, float | None]
    target: tuple[int, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def prepare_target(cls, data: Any) -> Any:
        values = dict(data)
        if isinstance(values.get("target"), list): values["target"] = tuple(values["target"])
        if isinstance(values.get("features"), dict):
            values["features"] = {key: values["features"].get(key) for key in FEATURE_ORDER}
        return values

    @model_validator(mode="after")
    def validate_feature_order(self) -> "FeatureRow":
        if tuple(self.features) != FEATURE_ORDER:
            raise ValueError("feature keys must exactly match frozen FEATURE_ORDER")
        if any(label not in {0, 1} for label in self.target):
            raise ValueError("target must be an 8-dimensional multi-hot vector")
        return self


class SequenceSample(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    sequence_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    participant_id: str = Field(min_length=1)
    task_segment_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    episode_ids: tuple[str, ...] = Field(min_length=1)
    values: tuple[tuple[float, ...], ...] = Field(min_length=1)
    feature_mask: tuple[tuple[int, ...], ...] = Field(min_length=1)
    targets: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def prepare_arrays(cls, data: Any) -> Any:
        values = dict(data)
        for key in ("episode_ids", "values", "feature_mask", "targets"):
            if isinstance(values.get(key), list):
                values[key] = tuple(tuple(item) if isinstance(item, list) else item for item in values[key])
        return values

    @model_validator(mode="after")
    def validate_shapes(self) -> "SequenceSample":
        if not (len(self.episode_ids) == len(self.values) == len(self.feature_mask) == len(self.targets)):
            raise ValueError("sequence timestep arrays must have identical length")
        if any(len(row) != len(FEATURE_ORDER) for row in self.values):
            raise ValueError("sequence values must have F feature columns")
        if any(len(row) != len(FEATURE_ORDER) for row in self.feature_mask):
            raise ValueError("sequence feature_mask must have F feature columns")
        if any(any(value not in {0, 1} for value in row) for row in self.feature_mask):
            raise ValueError("feature_mask values must be 0 or 1")
        if any(len(row) != len(LABEL_ORDER) for row in self.targets):
            raise ValueError("sequence targets must have 8 label columns")
        return self


class AlignmentRecord(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    truth_event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    matched_episode_ids: tuple[str, ...] = ()
    alignment_status: Literal["ONE_TO_ONE", "ONE_TO_MANY", "UNMATCHED", "AMBIGUOUS"]


class FeatureSplitManifest(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    split_name: Literal["train", "validation", "test"]
    source_dataset_id: str = Field(min_length=1)
    source_dataset_manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    episode_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    count: int = Field(ge=1)


class FeatureManifest(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    manifest_version: Literal["1.0.0"] = "1.0.0"
    feature_dataset_id: str = Field(min_length=1)
    feature_dataset_version: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    source_dataset_version: str = Field(min_length=1)
    source_dataset_manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_split_manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    feature_builder_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    target_spec_version: str = Field(min_length=1)
    truth_alignment_version: str = Field(min_length=1)
    episode_count: int = Field(ge=1)
    sequence_count: int = Field(ge=1)
    feature_order: tuple[str, ...]
    label_order: tuple[str, ...]
    split_counts: dict[str, dict[str, int]]
    missingness_summary: dict[str, Any]
    target_distribution: dict[str, Any]
    file_hashes: dict[str, str]
    files: tuple[str, ...]
    manifest_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def prepare_arrays(cls, data: Any) -> Any:
        values = dict(data)
        for key in ("feature_order", "label_order", "files"):
            if isinstance(values.get(key), list):
                values[key] = tuple(values[key])
        return values


class FeatureBuildResult(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    output_path: str = Field(min_length=1)
    manifest: FeatureManifest
    episode_count: int = Field(ge=1)
    sequence_count: int = Field(ge=1)
