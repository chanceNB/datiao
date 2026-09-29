"""Reproducible R1 Synthetic Development Dataset tooling."""

from .audit import run_integrity_audit, run_leakage_audit
from .io import load_materialized_dataset, reload_dataset
from .manifest import compute_case_record_hash, validate_case_record_hash, compute_case_collection_hash, compute_dataset_manifest_hash
from .models import (
    DatasetBuildResult,
    DatasetConfig,
    DatasetManifest,
    DatasetSummary,
    SplitConfig,
    SplitManifest,
)
from .split import assign_group_split, build_split_manifests

__all__ = [
    "DatasetBuildResult",
    "DatasetConfig",
    "DatasetManifest",
    "DatasetSummary",
    "SplitConfig",
    "SplitManifest",
    "assign_group_split",
    "build_split_manifests",
    "build_dataset",
    "load_dataset_config",
    "materialize_dataset",
    "reload_dataset",
    "load_materialized_dataset",
    "run_leakage_audit",
    "run_integrity_audit",
    "compute_dataset_manifest_hash",
    "compute_case_record_hash",
    "validate_case_record_hash",
    "compute_case_collection_hash",
]


def __getattr__(name: str):
    if name in {"build_dataset", "load_dataset_config", "materialize_dataset"}:
        from .builder import build_dataset, load_dataset_config, materialize_dataset

        return {"build_dataset": build_dataset, "load_dataset_config": load_dataset_config, "materialize_dataset": materialize_dataset}[name]
    raise AttributeError(name)
