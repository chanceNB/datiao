"""Hashing helpers for deterministic dataset manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..models.immutable import stable_json_hash
from .models import CaseManifestRecord, DatasetManifest, as_json_mapping


def compute_dataset_manifest_hash(manifest: DatasetManifest | Mapping[str, Any]) -> str:
    """Hash manifest content without introducing a self-referential hash."""

    payload = manifest.model_dump(mode="json") if isinstance(manifest, DatasetManifest) else dict(manifest)
    payload.pop("manifest_hash", None)
    return f"sha256:{stable_json_hash(payload)}"


def compute_case_manifest_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Compute the aggregate hash of ordered immutable case payloads."""

    return compute_case_collection_hash(records)


def compute_case_record_hash(record: CaseManifestRecord | Mapping[str, Any]) -> str:
    """Hash one case's immutable content, excluding split and this hash."""

    payload = as_json_mapping(record)
    payload.pop("split", None)
    payload.pop("case_manifest_hash", None)
    return f"sha256:{stable_json_hash(payload)}"


def validate_case_record_hash(record: CaseManifestRecord | Mapping[str, Any]) -> bool:
    expected = as_json_mapping(record).get("case_manifest_hash")
    return isinstance(expected, str) and expected == compute_case_record_hash(record)


def compute_case_collection_hash(records: Sequence[CaseManifestRecord | Mapping[str, Any]]) -> str:
    """Hash a stable ordered list of immutable case payloads."""

    # Split assignment is tracked by split_manifest_hash and must not make
    # the underlying case content appear to change.
    payloads = [as_json_mapping(record) for record in records]
    content = [
        {key: value for key, value in dict(record).items() if key not in {"split", "case_manifest_hash"}}
        for record in sorted(payloads, key=lambda item: item["case_id"])
    ]
    return f"sha256:{stable_json_hash(tuple(content))}"


def compute_split_manifest_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return f"sha256:{stable_json_hash(tuple(records))}"


def compute_config_hash(config: Mapping[str, Any]) -> str:
    return f"sha256:{stable_json_hash(config)}"
