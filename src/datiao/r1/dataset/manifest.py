"""Hashing helpers for deterministic dataset manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..models.immutable import stable_json_hash
from .models import DatasetManifest


def compute_dataset_manifest_hash(manifest: DatasetManifest | Mapping[str, Any]) -> str:
    """Hash manifest content without introducing a self-referential hash."""

    payload = manifest.model_dump(mode="json") if isinstance(manifest, DatasetManifest) else dict(manifest)
    payload.pop("manifest_hash", None)
    return f"sha256:{stable_json_hash(payload)}"


def compute_case_manifest_hash(records: Sequence[Mapping[str, Any]]) -> str:
    # Split assignment is tracked by split_manifest_hash and must not make
    # the underlying case content appear to change.
    content = []
    for record in records:
        item = dict(record)
        item.pop("split", None)
        content.append(item)
    return f"sha256:{stable_json_hash(tuple(content))}"


def compute_split_manifest_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return f"sha256:{stable_json_hash(tuple(records))}"


def compute_config_hash(config: Mapping[str, Any]) -> str:
    return f"sha256:{stable_json_hash(config)}"
