from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from ..features.io import reload_feature_dataset
from ..features.models import FEATURE_ORDER, LABEL_ORDER

EXPECTED_FEATURE_MANIFEST = "sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848"

@dataclass(frozen=True)
class FeatureMatrices:
    dataset: dict[str, Any]
    X: dict[str, np.ndarray]
    Y: dict[str, np.ndarray]
    episode_ids: dict[str, tuple[str, ...]]

def load_feature_matrices(path: str | Path, expected_manifest: str = EXPECTED_FEATURE_MANIFEST) -> FeatureMatrices:
    dataset = reload_feature_dataset(path)
    manifest = dataset["manifest"]
    if manifest.manifest_hash != expected_manifest:
        raise ValueError(f"unexpected feature manifest: {manifest.manifest_hash}")
    if tuple(manifest.feature_order) != FEATURE_ORDER or tuple(manifest.label_order) != LABEL_ORDER:
        raise ValueError("frozen feature/label order mismatch")
    rows = dataset["feature_rows"]
    X: dict[str, np.ndarray] = {}; Y: dict[str, np.ndarray] = {}; ids: dict[str, tuple[str, ...]] = {}
    for split in ("train", "validation", "test"):
        split_rows = tuple(row for row in rows if row.split == split)
        X[split] = np.asarray([[np.nan if row.features[name] is None else float(row.features[name]) for name in FEATURE_ORDER] for row in split_rows], dtype=np.float64)
        Y[split] = np.asarray([row.target for row in split_rows], dtype=np.int8)
        ids[split] = tuple(row.episode_id for row in split_rows)
        if X[split].shape[1] != len(FEATURE_ORDER) or Y[split].shape[1] != len(LABEL_ORDER):
            raise ValueError(f"matrix shape mismatch for {split}")
    return FeatureMatrices(dataset=dataset, X=X, Y=Y, episode_ids=ids)

def input_integrity_audit(data: FeatureMatrices) -> dict[str, Any]:
    errors: list[str] = []
    if any(data.X[s].shape[1] != 12 for s in data.X): errors.append("X must have exactly 12 columns")
    if any(data.Y[s].shape[1] != 8 for s in data.Y): errors.append("Y must have exactly 8 columns")
    if any(data.X[s].shape[0] == 0 for s in data.X): errors.append("all splits must be non-empty")
    id_sets = [set(data.episode_ids[s]) for s in ("train", "validation", "test")]
    if set.intersection(*id_sets): errors.append("episode IDs overlap across splits")
    if not all(np.all(np.isfinite(x[~np.isnan(x)])) for x in data.X.values()): errors.append("non-finite feature value")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "feature_order": list(FEATURE_ORDER), "label_order": list(LABEL_ORDER),
            "shapes": {s: {"X": list(data.X[s].shape), "Y": list(data.Y[s].shape)} for s in data.X},
            "missing_counts": {s: np.isnan(data.X[s]).sum(axis=0).astype(int).tolist() for s in data.X}}
