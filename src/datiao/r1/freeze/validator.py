"""Validate an R1 freeze without training or changing any source artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..dataset.io import reload_dataset
from ..features.io import reload_feature_dataset
from ..features.models import FEATURE_ORDER, LABEL_ORDER
from ..lightgbm.io import reload_lightgbm_run
from ..tcn.io import reload_tcn_run
from ..lightgbm.manifest import sha256_file
from ..tcn.manifest import semantic_json_hash


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_freeze(output="artifacts/r1_baseline_freeze_v1", canonical="manifests/r1_baseline_freeze_v1.json", dataset="artifacts/r1_synthetic_penprocess_v1", features="artifacts/r1_synthetic_features_v1", lightgbm_run="artifacts/r1_lightgbm_v1", tcn_run="artifacts/r1_tcn_v1"):
    root = Path(output)
    manifest = _read(root / "freeze_manifest.json")
    if semantic_json_hash({key: value for key, value in manifest.items() if key != "freeze_manifest_hash"}) != manifest["freeze_manifest_hash"]:
        raise ValueError("freeze manifest hash mismatch")
    if manifest != _read(canonical):
        raise ValueError("canonical and artifact freeze manifests differ")
    for name, expected_hash in manifest["report_hashes"].items():
        if sha256_file(root / f"{name}") != expected_hash:
            raise ValueError(f"freeze report hash mismatch: {name}")
    if manifest["status"] != "FROZEN" or manifest["validation_report_status"] != "PASS":
        raise ValueError("freeze is not valid and frozen")
    report = _read(root / "validation_report.json")
    if report.get("status") != "PASS":
        raise ValueError("freeze validation report is not PASS")
    dataset_data = reload_dataset(dataset)
    feature_data = reload_feature_dataset(features)
    lightgbm_data = reload_lightgbm_run(lightgbm_run, features)
    tcn_data = reload_tcn_run(tcn_run, features)
    dataset_manifest = dataset_data["manifest"]
    feature_manifest = feature_data["manifest"]
    if dataset_manifest.manifest_hash != manifest["dataset"]["manifest_hash"] or dataset_manifest.split_manifest_hash != manifest["dataset"]["split_manifest_hash"]:
        raise ValueError("dataset freeze lineage mismatch")
    if feature_manifest.manifest_hash != manifest["feature_dataset"]["manifest_hash"] or feature_manifest.source_dataset_manifest_hash != manifest["feature_dataset"]["source_dataset_manifest_hash"] or feature_manifest.source_split_manifest_hash != manifest["feature_dataset"]["source_split_manifest_hash"]:
        raise ValueError("feature freeze lineage mismatch")
    if tuple(feature_manifest.feature_order) != FEATURE_ORDER or tuple(feature_manifest.label_order) != LABEL_ORDER:
        raise ValueError("feature or label order mismatch")
    if feature_manifest.split_counts != manifest["feature_dataset"]["split_counts"]:
        raise ValueError("feature split lineage mismatch")
    if lightgbm_data["manifest"]["run_hash"] != manifest["lightgbm"]["run_hash"]:
        raise ValueError("LightGBM freeze lineage mismatch")
    if any(lightgbm_data["manifest"][key] != manifest["lightgbm"][key] for key in ("predictions_hash", "metrics_hash")):
        raise ValueError("LightGBM artifact lineage mismatch")
    if tcn_data["manifest"]["run_hash"] != manifest["pen_tcn"]["semantic_run_hash"]:
        raise ValueError("TCN freeze lineage mismatch")
    tcn_manifest = tcn_data["manifest"]
    tcn_keys = {"manifest_integrity_hash": "manifest_integrity_hash", "model_state_hash": "model_state_hash", "model_file_hash": "model_file_hash", "normalization_hash": "normalization_hash", "predictions_hash": "predictions_hash", "metrics_hash": "metrics_hash", "causal_diagnostics_hash": "diagnostics_hash", "comparison_hash": "comparison_hash"}
    if any(tcn_manifest[source] != manifest["pen_tcn"][target] for source, target in tcn_keys.items()):
        raise ValueError("TCN artifact lineage mismatch")
    return {"status": "PASS", "freeze_manifest_hash": manifest["freeze_manifest_hash"], "dataset_reload": True, "feature_reload": True, "lightgbm_reload": True, "tcn_reload": tcn_data["status"] == "PASS"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/r1_baseline_freeze_v1")
    parser.add_argument("--canonical", default="manifests/r1_baseline_freeze_v1.json")
    parser.add_argument("--dataset", default="artifacts/r1_synthetic_penprocess_v1")
    parser.add_argument("--features", default="artifacts/r1_synthetic_features_v1")
    parser.add_argument("--lightgbm-run", default="artifacts/r1_lightgbm_v1")
    parser.add_argument("--tcn-run", default="artifacts/r1_tcn_v1")
    args = parser.parse_args(argv)
    print(json.dumps(validate_freeze(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
