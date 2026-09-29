"""Build a small, deterministic R1 freeze record from already-frozen runs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..dataset.io import reload_dataset
from ..features.io import reload_feature_dataset
from ..features.models import FEATURE_ORDER, LABEL_ORDER
from ..lightgbm.io import reload_lightgbm_run
from ..lightgbm.manifest import sha256_file, write_json
from ..tcn.io import reload_tcn_run
from ..tcn.manifest import semantic_json_hash
from .models import R1_BASELINE_FREEZE_V1

SOURCE_COMMIT = R1_BASELINE_FREEZE_V1.source_baseline_commit
EXPECTED = {
    "dataset_manifest": "sha256:c493f792f9add5547427d002eb74fa0cc0bb663f7f0bb9b19c2bd8c97674af82",
    "split_manifest": "sha256:434580886db58540ad7b16e5dc777ad99063a7d16a7886237af258e9b1a85499",
    "feature_manifest": "sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848",
    "lightgbm_run": "sha256:3b6235b64974c244fe3dc80f81798d815404dc48786323ca1fbb94ead33118e0",
    "lightgbm_predictions": "sha256:42972dcff9a63bd16d505919fa109683ce3a87286b06537e205556e594b7b553",
    "lightgbm_metrics": "sha256:440ebc9995e70b5035b88c2513d5284301ba3026413999ae688cb1cb5c0e0fdc",
    "tcn_run": "sha256:1ff84b184c64240cfe4b0f97b0490ba025879b2ac30aa537f445e7ab86a11f62",
    "tcn_integrity": "sha256:582cea439ce3fae91cb5d3e565a01cb503eefc512b32a0075d90c871e73dade7",
    "tcn_state": "sha256:c516c8fa9ec6127c7d9bfeaa456a8604ed16ed2ada46e55223684255a2edb585",
    "tcn_model": "sha256:8840d665ac0b2877906c35ce4040abc125a9884b6780f2e711bf36425958dc5f",
    "tcn_normalization": "sha256:a9be2db2b2efafdc9a5eb6d1ccf1569d1525e8342faa7aa13c7ae6792de7929c",
    "tcn_predictions": "sha256:0aa434b52a352c5b7add0c25a2d1a1042c715ea59c86a40509c97bd6ce300b56",
    "tcn_metrics": "sha256:d87cfbe4b091795cba5872713e46c776c65fd61414fcdf6eb92e49f8d21702cc",
    "tcn_diagnostics": "sha256:9f9cbd8470ddafdef374756c82a0a8f66e6096063bda40a67df11a1c432f1738",
    "tcn_comparison": "sha256:b138884303d4c068edf536055007a298b875d444c3edfdd4575f7de57a10d852",
}
def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _aggregate(metrics):
    aggregate = metrics["aggregate"]
    return {key: aggregate[key] for key in ("micro_f1", "macro_f1_all_labels", "macro_roc_auc_evaluable_labels", "macro_average_precision_evaluable_labels", "hamming_loss", "subset_accuracy")}


def _coverage(rows):
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    return {
        "counts": {split: len(items) for split, items in by_split.items()},
        "unique_episode_ids": {split: len({item["episode_id"] for item in items}) for split, items in by_split.items()},
        "all_unique": len({item["episode_id"] for item in rows}) == len(rows),
        "total": len(rows),
    }


def _target_match(left, right, split):
    a = {row["episode_id"]: row for row in left if row["split"] == split}
    b = {row["episode_id"]: row for row in right if row["split"] == split}
    return set(a) == set(b) and all(a[key]["target"] == b[key]["target"] for key in a)


def _metric_delta(left, right):
    return {key: left[key] - right[key] for key in ("micro_f1", "macro_f1_all_labels", "macro_roc_auc_evaluable_labels")}


def _support_metadata(lightgbm_manifest):
    result = {}
    for label in LABEL_ORDER:
        learner = lightgbm_manifest["label_learners"][label]
        result[label] = {
            split: {"positive": learner[f"{split}_positive"], "negative": learner[f"{split}_negative"]}
            for split in ("train", "validation", "test")
        }
    return result


def build_freeze(dataset, features, lightgbm_run, tcn_run, output, overwrite=False, canonical="manifests/r1_baseline_freeze_v1.json"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    dataset_data = reload_dataset(dataset)
    feature_data = reload_feature_dataset(features)
    lightgbm_data = reload_lightgbm_run(lightgbm_run, features)
    tcn_data = reload_tcn_run(tcn_run, features)
    dataset_manifest = dataset_data["manifest"].model_dump(mode="json")
    feature_manifest = feature_data["manifest"].model_dump(mode="json")
    lightgbm_manifest = lightgbm_data["manifest"]
    tcn_manifest = _read_json(Path(tcn_run) / "run_manifest.json")
    lightgbm_rows = _read_jsonl(Path(lightgbm_run) / "predictions.jsonl")
    tcn_rows = _read_jsonl(Path(tcn_run) / "predictions.jsonl")
    lightgbm_metrics = lightgbm_data["metrics"]
    tcn_metrics = _read_json(Path(tcn_run) / "metrics.json")

    checks = {
        "dataset_reload": True,
        "feature_reload": True,
        "lightgbm_reload": True,
        "tcn_reload": tcn_data["status"] == "PASS",
        "dataset_hash_match": dataset_manifest["manifest_hash"] == EXPECTED["dataset_manifest"],
        "feature_hash_match": feature_manifest["manifest_hash"] == EXPECTED["feature_manifest"],
        "lightgbm_hash_match": lightgbm_manifest["run_hash"] == EXPECTED["lightgbm_run"] and lightgbm_manifest["predictions_hash"] == EXPECTED["lightgbm_predictions"] and lightgbm_manifest["metrics_hash"] == EXPECTED["lightgbm_metrics"],
        "tcn_hash_match": all(tcn_manifest[key] == EXPECTED[name] for key, name in (("run_hash", "tcn_run"), ("manifest_integrity_hash", "tcn_integrity"), ("model_state_hash", "tcn_state"), ("model_file_hash", "tcn_model"), ("normalization_hash", "tcn_normalization"), ("predictions_hash", "tcn_predictions"), ("metrics_hash", "tcn_metrics"), ("causal_diagnostics_hash", "tcn_diagnostics"), ("comparison_hash", "tcn_comparison"))),
        "feature_order_match": tuple(feature_manifest["feature_order"]) == FEATURE_ORDER,
        "label_order_match": tuple(feature_manifest["label_order"]) == LABEL_ORDER,
        "split_match": feature_manifest["split_counts"] == {"train": {"episodes": 490, "sequences": 252}, "validation": {"episodes": 105, "sequences": 54}, "test": {"episodes": 105, "sequences": 54}},
        "lightgbm_prediction_episode_coverage": _coverage(lightgbm_rows),
        "tcn_prediction_episode_coverage": _coverage(tcn_rows),
        "validation_episode_equality": set(row["episode_id"] for row in lightgbm_rows if row["split"] == "validation") == set(row["episode_id"] for row in tcn_rows if row["split"] == "validation"),
        "test_episode_equality": set(row["episode_id"] for row in lightgbm_rows if row["split"] == "test") == set(row["episode_id"] for row in tcn_rows if row["split"] == "test"),
        "validation_target_equality": _target_match(lightgbm_rows, tcn_rows, "validation"),
        "test_target_equality": _target_match(lightgbm_rows, tcn_rows, "test"),
        "feature_leakage": feature_data["leakage_audit"]["status"],
        "feature_alignment": feature_data["alignment_audit"]["status"],
        "feature_range": feature_data["range_audit"]["status"],
        "feature_split": feature_data["split_audit"]["status"],
    }
    coverage_ok = all(
        value["total"] == 700 and value["all_unique"] and value["counts"] == {"train": 490, "validation": 105, "test": 105}
        for value in (checks["lightgbm_prediction_episode_coverage"], checks["tcn_prediction_episode_coverage"])
    )
    bool_checks_ok = all(value is True or value == "PASS" for key, value in checks.items() if key not in {"lightgbm_prediction_episode_coverage", "tcn_prediction_episode_coverage"})
    checks["status"] = "PASS" if bool_checks_ok and coverage_ok else "FAIL"
    validation_report = checks
    write_json(output / "validation_report.json", validation_report)

    comparison = {
        "description": "LightGBM and Pen TCN are two complementary frozen R1 baselines.",
        "validation": {"lightgbm": _aggregate(lightgbm_metrics["validation"]), "tcn": _aggregate(tcn_metrics["validation"]), "tcn_minus_lightgbm": _metric_delta(_aggregate(tcn_metrics["validation"]), _aggregate(lightgbm_metrics["validation"]))},
        "synthetic_test": {"lightgbm": _aggregate(lightgbm_metrics["test"]), "tcn": _aggregate(tcn_metrics["test"]), "tcn_minus_lightgbm": _metric_delta(_aggregate(tcn_metrics["test"]), _aggregate(lightgbm_metrics["test"]))},
        "causal_diagnostics": {"validation": {"lightgbm": tcn_manifest["comparison"]["lightgbm_causal_diagnostics"]["validation"], "tcn": tcn_manifest["comparison"]["causal_diagnostics"]["validation"]}, "synthetic_test": {"lightgbm": tcn_manifest["comparison"]["lightgbm_causal_diagnostics"]["test"], "tcn": tcn_manifest["comparison"]["causal_diagnostics"]["test"]}},
        "per_label": {"validation": {"lightgbm": lightgbm_metrics["validation"]["per_label"], "tcn": tcn_metrics["validation"]["per_label"]}, "synthetic_test": {"lightgbm": lightgbm_metrics["test"]["per_label"], "tcn": tcn_metrics["test"]["per_label"]}},
    }
    write_json(output / "baseline_comparison.json", comparison)

    limitations = {
        "synthetic_only": True,
        "rule_generated_shortcut_risk": True,
        "writing_degenerate_all_positive": True,
        "question_leave_retrospective": True,
        "process_end_external_signal": True,
        "short_sequence_limited_context": True,
        "tcn_receptive_field_not_reached": {"receptive_field": 7, "sequences_reaching": 0, "ratio": 0.0},
        "tcn_max_epoch_boundary": {"best_epoch": tcn_manifest["best_epoch"], "max_epochs": 200, "early_stopping_before_cap": tcn_manifest["best_epoch"] < 200},
        "dataset_type": dataset_manifest["dataset_type"],
    }
    write_json(output / "limitations.json", limitations)

    report_hashes = {name: sha256_file(output / name) for name in ("validation_report.json", "baseline_comparison.json", "limitations.json")}
    freeze_manifest = {
        "freeze_id": R1_BASELINE_FREEZE_V1.freeze_id,
        "freeze_version": R1_BASELINE_FREEZE_V1.freeze_version,
        "status": "FROZEN" if checks["status"] == "PASS" else "INVALID",
        "source_baseline_commit": SOURCE_COMMIT,
        "dataset": {"id": dataset_manifest["dataset_id"], "version": dataset_manifest["dataset_version"], "type": dataset_manifest["dataset_type"], "manifest_hash": dataset_manifest["manifest_hash"], "split_manifest_hash": dataset_manifest["split_manifest_hash"]},
        "feature_dataset": {"id": feature_manifest["feature_dataset_id"], "version": feature_manifest["feature_dataset_version"], "manifest_hash": feature_manifest["manifest_hash"], "source_dataset_manifest_hash": feature_manifest["source_dataset_manifest_hash"], "source_split_manifest_hash": feature_manifest["source_split_manifest_hash"], "feature_order": feature_manifest["feature_order"], "label_order": feature_manifest["label_order"], "episodes": feature_manifest["episode_count"], "sequences": feature_manifest["sequence_count"], "split_counts": feature_manifest["split_counts"]},
        "lightgbm": {"baseline_id": lightgbm_manifest["baseline_id"], "version": lightgbm_manifest["baseline_version"], "run_hash": lightgbm_manifest["run_hash"], "predictions_hash": lightgbm_manifest["predictions_hash"], "metrics_hash": lightgbm_manifest["metrics_hash"]},
        "pen_tcn": {"baseline_id": tcn_manifest["baseline_id"], "version": tcn_manifest["baseline_version"], "semantic_run_hash": tcn_manifest["run_hash"], "manifest_integrity_hash": tcn_manifest["manifest_integrity_hash"], "model_state_hash": tcn_manifest["model_state_hash"], "model_file_hash": tcn_manifest["model_file_hash"], "normalization_hash": tcn_manifest["normalization_hash"], "predictions_hash": tcn_manifest["predictions_hash"], "metrics_hash": tcn_manifest["metrics_hash"], "diagnostics_hash": tcn_manifest["causal_diagnostics_hash"], "comparison_hash": tcn_manifest["comparison_hash"]},
        "metrics": {"validation": _aggregate(tcn_metrics["validation"]), "synthetic_test": _aggregate(tcn_metrics["test"])},
        "support_metadata": _support_metadata(lightgbm_manifest),
        "target_causality": _read_json(Path(tcn_run) / "target_causality_audit.json"),
        "temporal_context": _read_json(Path(tcn_run) / "sequence_length_audit.json"),
        "limitations": limitations,
        "report_hashes": report_hashes,
        "validation_report_status": checks["status"],
    }
    freeze_manifest["freeze_manifest_hash"] = semantic_json_hash(freeze_manifest)
    write_json(output / "freeze_manifest.json", freeze_manifest)
    canonical_path = Path(canonical)
    write_json(canonical_path, freeze_manifest)
    return freeze_manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/r1_synthetic_penprocess_v1")
    parser.add_argument("--features", default="artifacts/r1_synthetic_features_v1")
    parser.add_argument("--lightgbm-run", default="artifacts/r1_lightgbm_v1")
    parser.add_argument("--tcn-run", default="artifacts/r1_tcn_v1")
    parser.add_argument("--output", default="artifacts/r1_baseline_freeze_v1")
    parser.add_argument("--canonical", default="manifests/r1_baseline_freeze_v1.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_freeze(args.dataset, args.features, args.lightgbm_run, args.tcn_run, args.output, args.overwrite, args.canonical), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
