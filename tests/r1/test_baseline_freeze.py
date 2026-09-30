import ast
import json
import shutil
from pathlib import Path

import pytest

from datiao.r1.freeze.builder import build_freeze
from datiao.r1.freeze.validator import validate_freeze
from datiao.r1.tcn.manifest import semantic_json_hash


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "artifacts" / "r1_baseline_freeze_v1"
CANONICAL = ROOT / "manifests" / "r1_baseline_freeze_v1.json"


def test_freeze_artifact_validates_and_has_no_winner():
    result = validate_freeze()
    assert result["status"] == "PASS"
    manifest = json.loads((FREEZE / "freeze_manifest.json").read_text(encoding="utf-8"))
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    comparison = json.loads((FREEZE / "baseline_comparison.json").read_text(encoding="utf-8"))
    lightgbm_metrics = json.loads((ROOT / "artifacts" / "r1_lightgbm_v1" / "metrics.json").read_text(encoding="utf-8"))
    tcn_metrics = json.loads((ROOT / "artifacts" / "r1_tcn_v1" / "metrics.json").read_text(encoding="utf-8"))
    tcn_comparison = json.loads((ROOT / "artifacts" / "r1_tcn_v1" / "comparison_lightgbm.json").read_text(encoding="utf-8"))
    assert manifest == canonical
    assert manifest["status"] == "FROZEN"
    assert manifest["source_baseline_commit"] == "2846c7f7346ebe36f12fe63fb5813e2d230c162a"
    assert manifest["feature_dataset"]["feature_order"] == [
        "duration_s", "path_length_norm", "mean_speed_norm_per_s", "pause_before_s",
        "pause_inside_s", "bbox_width_norm", "bbox_height_norm", "question_occupancy",
        "visit_index", "return_count", "previous_ink_iou", "quality_valid",
    ]
    assert manifest["feature_dataset"]["label_order"] == [
        "WRITING", "QUESTION_VISIT", "QUESTION_LEAVE", "RETURN",
        "REVISION_CANDIDATE", "PAGE_CHANGE", "PROCESS_END", "UNKNOWN",
    ]
    assert comparison["description"]
    assert "winner" not in comparison
    assert set(manifest["metrics"]) == {"lightgbm", "pen_tcn"}
    metric_keys = {"micro_f1", "macro_f1_all_labels", "macro_roc_auc_evaluable_labels", "macro_average_precision_evaluable_labels", "hamming_loss", "subset_accuracy"}
    for model, source in (("lightgbm", lightgbm_metrics), ("pen_tcn", tcn_metrics)):
        source_test_key = "test"
        assert set(manifest["metrics"][model]) == {"validation", "synthetic_test"}
        assert set(manifest["metrics"][model]["validation"]) == metric_keys
        assert set(manifest["metrics"][model]["synthetic_test"]) == metric_keys
        assert manifest["metrics"][model]["validation"] == {key: source["validation"]["aggregate"][key] for key in metric_keys}
        assert manifest["metrics"][model]["synthetic_test"] == {key: source[source_test_key]["aggregate"][key] for key in metric_keys}
    assert manifest["comparison_summary"]["synthetic_test"]["tcn_minus_lightgbm"]["micro_f1"] == pytest.approx(0.018523630631603782)
    assert manifest["causal_diagnostics"]["labels"] == ["WRITING", "QUESTION_VISIT", "RETURN", "REVISION_CANDIDATE", "PAGE_CHANGE", "UNKNOWN"]
    for split, source_split in (("validation", "validation"), ("synthetic_test", "test")):
        assert manifest["causal_diagnostics"][split]["pen_tcn"] == {key: tcn_comparison["causal_diagnostics"][source_split]["aggregate"][key] for key in metric_keys}
        assert manifest["causal_diagnostics"][split]["lightgbm"] == {key: tcn_comparison["lightgbm_causal_diagnostics"][source_split]["aggregate"][key] for key in metric_keys}


def test_freeze_builder_has_no_training_entrypoints():
    source = (ROOT / "src" / "datiao" / "r1" / "freeze" / "builder.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all("train" not in module and "training" not in module for module in imported_modules)
    called_names = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert not {"fit", "train", "backward", "optimizer"}.intersection(called_names)


def test_freeze_manifest_tamper_is_rejected(tmp_path):
    output = tmp_path / "freeze"
    shutil.copytree(FREEZE, output)
    canonical = tmp_path / "canonical.json"
    shutil.copy2(CANONICAL, canonical)
    tampered = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    tampered["status"] = "INVALID"
    tampered_path = output / "freeze_manifest.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze manifest hash mismatch"):
        validate_freeze(output=output, canonical=canonical)


def _copy_freeze_for_manifest_tamper(tmp_path, mutate):
    output = tmp_path / "freeze"
    shutil.copytree(FREEZE, output)
    canonical = tmp_path / "canonical.json"
    shutil.copy2(CANONICAL, canonical)
    manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["freeze_manifest_hash"] = semantic_json_hash({key: value for key, value in manifest.items() if key != "freeze_manifest_hash"})
    encoded = json.dumps(manifest)
    (output / "freeze_manifest.json").write_text(encoded, encoding="utf-8")
    canonical.write_text(encoded, encoding="utf-8")
    return output, canonical


@pytest.mark.parametrize("model", ["lightgbm", "pen_tcn"])
def test_tampered_canonical_metric_is_rejected(tmp_path, model):
    output, canonical = _copy_freeze_for_manifest_tamper(
        tmp_path,
        lambda manifest: manifest["metrics"][model]["validation"].__setitem__("micro_f1", 0.0),
    )
    with pytest.raises(ValueError, match="canonical baseline metrics mismatch"):
        validate_freeze(output=output, canonical=canonical)


def test_tampered_comparison_delta_is_rejected(tmp_path):
    output, canonical = _copy_freeze_for_manifest_tamper(
        tmp_path,
        lambda manifest: manifest["comparison_summary"]["validation"]["tcn_minus_lightgbm"].__setitem__("micro_f1", 0.0),
    )
    with pytest.raises(ValueError, match="canonical comparison summary mismatch"):
        validate_freeze(output=output, canonical=canonical)


@pytest.mark.parametrize("kind", ["feature_manifest", "lightgbm_run_hash", "tcn_run_hash", "label_order"])
def test_freeze_source_lineage_tamper_is_rejected(tmp_path, kind):
    dataset = tmp_path / "dataset"
    features = tmp_path / "features"
    lightgbm = tmp_path / "lightgbm"
    tcn = tmp_path / "tcn"
    output = tmp_path / "freeze"
    canonical = tmp_path / "canonical.json"
    shutil.copytree(ROOT / "artifacts" / "r1_synthetic_penprocess_v1", dataset)
    shutil.copytree(ROOT / "artifacts" / "r1_synthetic_features_v1", features)
    shutil.copytree(ROOT / "artifacts" / "r1_lightgbm_v1", lightgbm)
    shutil.copytree(ROOT / "artifacts" / "r1_tcn_v1", tcn)
    shutil.copytree(FREEZE, output)
    shutil.copy2(CANONICAL, canonical)
    if kind in {"feature_manifest", "label_order"}:
        path = features / "feature_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if kind == "feature_manifest":
            value["manifest_hash"] = "sha256:" + "0" * 64
        else:
            value["label_order"] = list(reversed(value["label_order"]))
    elif kind == "lightgbm_run_hash":
        path = lightgbm / "run_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["run_hash"] = "sha256:" + "0" * 64
    else:
        path = tcn / "run_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["run_hash"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        validate_freeze(output=output, canonical=canonical, dataset=dataset, features=features, lightgbm_run=lightgbm, tcn_run=tcn)


def test_freeze_builder_is_byte_deterministic(tmp_path):
    first_manifest = build_freeze(
        "artifacts/r1_synthetic_penprocess_v1", "artifacts/r1_synthetic_features_v1",
        "artifacts/r1_lightgbm_v1", "artifacts/r1_tcn_v1", tmp_path / "one", canonical=tmp_path / "one.json",
    )
    second_manifest = build_freeze(
        "artifacts/r1_synthetic_penprocess_v1", "artifacts/r1_synthetic_features_v1",
        "artifacts/r1_lightgbm_v1", "artifacts/r1_tcn_v1", tmp_path / "two", canonical=tmp_path / "two.json",
    )
    assert first_manifest["freeze_manifest_hash"] == second_manifest["freeze_manifest_hash"]
    for name in ("freeze_manifest.json", "validation_report.json", "baseline_comparison.json", "limitations.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
