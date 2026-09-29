"""Regression tests for immutable case identity and independently verified assets."""

import json

import pytest

from datiao.r1.dataset import load_dataset_config, materialize_dataset, reload_dataset
from datiao.r1.dataset.audit import _validate_split_semantics, run_integrity_audit
from datiao.r1.dataset.manifest import compute_case_collection_hash, compute_case_record_hash, validate_case_record_hash


@pytest.fixture
def dataset(tmp_path):
    config = load_dataset_config("configs/r1_synthetic_dataset_v1.json").model_copy(update={"participant_count": 6})
    output = tmp_path / "dataset"
    result = materialize_dataset(config, output)
    return config, output, result, reload_dataset(output)


def test_every_case_hash_is_independently_valid_in_all_splits(dataset):
    _, _, result, loaded = dataset
    records = [case.model_dump(mode="json") for case in loaded["cases"]]
    assert {case["split"] for case in records} == {"train", "validation", "test"}
    assert all(validate_case_record_hash(case) for case in records)
    assert all(validate_case_record_hash(case) for case in loaded["cases"])
    for case in records:
        assert compute_case_record_hash(case) == compute_case_record_hash({**case, "split": "test", "case_manifest_hash": "ignored"})
    assert result.manifest.case_manifest_hash == compute_case_collection_hash(records)
    assert compute_case_collection_hash(records) == compute_case_collection_hash(list(reversed(records)))
    changed = {**records[0], "seed": records[0]["seed"] + 1}
    assert not validate_case_record_hash(changed)
    assert compute_case_collection_hash([changed, *records[1:]]) != result.manifest.case_manifest_hash


def test_split_seed_changes_only_split_assets_and_config(dataset, tmp_path):
    config, _, first, loaded = dataset
    second = materialize_dataset(config.model_copy(update={"split_seed": config.split_seed + 1}), tmp_path / "second")
    other = reload_dataset(tmp_path / "second")
    left = {case.case_id: (case.case_manifest_hash, case.raw_records_hash) for case in loaded["cases"]}
    right = {case.case_id: (case.case_manifest_hash, case.raw_records_hash) for case in other["cases"]}
    assert left == right
    assert first.manifest.case_manifest_hash == second.manifest.case_manifest_hash
    assert first.manifest.split_manifest_hash != second.manifest.split_manifest_hash
    assert first.manifest.manifest_hash != second.manifest.manifest_hash
    assert first.manifest.file_hashes["cases.jsonl"] != second.manifest.file_hashes["cases.jsonl"]
    for name in ("raw_records", "points", "strokes", "mappings", "regions", "truth_events", "predicted_events"):
        assert first.manifest.file_hashes[f"{name}.jsonl"] == second.manifest.file_hashes[f"{name}.jsonl"]


@pytest.mark.parametrize("fault,expected", [
    ("missing", "union"),
    ("overlap", "disjoint"),
    ("label", "label"),
    ("participants", "participant_ids"),
    ("groups", "group_ids"),
    ("count", "count"),
    ("dataset_id", "dataset_id"),
    ("dataset_version", "dataset_version"),
    ("split_seed", "split_seed"),
    ("split_version", "split_version"),
    ("split_name", "exactly train"),
])
def test_split_semantics_rejects_inconsistent_objects(dataset, fault, expected):
    config, _, _, loaded = dataset
    cases = [case.model_dump(mode="json") for case in loaded["cases"]]
    splits = list(loaded["split_manifests"])
    victim = splits[-1]
    if fault == "missing":
        victim = victim.model_copy(update={"case_ids": victim.case_ids[1:], "count": victim.count - 1})
    elif fault == "overlap":
        victim = victim.model_copy(update={"case_ids": (*victim.case_ids, splits[0].case_ids[0]), "count": victim.count + 1})
    elif fault == "label":
        next(case for case in cases if case["case_id"] == victim.case_ids[0])["split"] = "train"
    elif fault == "participants":
        victim = victim.model_copy(update={"participant_ids": ("sim_p_wrong",)})
    elif fault == "groups":
        victim = victim.model_copy(update={"group_ids": ("wrong",)})
    elif fault == "count":
        victim = victim.model_copy(update={"count": victim.count + 1})
    elif fault == "split_name":
        victim = victim.model_copy(update={"split_name": "train"})
    else:
        victim = victim.model_copy(update={fault: config.split_seed + 1 if fault == "split_seed" else "wrong"})
    splits[-1] = victim
    valid, errors = _validate_split_semantics(
        cases, splits, expected_dataset_id=config.dataset_id,
        expected_dataset_version=config.dataset_version, expected_split_seed=config.split_seed,
    )
    assert not valid
    assert any(expected in error for error in errors)


def audit_loaded(output, loaded):
    return run_integrity_audit(
        loaded["cases"], loaded["point_records"], loaded["stroke_records"],
        loaded["mapping_records"], loaded["truth_records"], loaded["prediction_records"],
        manifest=loaded["manifest"], dataset_root=output,
        expected_split_seed=loaded["config"].split_seed,
    )


@pytest.mark.parametrize("asset", ["points.jsonl", "splits/test.json"])
def test_actual_file_tampering_fails_audit_and_reload(dataset, asset):
    _, output, _, loaded = dataset
    path = output / asset
    if asset.endswith(".jsonl"):
        path.write_bytes(path.read_bytes() + b" ")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["case_ids"].pop()
        path.write_text(json.dumps(payload), encoding="utf-8")
    audit = audit_loaded(output, loaded)
    assert audit.status == "FAIL"
    assert not audit.hashes
    if asset.startswith("splits/"):
        assert not audit.split_completeness
    with pytest.raises(ValueError, match="hash"):
        reload_dataset(output)


@pytest.mark.parametrize("kind", ["leakage", "integrity"])
def test_reload_recomputes_audits_and_rejects_stale_report(dataset, kind, monkeypatch):
    from datiao.r1.dataset import io

    _, output, _, _ = dataset
    called = []
    for name in ("run_leakage_audit", "run_integrity_audit"):
        original = getattr(io, name)

        def tracked(*args, _original=original, _name=name, **kwargs):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(io, name, tracked)
    path = output / "audits" / f"{kind}_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"stored {kind} audit differs"):
        reload_dataset(output)
    assert called == ["run_leakage_audit", "run_integrity_audit"]


def test_hash_audit_requires_actual_root_and_cannot_accept_boolean_claim(dataset):
    _, _, _, loaded = dataset
    args = (loaded["cases"], loaded["point_records"], loaded["stroke_records"], loaded["mapping_records"], loaded["truth_records"], loaded["prediction_records"])
    audit = run_integrity_audit(*args, split_manifests=loaded["split_manifests"])
    assert audit.status == "FAIL"
    assert not audit.hashes
    with pytest.raises(TypeError):
        run_integrity_audit(*args, file_hashes_valid=True)
