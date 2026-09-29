import pytest

from datiao.r1.dataset import (
    DatasetConfig,
    SplitConfig,
    load_dataset_config,
    materialize_dataset,
    reload_dataset,
    run_leakage_audit,
)


CONFIG_PATH = "configs/r1_synthetic_dataset_v1.json"


def small_config() -> DatasetConfig:
    config = load_dataset_config(CONFIG_PATH)
    return config.model_copy(update={"participant_count": 3, "split_ratio": SplitConfig(train=0.34, validation=0.33, test=0.33)})


def test_dataset_build_materializes_all_scenarios_and_reloads(tmp_path):
    result = materialize_dataset(small_config(), tmp_path / "dataset")
    assert result.manifest.dataset_type == "synthetic"
    assert result.manifest.participant_count == 3
    assert result.manifest.case_count == 3 * 18
    assert set(result.manifest.scenario_counts) == set(small_config().scenario_types)
    assert result.manifest.task_family_counts
    assert result.summary.audit_status == {"leakage": "PASS", "integrity": "PASS"}
    reloaded = reload_dataset(tmp_path / "dataset")
    by_participant = {}
    by_session = {}
    for case in reloaded["cases"]:
        by_participant.setdefault(case.participant_id, set()).add(case.split)
        by_session.setdefault(case.session_id, set()).add(case.split)
    assert all(len(splits) == 1 for splits in by_participant.values())
    assert all(len(splits) == 1 for splits in by_session.values())
    split_case_ids = [case_id for split in result.split_manifests for case_id in split.case_ids]
    assert len(split_case_ids) == len(set(split_case_ids)) == result.manifest.case_count
    assert len(reloaded["cases"]) == result.manifest.case_count
    assert len(reloaded["points"]) == result.manifest.point_count
    assert len(reloaded["predicted_events"]) == result.manifest.predicted_event_count
    assert len(reloaded["regions"]) > 0


def test_same_config_is_byte_deterministic_and_split_seed_only_changes_assignment(tmp_path):
    config = small_config()
    first = materialize_dataset(config, tmp_path / "a")
    second = materialize_dataset(config, tmp_path / "b")
    assert first.manifest.manifest_hash == second.manifest.manifest_hash
    assert first.manifest.file_hashes == second.manifest.file_hashes

    changed_split = materialize_dataset(config.model_copy(update={"split_seed": config.split_seed + 1}), tmp_path / "c")
    assert changed_split.manifest.case_manifest_hash == first.manifest.case_manifest_hash
    assert changed_split.manifest.split_manifest_hash != first.manifest.split_manifest_hash
    content_files = [key for key in first.manifest.file_hashes if key not in {"config_snapshot.json", "splits/train.json", "splits/validation.json", "splits/test.json", "cases.jsonl"}]
    assert all(first.manifest.file_hashes[key] == changed_split.manifest.file_hashes[key] for key in content_files)

    changed_data = materialize_dataset(config.model_copy(update={"dataset_seed": config.dataset_seed + 1}), tmp_path / "d")
    assert changed_data.manifest.manifest_hash != first.manifest.manifest_hash
    assert changed_data.manifest.case_manifest_hash != first.manifest.case_manifest_hash
    assert changed_data.manifest.file_hashes["raw_records.jsonl"] != first.manifest.file_hashes["raw_records.jsonl"]


def test_leakage_audit_fails_for_participant_and_duplicate_raw_source():
    records = (
        {"case_id": "c1", "participant_id": "p1", "session_id": "s1", "group_id": "p1", "raw_records_hash": "h", "split": "train"},
        {"case_id": "c2", "participant_id": "p1", "session_id": "s2", "group_id": "p1", "raw_records_hash": "h", "split": "test"},
    )
    audit = run_leakage_audit(records)
    assert audit.status == "FAIL"
    assert audit.participant_overlap == ("p1",)
    assert audit.duplicate_raw_source == ("h",)


def test_output_refuses_silent_overwrite(tmp_path):
    output = tmp_path / "dataset"
    materialize_dataset(small_config(), output)
    with pytest.raises(FileExistsError):
        materialize_dataset(small_config(), output)


def test_invalid_dataset_config_rejects_bad_split_ratio():
    payload = load_dataset_config(CONFIG_PATH).model_dump(mode="json")
    payload["split_ratio"] = {"train": 0.8, "validation": 0.1, "test": 0.2}
    with pytest.raises(ValueError, match="sum to 1.0"):
        DatasetConfig.model_validate(payload)
