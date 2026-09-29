import torch

from datiao.r1.tcn.model import PenTCN, model_state_hash
from datiao.r1.tcn.manifest import manifest_integrity_hash, semantic_run_hash


def test_model_state_hash_is_semantic_and_stable():
    torch.manual_seed(12)
    first = PenTCN()
    second = PenTCN()
    second.load_state_dict(first.state_dict())
    assert model_state_hash(first) == model_state_hash(second)


def test_semantic_run_hash_excludes_model_file_hash():
    base = {"model_state_hash": "sha256:state", "predictions_hash": "sha256:pred", "model_file_hash": "sha256:file-a"}
    changed = dict(base, model_file_hash="sha256:file-b")
    assert semantic_run_hash(base) == semantic_run_hash(changed)


def test_semantic_hash_ignores_artifact_container_hashes_but_integrity_hash_changes():
    base = {
        "model_state_hash": "sha256:state",
        "predictions_hash": "sha256:pred",
        "metrics_hash": "sha256:metrics",
        "artifact_hashes": {"model/best_model.pt": "sha256:file-a", "environment.json": "sha256:env-a"},
    }
    changed = {**base, "artifact_hashes": {"model/best_model.pt": "sha256:file-b", "environment.json": "sha256:env-b"}}
    assert semantic_run_hash(base) == semantic_run_hash(changed)
    assert manifest_integrity_hash(base) != manifest_integrity_hash(changed)


def test_semantic_content_changes_change_run_hash():
    base = {"model_state_hash": "sha256:state", "predictions_hash": "sha256:pred", "metrics_hash": "sha256:metrics", "normalization_hash": "sha256:norm", "config_snapshot_hash": "sha256:config", "architecture": {"receptive_field": 7}}
    for field in ("model_state_hash", "predictions_hash", "metrics_hash", "normalization_hash", "config_snapshot_hash", "architecture"):
        changed = dict(base)
        changed[field] = {"receptive_field": 8} if field == "architecture" else "sha256:changed"
        assert semantic_run_hash(base) != semantic_run_hash(changed)


def test_manifest_integrity_hash_is_stored_after_run_hash_and_covers_it():
    payload = {"model_state_hash": "sha256:state", "artifact_hashes": {"model/best_model.pt": "sha256:file"}}
    payload["run_hash"] = semantic_run_hash(payload)
    payload["manifest_integrity_hash"] = manifest_integrity_hash(payload)
    assert semantic_run_hash(payload) == payload["run_hash"]
    tampered = dict(payload, run_hash="sha256:tampered")
    assert manifest_integrity_hash(tampered) != payload["manifest_integrity_hash"]
