import torch

from datiao.r1.tcn.model import PenTCN, model_state_hash
from datiao.r1.tcn.manifest import semantic_run_hash


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
