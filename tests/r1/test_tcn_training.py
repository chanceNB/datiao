import inspect

import torch

from datiao.r1.features.models import SequenceSample
from datiao.r1.tcn.data import Normalizer, make_loader
from datiao.r1.tcn.model import PenTCN
from datiao.r1.tcn.train import evaluate, fit


def test_fit_api_does_not_accept_test_loader():
    assert "test_loader" not in inspect.signature(fit).parameters


def test_best_state_snapshot_is_not_aliased_to_live_model():
    model = PenTCN()
    state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(1, 2, 24)).square().mean()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(state[key], model.state_dict()[key]) for key in state)


def test_evaluate_accepts_loader_containing_only_one_split():
    sample = SequenceSample(
        sequence_id="sequence-1",
        case_id="case-1",
        participant_id="participant-1",
        task_segment_id="segment-1",
        split="validation",
        values=[[0.0] * 12, [0.0] * 12],
        feature_mask=[[1] * 12, [1] * 12],
        targets=[[0] * 8, [0] * 8],
        episode_ids=["episode-1", "episode-2"],
    )
    normalizer = Normalizer(
        mean=[0.0] * 12,
        std=[1.0] * 12,
        observed_count=[1] * 12,
        zero_variance_features=[],
    )
    loader = make_loader([sample], batch_size=1, shuffle=False, seed=1)
    result = evaluate(PenTCN().eval(), loader, normalizer, threshold=0.5)
    assert set(result.metrics) == {"validation"}
