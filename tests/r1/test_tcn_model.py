import torch

from datiao.r1.tcn.model import PenTCN, compute_receptive_field, masked_bce_components, masked_bce_with_logits


def test_fixed_receptive_field_is_computed_from_architecture():
    assert compute_receptive_field(2, (1, 2), convs_per_block=2) == 7


def test_forward_returns_eight_logits_per_timestep():
    model = PenTCN()
    logits = model(torch.randn(2, 5, 24))
    assert logits.shape == (2, 5, 8)
    assert torch.isfinite(logits).all()


def test_future_perturbation_does_not_change_earlier_logits():
    torch.manual_seed(7)
    model = PenTCN().eval()
    x = torch.randn(1, 6, 24)
    changed = x.clone()
    changed[:, 4:] += 100.0
    with torch.no_grad():
        first = model(x)
        second = model(changed)
    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-7, rtol=0)


def test_padding_invariance_in_eval_mode():
    torch.manual_seed(8)
    model = PenTCN().eval()
    short = torch.randn(1, 2, 24)
    long = torch.cat([short, torch.randn(1, 5, 24)], dim=1)
    padded = torch.cat([short, torch.zeros(1, 5, 24)], dim=1)
    with torch.no_grad():
        alone = model(short)
        together = model(torch.cat([padded, long], dim=0))
    assert torch.allclose(alone, together[:1, :2], atol=2e-7, rtol=0)


def test_masked_loss_ignores_padding_targets():
    logits = torch.zeros(1, 3, 8, requires_grad=True)
    targets = torch.zeros(1, 3, 8)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
    first = masked_bce_with_logits(logits, targets, mask)
    targets[:, 2] = 1.0
    second = masked_bce_with_logits(logits, targets, mask)
    assert torch.equal(first, second)


def test_masked_bce_components_returns_global_sum_and_valid_label_count():
    logits = torch.zeros(1, 3, 2)
    targets = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    padding = torch.tensor([[1., 1., 0.]])
    loss_sum, valid_count = masked_bce_components(logits, targets, padding)
    assert valid_count.item() == 4
    assert torch.allclose(loss_sum, torch.tensor(4 * 0.6931471805599453), atol=1e-6)


def test_real_optimizer_step_changes_trainable_parameter_and_stays_finite():
    model = PenTCN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 24)
    y = torch.randint(0, 2, (2, 3, 8)).float()
    mask = torch.ones(2, 3)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    loss = masked_bce_with_logits(model(x), y, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    assert any(not torch.equal(before[name], param) for name, param in model.named_parameters())
