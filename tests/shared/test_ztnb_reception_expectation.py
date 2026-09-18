"""Behavioral regressions for the zero-truncated NB reception expectation.

Isolated from #1575 (component 2 of 3): the gated ``hurdle_negbin`` head fits an
untruncated NB-2 mean ``mu`` but reported ``sigmoid(gate) * mu``; the corrected
expectation is ``sigmoid(gate) * E[Y | Y > 0]``. Legacy checkpoints keep the
legacy law through the per-head version buffer, and a warm start keeps the new
fit's requested mode.
"""

import copy
import math

import pytest
import torch
from scipy.stats import nbinom

from src.shared.neural_net import (
    GatedHead,
    MultiHeadNetWithHistory,
    build_multihead_net_with_history,
    load_warm_start_state,
    ztnb2_conditional_mean,
)
from src.shared.registry import ALL_POSITIONS, INFERENCE_REGISTRY, get_config
from src.shared.training import ztnb2_log_prob

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mu,alpha", [(1e-5, 0.2), (0.1, 0.001), (1, 1), (3, 0.5), (100, 10)])
def test_conditional_mean_matches_independent_distribution(mu, alpha):
    rate = torch.tensor([mu], dtype=torch.float64, requires_grad=True)
    log_alpha = torch.tensor([math.log(alpha)], dtype=torch.float64, requires_grad=True)
    actual = ztnb2_conditional_mean(rate, log_alpha)
    r = 1 / alpha
    expected = mu / nbinom.sf(0, r, r / (r + mu))
    assert actual.item() == pytest.approx(expected, rel=1e-8)
    actual.sum().backward()
    assert torch.isfinite(rate.grad).all() and (rate.grad > 0).all()
    assert torch.isfinite(log_alpha.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_probability_arithmetic_uses_fp32_under_autocast_dtypes(dtype):
    result = ztnb2_conditional_mean(
        torch.tensor([1e-5, 1], dtype=dtype), torch.zeros(2, dtype=dtype)
    )
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([1.00001, 2.0]), atol=1e-5, rtol=1e-5)


def _controlled_head(correct):
    head = GatedHead(2, gate_hidden=2, value_hidden=2, correct_ztnb_mean=correct).double()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.gate[-1].bias.fill_(math.log(3))  # P(positive) = 0.75
        head.value_mu[0].bias.fill_(math.log(math.expm1(1 - 1e-6)))
    return head


def test_reported_expectation_agrees_with_fitted_probability_mass():
    head = _controlled_head(True)
    prediction, gate, mu, dispersion = head(torch.zeros((1, 2), dtype=torch.float64))
    values = torch.arange(1, 1000, dtype=torch.float64)
    mass = ztnb2_log_prob(values, mu, dispersion).exp()
    assert mass.sum().item() == pytest.approx(1)
    assert prediction.item() == pytest.approx((torch.sigmoid(gate) * (values * mass).sum()).item())
    assert prediction.item() == pytest.approx(1.5)
    assert _controlled_head(False)(torch.zeros((1, 2), dtype=torch.float64))[
        0
    ].item() == pytest.approx(0.75)


def test_checkpoint_controls_new_and_legacy_expectation(tmp_path):
    inputs = torch.zeros((1, 2), dtype=torch.float64)
    new = _controlled_head(True)
    path = tmp_path / "head.pt"
    torch.save(new.state_dict(), path)
    restored = _controlled_head(False)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.correct_ztnb_mean
    torch.testing.assert_close(restored(inputs)[0], new(inputs)[0])
    legacy = {key: value for key, value in new.state_dict().items() if key != "_ztnb_mean_version"}
    restored.load_state_dict(legacy, strict=True)
    assert not restored.correct_ztnb_mean
    assert restored(inputs)[0].item() == pytest.approx(0.75)
    assert restored.state_dict()["_ztnb_mean_version"].item() == 0
    for version in (99, 0.5):
        invalid = dict(new.state_dict(), _ztnb_mean_version=torch.tensor(version))
        with pytest.raises(RuntimeError, match="Unsupported gated-head"):
            restored.load_state_dict(invalid)


def test_warm_start_keeps_new_training_recipe_while_reusing_legacy_weights():
    legacy = _controlled_head(False)
    fresh = _controlled_head(True)
    load_warm_start_state(fresh, legacy.state_dict())
    assert fresh.correct_ztnb_mean
    assert fresh._ztnb_mean_version.item() == 1
    assert fresh(torch.zeros((1, 2), dtype=torch.float64))[0].item() == pytest.approx(1.5)
    # The reverse experiment likewise keeps its requested legacy output mode.
    load_warm_start_state(legacy, fresh.state_dict())
    assert not legacy.correct_ztnb_mean
    assert legacy._ztnb_mean_version.item() == 0


@pytest.mark.parametrize("position", ALL_POSITIONS)
def test_all_six_factory_and_serving_configs_agree(position):
    cfg = get_config(position)
    reg = INFERENCE_REGISTRY[position]
    if reg.get("attn_history_structure") == "nested":
        assert position == "K"
        return
    trained = build_multihead_net_with_history(
        cfg, static_dim=2, game_dim=2, targets=list(cfg["targets"])
    )
    served = MultiHeadNetWithHistory(
        static_dim=2, game_dim=2, target_names=list(cfg["targets"]), **reg["attn_nn_kwargs_static"]
    )
    trained_modes = {
        name: head.correct_ztnb_mean
        for name, head in trained.heads.items()
        if isinstance(head, GatedHead)
    }
    served_modes = {
        name: head.correct_ztnb_mean
        for name, head in served.heads.items()
        if isinstance(head, GatedHead)
    }
    assert trained_modes == served_modes
    assert {name for name, mode in trained_modes.items() if mode} == (
        {"receptions"} if position in {"RB", "WR", "TE"} else set()
    )
    served.load_state_dict(trained.state_dict())


def test_corrected_head_supports_stacked_forward_and_gradients():
    heads = [_controlled_head(True), _controlled_head(True)]
    template = copy.deepcopy(heads[0])
    parameters, buffers = torch.func.stack_module_state(heads)
    x = torch.ones((3, 2), dtype=torch.float64)

    def forward(p, b):
        return torch.func.functional_call(template, (p, b), (x,))[0]

    predictions = torch.vmap(forward)(parameters, buffers)
    torch.testing.assert_close(predictions, torch.full((2, 3), 1.5, dtype=torch.float64))
    predictions.sum().backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all() for value in parameters.values()
    )
