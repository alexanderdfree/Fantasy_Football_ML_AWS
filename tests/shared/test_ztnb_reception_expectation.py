"""Behavioral regressions for the zero-truncated NB reception expectation.

Isolated from #1575 (component 2 of 3): the gated ``hurdle_negbin`` head fits an
untruncated NB-2 mean ``mu`` but reported ``sigmoid(gate) * mu``; the corrected
expectation is ``sigmoid(gate) * E[Y | Y > 0]``. Legacy checkpoints keep the
legacy law through the per-head version buffer, and a warm start keeps the new
fit's requested mode.
"""

import copy
import math
from decimal import Decimal, localcontext

import pytest
import torch
from scipy.stats import nbinom

from src.shared.neural_net import (
    GatedHead,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
    build_multihead_net_with_history,
    build_multihead_net_with_nested_history,
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


@pytest.mark.parametrize("mu,log_alpha", [(1e-6, -8), (1, 90), (1e-6, 70)])
def test_extreme_mean_and_gradients_match_decimal_reference(mu, log_alpha):
    # Decimal evaluates the distribution directly, independently of the tensor
    # implementation's series/log-domain identities. No model is fitted.
    with localcontext() as context:
        context.prec = 120
        rate, dispersion = Decimal(str(mu)), Decimal(str(log_alpha))

        def expectation(m, a):
            alpha = a.exp()
            zero_mass = (-(1 + alpha * m).ln() / alpha).exp()
            return m / (1 - zero_mass)

        eps = Decimal("1e-25")
        expected = expectation(rate, dispersion)
        d_rate = (
            expectation(rate * (1 + eps), dispersion) - expectation(rate * (1 - eps), dispersion)
        ) / (2 * eps * rate)
        d_dispersion = (
            expectation(rate, dispersion + eps) - expectation(rate, dispersion - eps)
        ) / (2 * eps)

    rate_tensor = torch.tensor([mu], dtype=torch.float32, requires_grad=True)
    dispersion_tensor = torch.tensor([log_alpha], dtype=torch.float32, requires_grad=True)
    actual = ztnb2_conditional_mean(rate_tensor, dispersion_tensor)
    gradients = torch.autograd.grad(actual.sum(), (rate_tensor, dispersion_tensor))
    for value, reference in zip(
        (actual, *gradients), (expected, d_rate, d_dispersion), strict=True
    ):
        assert torch.isfinite(value).all()
        assert value.item() == pytest.approx(float(reference), rel=3e-5, abs=1e-14)


@pytest.mark.parametrize(
    "mu_dtype,alpha_dtype", [(torch.float16, torch.float64), (torch.float32, torch.float16)]
)
def test_probability_arithmetic_preserves_widest_input_dtype(mu_dtype, alpha_dtype):
    result = ztnb2_conditional_mean(
        torch.ones(1, dtype=mu_dtype), torch.tensor([50.0], dtype=alpha_dtype)
    )
    expected_dtype = torch.promote_types(torch.float32, torch.promote_types(mu_dtype, alpha_dtype))
    assert result.dtype == expected_dtype
    assert torch.isfinite(result).all()
    assert result.item() == pytest.approx(math.exp(50) / 50, rel=1e-5)


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
    torch.save(restored.state_dict(), path)
    legacy_resaved = _controlled_head(True)
    legacy_resaved.load_state_dict(torch.load(path, weights_only=True))
    assert not legacy_resaved.correct_ztnb_mean
    torch.testing.assert_close(legacy_resaved(inputs)[0], restored(inputs)[0], rtol=0, atol=0)
    for version in (99, 0.5):
        invalid = dict(new.state_dict(), _ztnb_mean_version=torch.tensor(version))
        with pytest.raises(RuntimeError, match="Unsupported gated-head"):
            restored.load_state_dict(invalid)


@pytest.mark.parametrize("legacy_has_version", [False, True])
def test_warm_start_keeps_new_training_recipe_while_reusing_legacy_weights(legacy_has_version):
    legacy = _controlled_head(False)
    fresh = _controlled_head(True)
    state = legacy.state_dict()
    if not legacy_has_version:
        state.pop("_ztnb_mean_version")
    load_warm_start_state(fresh, state)
    assert fresh.correct_ztnb_mean
    assert fresh._ztnb_mean_version.item() == 1
    assert fresh(torch.zeros((1, 2), dtype=torch.float64))[0].item() == pytest.approx(1.5)
    # The reverse experiment likewise keeps its requested legacy output mode.
    load_warm_start_state(legacy, fresh.state_dict())
    assert not legacy.correct_ztnb_mean
    assert legacy._ztnb_mean_version.item() == 0


@pytest.mark.parametrize("position", ALL_POSITIONS)
def test_all_six_factory_and_serving_configs_agree(position, tmp_path):
    cfg = get_config(position)
    reg = INFERENCE_REGISTRY[position]
    if reg.get("attn_history_structure") == "nested":
        assert position == "K"
        trained = build_multihead_net_with_nested_history(
            cfg,
            static_dim=2,
            kick_dim=2,
            max_games=cfg["attn_max_games"],
            game_dim=len(cfg["attn_history_stats"]),
            targets=list(cfg["targets"]),
        )
        served = MultiHeadNetWithNestedHistory(
            static_dim=2,
            kick_dim=2,
            target_names=list(cfg["targets"]),
            **reg["attn_nn_kwargs_static"],
        )
        assert not any(isinstance(head, GatedHead) for head in trained.modules())
        inputs = (
            torch.zeros(2, 2),
            torch.ones(2, 3, 2, 2),
            torch.ones(2, 3, dtype=torch.bool),
            torch.ones(2, 3, 2, dtype=torch.bool),
            torch.ones(2, 3, len(cfg["attn_history_stats"])),
        )
        _assert_saved_forward_parity(trained, served, inputs, tmp_path)
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
    inputs = (torch.zeros(2, 2), torch.ones(2, 3, 2), torch.ones(2, 3, dtype=torch.bool))
    _assert_saved_forward_parity(trained, served, inputs, tmp_path)


def _assert_saved_forward_parity(trained, served, inputs, tmp_path):
    trained.eval()
    path = tmp_path / "attention.pt"
    torch.save(trained.state_dict(), path)
    served.load_state_dict(torch.load(path, weights_only=True))
    served.eval()
    with torch.no_grad():
        before, after = trained(*inputs), served(*inputs)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)


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
