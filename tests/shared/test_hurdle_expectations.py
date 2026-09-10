"""Reported hurdle means must match the distribution optimized by the loss."""

import copy
import math

import numpy as np
import pytest
import torch

from src.shared.evaluation import build_gate_info
from src.shared.neural_net import MultiHeadNetWithHistory, build_multihead_net_with_history
from src.shared.registry import get_config, get_inference_spec
from src.shared.training import hurdle_negbin_value_loss, ztnb2_log_prob, ztp_log_prob

pytestmark = pytest.mark.unit


def _model(family="hurdle_negbin", *, dtype=torch.float64):
    cfg = {
        "nn_backbone_layers": [4],
        "nn_head_hidden": 4,
        "nn_dropout": 0.0,
        "attn_d_model": 2,
        "attn_n_heads": 1,
        "attn_gated": True,
        "gated_targets": ["receptions", "receiving_tds"],
        "head_losses": {"receptions": family, "receiving_tds": "poisson_nll"},
    }
    model = build_multihead_net_with_history(
        cfg, static_dim=2, game_dim=1, targets=["receptions", "receiving_tds"]
    ).to(dtype=dtype)
    return model.eval()


def _fix_head(head, *, mu=1.0, alpha=1.0, logit=0.0):
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.gate[-1].bias.fill_(logit)
        head.value_mu[0].bias.fill_(math.log(math.expm1(mu - 1e-6)))
        head.value_log_alpha.bias.fill_(math.log(alpha))


def _inputs(dtype=torch.float64, n=4):
    return (
        torch.zeros(n, 2, dtype=dtype),
        torch.zeros(n, 3, 1, dtype=dtype),
        torch.ones(n, 3, dtype=torch.bool),
    )


@pytest.mark.parametrize("position", ["RB", "WR", "TE"])
def test_production_reception_head_reports_the_fitted_distribution_mean(position):
    cfg = copy.deepcopy(get_config(position))
    model = (
        build_multihead_net_with_history(cfg, static_dim=2, game_dim=1, targets=cfg["targets"])
        .double()
        .eval()
    )
    assert cfg["head_losses"]["receptions"] == "hurdle_negbin"
    _fix_head(model.heads["receptions"])
    out = model(*_inputs())
    # NB(mu=1, alpha=1) is geometric: P0=1/2, positive mean=2.
    # A gate of1/2 therefore has marginal mean1, not1/2.
    torch.testing.assert_close(out["receptions"], torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("family", ["hurdle_negbin", "hurdle_poisson"])
@pytest.mark.parametrize("mu,alpha", [(1e-5, 0.1), (0.2, 1.0), (1.0, 1.0), (20.0, 3.0)])
def test_reported_mean_matches_normalized_discrete_first_moment(family, mu, alpha):
    model = _model(family)
    _fix_head(model.heads["receptions"], mu=mu, alpha=alpha)
    out = model(*_inputs())
    y = torch.arange(1, 4097, dtype=torch.float64)
    raw_mu = out["receptions_value_mu"][0].expand_as(y)
    if family == "hurdle_negbin":
        log_prob = ztnb2_log_prob(y, raw_mu, out["receptions_value_log_alpha"][0].expand_as(y))
    else:
        log_prob = ztp_log_prob(y, raw_mu)
    probability = log_prob.exp()
    torch.testing.assert_close(
        probability.sum(), torch.tensor(1.0, dtype=y.dtype), atol=1e-7, rtol=0
    )
    oracle = (y * probability).sum() * out["receptions_gate_logit"][0].sigmoid()
    torch.testing.assert_close(out["receptions"][0], oracle, atol=1e-7, rtol=1e-7)


def test_mixed_head_routes_preserve_raw_likelihood_and_ordinary_poisson_output():
    model = _model()
    for head in model.heads.values():
        _fix_head(head)
    legacy = _model("huber")
    legacy.load_state_dict(model.state_dict())
    current, original = model(*_inputs()), legacy(*_inputs())
    assert "receiving_tds_value_conditional_mean" not in current
    for target in ("receptions", "receiving_tds"):
        for suffix in ("gate_logit", "value_mu", "value_log_alpha"):
            torch.testing.assert_close(
                current[f"{target}_{suffix}"], original[f"{target}_{suffix}"], rtol=0, atol=0
            )
    torch.testing.assert_close(current["receiving_tds"], original["receiving_tds"], rtol=0, atol=0)
    truth = {"receptions": torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)}
    torch.testing.assert_close(
        hurdle_negbin_value_loss(current, truth, "receptions"),
        hurdle_negbin_value_loss(original, truth, "receptions"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(current["receptions"], 2 * original["receptions"])


@pytest.mark.parametrize("family", ["hurdle_negbin", "hurdle_poisson"])
def test_tiny_rate_and_zero_gate_keep_a_valid_conditional_mean(family):
    model = _model(family, dtype=torch.float32)
    _fix_head(model.heads["receptions"], mu=2e-6, alpha=1e-6, logit=-1000.0)
    out = model(*_inputs(torch.float32))
    assert torch.equal(out["receptions"], torch.zeros(4))
    mean = out["receptions_value_conditional_mean"]
    assert torch.isfinite(mean).all() and (mean >= 1).all()
    info = build_gate_info(
        {key: value.detach().numpy() for key, value in out.items()}, ["receptions"]
    )
    np.testing.assert_array_equal(info["receptions"]["value_mu"], mean.detach().numpy())


def test_mean_gradient_reaches_gate_rate_and_nb_dispersion():
    model = _model()
    head = model.heads["receptions"]
    _fix_head(head)
    model(*_inputs())["receptions"].sum().backward()
    for parameter in (head.gate[-1].bias, head.value_mu[0].bias, head.value_log_alpha.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("family", ["hurdle_negbin", "hurdle_poisson"])
def test_low_precision_hurdle_head_preserves_small_positive_mass(dtype, family):
    head = _model(family, dtype=dtype).heads["receptions"]
    _fix_head(head, mu=2e-6, alpha=1e-6)
    prediction, gate_logit, mu, log_alpha = head(torch.zeros(4, 6, dtype=dtype))
    conditional = head.conditional_mean(mu, log_alpha)
    assert torch.isfinite(prediction).all() and torch.isfinite(conditional).all()
    torch.testing.assert_close(
        prediction.float(), gate_logit.float().sigmoid(), rtol=1e-4, atol=1e-4
    )
    assert (conditional >= 1).all()


@pytest.mark.parametrize("position", ["RB", "WR", "TE"])
def test_serving_reload_interprets_legacy_weights_with_the_same_loss_family(position):
    cfg = copy.deepcopy(get_config(position))
    trained = (
        build_multihead_net_with_history(cfg, static_dim=2, game_dim=1, targets=cfg["targets"])
        .double()
        .eval()
    )
    _fix_head(trained.heads["receptions"])
    # Metadata introduces no parameters or changed tensor shapes, so existing
    # state_dicts reload with the corrected expectation of their fitted law.
    served = (
        MultiHeadNetWithHistory(
            static_dim=2,
            game_dim=1,
            target_names=cfg["targets"],
            **get_inference_spec(position)["attn_nn_kwargs_static"],
        )
        .double()
        .eval()
    )
    served.load_state_dict(trained.state_dict())
    out = served(*_inputs())
    torch.testing.assert_close(out["receptions"], torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(out["receptions"], trained(*_inputs())["receptions"], rtol=0, atol=0)


def test_stacked_functional_forward_preserves_hurdle_expectations():
    from src.tuning.ab_ensemble_seeds import stack_models

    models = [_model(), _model()]
    for index, model in enumerate(models):
        _fix_head(model.heads["receptions"], mu=float(index + 1))
    template, params, buffers = stack_models(models, torch.device("cpu"))
    template.eval()

    def forward(p, b):
        return torch.func.functional_call(template, (p, b), _inputs())

    out = torch.vmap(forward)(params, buffers)
    expected = torch.tensor([1.0, 1.5], dtype=torch.float64).unsqueeze(1).expand(-1, 4)
    torch.testing.assert_close(out["receptions"], expected)
