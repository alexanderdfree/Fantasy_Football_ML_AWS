"""Gradient recovery and artifact compatibility for Poisson log-rate heads."""

import importlib

import numpy as np
import pytest
import torch

from src.shared.neural_net import (
    GatedHead,
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
    PoissonLogRateHead,
    build_multihead_net,
    build_multihead_net_with_history,
    build_multihead_net_with_nested_history,
    initialize_poisson_heads,
)
from src.shared.position_pipeline import build_pipeline_config
from src.shared.registry import get_inference_spec
from src.shared.training import MultiTargetLoss

pytestmark = pytest.mark.unit


def _small_model(kind):
    common = dict(
        target_names=["count", "yards"],
        backbone_layers=[8],
        head_hidden=4,
        dropout=0,
        log_rate_targets={"count"},
    )
    static = torch.randn(4, 3)
    if kind == "base":
        return MultiHeadNet(input_dim=3, **common), (static,)
    if kind == "flat":
        model = MultiHeadNetWithHistory(static_dim=3, game_dim=2, d_model=4, **common)
        return model, (static, torch.randn(4, 2, 2), torch.ones(4, 2, dtype=torch.bool))
    model = MultiHeadNetWithNestedHistory(static_dim=3, kick_dim=2, d_model=4, **common)
    return model, (
        static,
        torch.randn(4, 2, 2, 2),
        torch.ones(4, 2, dtype=torch.bool),
        torch.ones(4, 2, 2, dtype=torch.bool),
    )


@pytest.mark.parametrize("kind", ["base", "flat", "nested"])
@pytest.mark.parametrize("raw", [-1.0, -100.0])
def test_positive_labels_reach_negative_head_outputs(kind, raw):
    model, inputs = _small_model(kind)
    model.eval()
    with torch.no_grad():
        model.heads["count"][-1].weight.zero_()
        model.heads["count"][-1].bias.fill_(raw)
    criterion = MultiTargetLoss(["count"], {"count": 1}, head_losses={"count": "poisson_nll"})
    preds = model(*inputs)
    loss, _ = criterion(preds, {"count": torch.ones(4)})
    loss.backward()
    assert torch.isfinite(loss)
    assert model.heads["count"][-1].bias.grad.item() == pytest.approx(np.exp(raw) - 1)
    # The mean remains a non-negative count. The loss uses the separate log
    # rate so a tiny rate cannot suppress the positive label's gradient.
    assert (preds["count"] >= 0).all()
    assert "yards_log_rate" not in preds


@pytest.mark.parametrize("method", ["eager", "capturable_train", "capturable_val"])
def test_log_rate_loss_and_gradient_match_poisson_likelihood(method):
    log_rate = torch.tensor([-100.0, -2.0, 1.0], requires_grad=True)
    y = torch.tensor([1.0, 0.0, 3.0])
    criterion = MultiTargetLoss(["count"], {"count": 2}, head_losses={"count": "poisson_nll"})
    preds = {"count": log_rate.exp(), "count_log_rate": log_rate}
    if method == "eager":
        loss = criterion(preds, {"count": y})[0]
    elif method == "capturable_train":
        loss = criterion.compute_combined_capturable(preds, {"count": y})
    else:
        loss = criterion._compute_loss_components_capturable(preds, {"count": y})[0]
    torch.testing.assert_close(loss, 2 * (log_rate.exp() - y * log_rate).mean())
    loss.backward()
    torch.testing.assert_close(log_rate.grad, 2 * (log_rate.detach().exp() - y) / len(y))


def test_initial_rate_uses_training_prevalence_and_leaves_other_heads_alone():
    model, inputs = _small_model("base")
    model.eval()
    before = model(*inputs)["yards"].detach().clone()
    initialize_poisson_heads(model, {"count": np.array([0, 0, 1, 0], dtype=np.float32)})
    preds = model(*inputs)
    torch.testing.assert_close(preds["count"], torch.full((4,), 0.25))
    torch.testing.assert_close(preds["yards"], before, rtol=0, atol=0)


def test_gated_poisson_head_retains_its_marginal_rate_loss():
    model = MultiHeadNetWithHistory(
        static_dim=3,
        game_dim=2,
        target_names=["count"],
        backbone_layers=[8],
        head_hidden=4,
        gated=True,
        gated_targets=["count"],
        log_rate_targets={"count"},
    ).eval()
    preds = model(torch.randn(4, 3), torch.randn(4, 2, 2), torch.ones(4, 2, dtype=torch.bool))
    assert isinstance(model.heads["count"], GatedHead)
    assert "count_log_rate" not in preds
    torch.testing.assert_close(
        preds["count"], preds["count_gate_logit"].sigmoid() * preds["count_value_mu"]
    )


def _position_model_pair(pos, attention, cfg):
    spec = get_inference_spec(pos)
    targets = spec["targets"]
    if not attention:
        return (
            build_multihead_net(cfg, input_dim=3, targets=targets),
            MultiHeadNet(input_dim=3, target_names=targets, **spec["nn_kwargs"]),
            (torch.randn(4, 3),),
        )
    if pos == "K":
        model = build_multihead_net_with_nested_history(
            cfg, static_dim=3, kick_dim=2, max_games=17, targets=targets, game_dim=16
        )
        served = MultiHeadNetWithNestedHistory(
            static_dim=3, kick_dim=2, target_names=targets, **spec["attn_nn_kwargs_static"]
        )
        inputs = (
            torch.randn(4, 3),
            torch.randn(4, 17, 2, 2),
            torch.ones(4, 17, dtype=torch.bool),
            torch.ones(4, 17, 2, dtype=torch.bool),
            torch.randn(4, 17, 16),
        )
    else:
        model = build_multihead_net_with_history(cfg, static_dim=3, game_dim=2, targets=targets)
        served = MultiHeadNetWithHistory(
            static_dim=3, game_dim=2, target_names=targets, **spec["attn_nn_kwargs_static"]
        )
        inputs = (torch.randn(4, 2, 2), torch.ones(4, 2, dtype=torch.bool))
        inputs = (torch.randn(4, 3), *inputs)
    return model, served, inputs


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_training_and_serving_roundtrip_preserves_new_and_legacy_semantics(pos, attention, legacy):
    pc = importlib.import_module(f"src.{pos.lower()}.config").POSITION_CONFIG
    cfg = build_pipeline_config(pos, pc)
    cfg["nn_poisson_log_rate"] = not legacy
    trained, served, inputs = _position_model_pair(pos, attention, cfg)
    trained.eval()
    served.eval()
    expected = trained(*inputs)
    served.load_state_dict(trained.state_dict(), strict=True)
    actual = served(*inputs)
    assert actual.keys() == expected.keys()
    for target in expected:
        torch.testing.assert_close(actual[target], expected[target], rtol=0, atol=0)
    # A legacy model saved again retains its legacy semantics via marker=0.
    _, reloaded, _ = _position_model_pair(pos, attention, cfg)
    reloaded.eval()
    reloaded.load_state_dict(served.state_dict(), strict=True)
    for target, value in reloaded(*inputs).items():
        torch.testing.assert_close(value, expected[target], rtol=0, atol=0)
    if legacy:
        assert not any(k.endswith("_log_rate") for k in actual)


def test_legacy_poisson_shorthand_and_ab_override():
    cfg = dict(nn_backbone_layers=[8], nn_head_hidden=4, nn_dropout=0, poisson_targets=["count"])
    model = build_multihead_net(cfg, input_dim=3, targets=["count"])
    assert isinstance(model.heads["count"], PoissonLogRateHead)
    legacy = build_multihead_net(
        {**cfg, "nn_poisson_log_rate": False}, input_dim=3, targets=["count"]
    )
    assert not isinstance(legacy.heads["count"], PoissonLogRateHead)


def test_stacked_loss_keeps_negative_log_rate_gradients():
    from src.tuning.ab_ensemble_seeds import stack_models

    models = [_small_model("base")[0].eval() for _ in range(2)]
    for model in models:
        initialize_poisson_heads(model, {"count": np.array([0.01])})
    template, params, buffers = stack_models(models, torch.device("cpu"))
    criterion = MultiTargetLoss(["count"], {"count": 1}, head_losses={"count": "poisson_nll"})
    x = torch.randn(4, 3)

    def loss_for_member(p, b):
        preds = torch.func.functional_call(template, (p, b), (x,))
        return criterion.compute_combined_capturable(preds, {"count": torch.ones(4)})

    losses = torch.vmap(loss_for_member)(params, buffers)
    losses.sum().backward()
    torch.testing.assert_close(params["heads.count.2.bias"].grad, torch.full((2, 1), -0.99))
