"""Behavioral regressions for sparse feature scaling and reception expectations."""

import copy
import math

import joblib
import numpy as np
import pytest
import torch
from scipy.stats import nbinom
from sklearn.preprocessing import StandardScaler

from src.shared.feature_build import MagnitudePreservingScaler, make_nn_scaler, scale_and_clip
from src.shared.neural_net import (
    GatedHead,
    build_multihead_net_with_history,
    load_warm_start_state,
    ztnb2_conditional_mean,
)
from src.shared.pipeline import _scale_xs
from src.shared.registry import ALL_POSITIONS, INFERENCE_REGISTRY, get_config
from src.shared.training import ztnb2_log_prob

pytestmark = pytest.mark.unit


def _sparse_training():
    # Both positive observations collapse to +4 under the old all-row z-score.
    x = np.zeros((100, 2), dtype=np.float64)
    x[:, 0] = np.arange(100)
    x[-2:, 1] = [5, 10]
    return x


def test_positive_magnitudes_survive_clipping_and_zero_stays_zero():
    train = _sparse_training()
    future = np.array([[1, 0], [1, 5.833333], [1, 20.297619], [1, 10000]])
    old = StandardScaler().fit(train)
    assert np.array_equal(scale_and_clip(old, future)[1:, 1], [4, 4, 4])
    scaler, (scaled_train, scaled_future) = _scale_xs(
        train,
        future,
        feature_cols=["ordinary", "inherited_opportunity"],
        magnitude_features=["inherited_opportunity"],
    )
    assert isinstance(scaler, MagnitudePreservingScaler)
    assert scaler.magnitude_scales_.tolist() == [7.5]
    assert scaled_future[0, 1] == 0
    assert np.all(np.diff(scaled_future[:, 1]) > 0)
    assert np.all(scaled_future[:, 1] < 4)
    np.testing.assert_array_equal(scaled_train[:, 0], scale_and_clip(old, train)[:, 0])
    np.testing.assert_array_equal(scaled_future[:, 0], scale_and_clip(old, future)[:, 0])


def test_scaler_roundtrip_train_only_and_in_place(tmp_path):
    train = _sparse_training()
    original = train.copy()
    scaler = MagnitudePreservingScaler((1,)).fit(train)
    future = np.array([[30.0, 6.0], [50.0, 20.0]])
    expected = scaler.transform(future)
    np.testing.assert_array_equal(train, original)
    np.testing.assert_allclose(scaler.inverse_transform(expected), future)
    in_place = future.copy()
    np.testing.assert_array_equal(scaler.transform(in_place, copy=False), expected)
    np.testing.assert_allclose(scaler.inverse_transform(expected.copy(), copy=False), future)
    # Inference data never enters the fitted magnitude scale.
    scaler.transform(np.array([[0.0, 1e9]]))
    assert scaler.magnitude_scales_.tolist() == [7.5]
    path = tmp_path / "scaler.pkl"
    joblib.dump(scaler, path)
    np.testing.assert_array_equal(scale_and_clip(joblib.load(path), future), expected)
    # Historical artifacts remain ordinary StandardScalers.
    old = StandardScaler().fit(train)
    joblib.dump(old, path)
    np.testing.assert_array_equal(
        scale_and_clip(joblib.load(path), future), np.clip(old.transform(future), -4, 4)
    )


def test_absent_features_and_zero_only_training():
    assert type(make_nn_scaler(["ordinary"], ["inherited_opportunity"])) is StandardScaler
    assert type(make_nn_scaler(["inherited_opportunity"], [])) is StandardScaler
    scaler = MagnitudePreservingScaler((0,)).fit(np.zeros((4, 1)))
    values = scaler.transform(np.array([[0], [1], [2]], dtype=float))[:, 0]
    assert values[0] == 0 and np.all(np.diff(values) > 0)
    with pytest.raises(ValueError, match="exact input columns"):
        _scale_xs(np.zeros((3, 2)), feature_cols=["inherited_opportunity"])


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
    cols = cfg["get_feature_columns_fn"]()
    scaler = make_nn_scaler(cols, cfg["nn_magnitude_features"])
    assert isinstance(scaler, MagnitudePreservingScaler) == (position == "WR")
    if reg.get("attn_history_structure") == "nested":
        assert position == "K"
        return
    from src.shared.neural_net import MultiHeadNetWithHistory

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
