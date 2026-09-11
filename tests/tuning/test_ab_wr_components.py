"""Prevent a component screen from silently comparing different recipes."""

import itertools

import pytest

from src.shared.neural_net import build_multihead_net, build_multihead_net_with_history
from src.shared.registry import get_config
from src.tuning import ab_wr_components as spec
from src.tuning.ab_harness import resolve_spec

pytestmark = pytest.mark.unit


def test_factorial_is_complete_and_smoke_keeps_baseline():
    assert set(spec.RECIPES.values()) == set(itertools.product([False, True], repeat=3))
    resolved = resolve_spec("src.tuning.ab_wr_components", seeds=[42], only=["corrected"])
    assert list(resolved.variants) == ["baseline", "corrected"]
    assert resolved.positions == ["WR"]


@pytest.mark.parametrize("arm", spec.RECIPES)
def test_each_arm_changes_only_three_switches_and_activates_real_wr_heads(arm):
    config = get_config("WR")
    before = config.copy()
    spec.configure(config, arm=arm)
    switches = {"nn_magnitude_features", "nn_correct_ztnb_mean", "nn_poisson_log_rate"}
    assert config.keys() == before.keys()
    for key in config.keys() - switches:
        assert config[key] is before[key], key
    magnitude, expectation, poisson = spec.RECIPES[arm]
    assert config["nn_magnitude_features"] == (("inherited_opportunity",) if magnitude else ())
    assert config["nn_correct_ztnb_mean"] == expectation
    assert config["nn_poisson_log_rate"] == poisson
    models = {
        "nn": build_multihead_net(config, input_dim=3, targets=config["targets"]),
        "attn_nn": build_multihead_net_with_history(
            config, static_dim=3, game_dim=2, targets=config["targets"]
        ),
    }
    observed = spec.activation(models, arm)
    assert observed["nn_log_rate_heads"] == (2 if poisson else 0)
    assert observed["attn_nn_log_rate_heads"] == (1 if poisson else 0)
    assert observed["attn_reception_expectation_corrected"] == int(expectation)
    if arm != "baseline":
        with pytest.raises(ValueError):
            # The all-legacy expectation must reject every active head switch;
            # magnitude-only is encoded in the scaler rather than the heads.
            if arm == "magnitude_only":
                spec.activation(models, "corrected")
            else:
                spec.activation(models, "baseline")


def test_stacked_execution_is_rejected(monkeypatch):
    monkeypatch.setenv("FF_AB_STACKED", "true")
    with pytest.raises(ValueError, match="nonstacked"):
        spec.configure(get_config("WR"), arm="corrected")
