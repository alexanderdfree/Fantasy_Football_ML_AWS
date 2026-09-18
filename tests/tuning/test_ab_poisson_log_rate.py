"""The sparse-head comparison must retain the diagnostics it is testing."""

import pandas as pd
import pytest

from src.tuning import ab_poisson_log_rate as spec

pytestmark = pytest.mark.unit


def test_cli_defaults_to_eager_for_head_diagnostics(monkeypatch):
    calls = []
    monkeypatch.setattr(spec, "ab_main", lambda name, argv: calls.append((name, argv)))
    spec.main(["--positions", "RB", "--seeds", "42"])
    assert calls == [
        (
            "src.tuning.ab_poisson_log_rate",
            ["--no-stacked-seeds", "--positions", "RB", "--seeds", "42"],
        )
    ]


def test_missing_attention_head_diagnostics_fail_instead_of_succeeding(monkeypatch):
    monkeypatch.setattr(spec, "get_config", lambda pos: {"head_losses": {"count": "poisson_nll"}})
    monkeypatch.setattr(spec, "default_metric_fn", lambda result, pos: {})
    frame = pd.DataFrame({"count": [0, 1], "pred_nn_count": [0.1, 0.1]})
    with pytest.raises(ValueError, match="pred_attn_nn_count.*eager"):
        spec.metric_fn({"test_df": frame}, "RB")


def test_both_models_report_zero_fraction_and_calibration(monkeypatch):
    monkeypatch.setattr(spec, "get_config", lambda pos: {"head_losses": {"count": "poisson_nll"}})
    monkeypatch.setattr(spec, "default_metric_fn", lambda result, pos: {})
    frame = pd.DataFrame(
        {"count": [0, 1], "pred_nn_count": [0.0, 0.0], "pred_attn_nn_count": [0.5, 0.5]}
    )
    metrics = spec.metric_fn({"test_df": frame}, "RB")
    assert metrics["NN:count"]["zero_fraction"] == 1
    assert metrics["NN:count"]["bias"] == -0.5
    assert metrics["Attention NN:count"]["zero_fraction"] == 0
    assert metrics["Attention NN:count"]["bias"] == 0
