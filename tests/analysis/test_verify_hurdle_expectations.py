"""Acceptance probes must not label CPU execution as GPU evidence."""

import pytest


@pytest.mark.unit
def test_hurdle_probe_refuses_cpu(monkeypatch):
    import torch

    from src.analysis.verify_hurdle_expectations import verify_hurdle_expectations

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="actual CUDA"):
        verify_hurdle_expectations()


@pytest.mark.unit
def test_count_probe_refuses_cpu(monkeypatch):
    import torch

    from src.analysis.verify_count_likelihoods import verify_count_likelihoods

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="actual CUDA"):
        verify_count_likelihoods()


@pytest.mark.unit
def test_count_oracle_has_closed_form_geometric_and_poisson_controls():
    import math

    from src.analysis.verify_count_likelihoods import _reference

    assert _reference(1, 1.0, 0.0) == pytest.approx((-math.log(2), -0.5, 2 * math.log(2) - 1.5))
    assert _reference(1, 1.0) == pytest.approx((-math.log(math.expm1(1)), -1 / math.expm1(1), 0.0))


@pytest.mark.unit
def test_model_probe_requires_real_pipeline_attention_and_rows():
    import pandas as pd

    from src.tuning.ab_verify_model_corrections import metric_fn

    with pytest.raises(RuntimeError, match="normal RB pipeline"):
        metric_fn({"test_df": pd.DataFrame(), "attn_nn_metrics": {}}, "RB")


@pytest.mark.unit
def test_model_probe_reads_the_canonical_attention_metric_key(monkeypatch):
    import pandas as pd

    from src.analysis import (
        verify_count_likelihoods,
        verify_hurdle_expectations,
        verify_validation_reduction,
    )
    from src.tuning.ab_verify_model_corrections import metric_fn

    monkeypatch.setattr(
        verify_hurdle_expectations, "verify_hurdle_expectations", lambda seed: {"mock": seed}
    )
    monkeypatch.setattr(
        verify_count_likelihoods, "verify_count_likelihoods", lambda seed: {"mock": seed}
    )
    monkeypatch.setattr(
        verify_validation_reduction, "verify_validation_reduction", lambda seed: {"mock": seed}
    )
    result = metric_fn(
        {"test_df": pd.DataFrame({"week": [1]}), "attn_nn_metrics": {"total": {"mae": 2.0}}}, "RB"
    )
    assert result["pipeline"] == {"attention_mae": 2.0, "test_rows": 1.0}
    assert result["hurdle_expectations"] == result["validation_reduction"] == {"mock": 42}
    assert result["count_likelihoods"] == {"mock": 42}
