"""The CUDA count-likelihood probe must not label CPU execution as GPU evidence."""

import math

import pytest


@pytest.mark.unit
def test_count_probe_refuses_cpu(monkeypatch):
    import torch

    from src.analysis.verify_count_likelihoods import verify_count_likelihoods

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="actual CUDA"):
        verify_count_likelihoods()


@pytest.mark.unit
def test_count_oracle_has_closed_form_geometric_and_poisson_controls():
    from src.analysis.verify_count_likelihoods import _reference

    assert _reference(1, 1.0, 0.0) == pytest.approx((-math.log(2), -0.5, 2 * math.log(2) - 1.5))
    assert _reference(1, 1.0) == pytest.approx((-math.log(math.expm1(1)), -1 / math.expm1(1), 0.0))
