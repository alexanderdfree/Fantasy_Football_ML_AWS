"""Seed the selected accelerator without initializing unrequested backends."""

import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.shared.utils import seed_everything


@pytest.mark.unit
@pytest.mark.parametrize("requested", ["cpu", "mps", "cuda"])
def test_seed_everything_routes_to_selected_backend(monkeypatch, requested):
    monkeypatch.setenv("FF_DEVICE", requested)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    cuda_seed, mps_seed = Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seed)
    monkeypatch.setattr(torch.mps, "manual_seed", mps_seed)
    seed_everything(42)
    assert cuda_seed.call_count == int(requested == "cuda")
    assert mps_seed.call_count == int(requested == "mps")
    if requested == "mps":
        mps_seed.assert_called_once_with(42)


@pytest.mark.integration
def test_seed_everything_repeats_real_mps_dropout(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("Actual Apple MPS is required")
    monkeypatch.setenv("FF_DEVICE", "mps")
    cpu_state, mps_state = torch.random.get_rng_state(), torch.mps.get_rng_state()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        values = torch.ones(4096, device="mps")
        seed_everything(42)
        first = torch.nn.functional.dropout(values, p=0.5).cpu()
        seed_everything(42)
        second = torch.nn.functional.dropout(values, p=0.5).cpu()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        seed_everything(43)
        different = torch.nn.functional.dropout(values, p=0.5).cpu()
        assert not torch.equal(first, different)
    finally:
        torch.random.set_rng_state(cpu_state)
        torch.mps.set_rng_state(mps_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
