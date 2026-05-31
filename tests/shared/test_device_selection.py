"""Unit tests for the ``FF_DEVICE`` / ``--device`` resolver in src.shared.utils.

These guard the contract ``run_pipeline --device`` relies on: ``auto`` reproduces
the historical ``torch.cuda.is_available()`` behaviour (so Linux/macOS/CI are
unchanged when the flag is omitted), ``cpu`` forces the CPU path even on a
CUDA-visible host, and ``cuda`` fails loudly when no GPU is visible rather than
silently degrading to CPU.
"""

from __future__ import annotations

import pytest
import torch

from src.shared.utils import cuda_enabled, requested_device


@pytest.mark.unit
def test_requested_device_defaults_to_auto_when_unset(monkeypatch):
    monkeypatch.delenv("FF_DEVICE", raising=False)
    assert requested_device() == "auto"


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        ("auto", "auto"),
        ("CUDA", "cuda"),  # case-insensitive
        ("  cpu  ", "cpu"),  # surrounding whitespace tolerated
        ("gpu", "auto"),  # unrecognised → safe auto fallback, never a wrong guess
        ("", "auto"),
    ],
)
def test_requested_device_normalises_and_falls_back(monkeypatch, raw, expected):
    monkeypatch.setenv("FF_DEVICE", raw)
    assert requested_device() == expected


@pytest.mark.unit
@pytest.mark.parametrize("available", [True, False])
def test_cuda_enabled_auto_mirrors_torch_availability(monkeypatch, available):
    """auto (default) == torch.cuda.is_available(): the unchanged historical path."""
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert cuda_enabled() is available


@pytest.mark.unit
@pytest.mark.parametrize("available", [True, False])
def test_cuda_enabled_cpu_forces_off_even_when_available(monkeypatch, available):
    """--device cpu never uses CUDA, even on a CUDA-visible box."""
    monkeypatch.setenv("FF_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert cuda_enabled() is False


@pytest.mark.unit
def test_cuda_enabled_cuda_uses_gpu_when_available(monkeypatch):
    monkeypatch.setenv("FF_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cuda_enabled() is True


@pytest.mark.unit
def test_cuda_enabled_cuda_raises_when_unavailable(monkeypatch):
    """--device cuda fails loudly rather than silently running on CPU."""
    monkeypatch.setenv("FF_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device cuda"):
        cuda_enabled()
