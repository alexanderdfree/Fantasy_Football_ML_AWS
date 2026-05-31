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

from src.shared.utils import (
    amp_dtype,
    cuda_enabled,
    requested_amp_dtype,
    requested_device,
)


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


# --- amp_dtype: capability-gated mixed precision ----------------------------
# These guard the contract that keeps existing hosts unchanged: T4 (sm_75) stays
# on FP16, Mac/CPU stays AMP-off, and only sm_80+ (RTX 5080) gets BF16.


@pytest.mark.unit
def test_requested_amp_dtype_defaults_and_normalises(monkeypatch):
    monkeypatch.delenv("FF_AMP_DTYPE", raising=False)
    assert requested_amp_dtype() == "auto"
    for raw, expected in [
        ("bf16", "bf16"),
        ("FP16", "fp16"),  # case-insensitive
        ("  fp32 ", "fp32"),  # whitespace tolerated
        ("float16", "auto"),  # unrecognised → safe auto fallback
        ("", "auto"),
    ]:
        monkeypatch.setenv("FF_AMP_DTYPE", raw)
        assert requested_amp_dtype() == expected


@pytest.mark.unit
def test_amp_dtype_none_off_cuda(monkeypatch):
    """CPU/MPS (local Mac, CI): AMP off so runs stay byte-identical to FP32."""
    monkeypatch.delenv("FF_AMP_DTYPE", raising=False)
    monkeypatch.setenv("FF_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert amp_dtype() is None


@pytest.mark.unit
@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 9), (12, 0)])
def test_amp_dtype_auto_is_fp16_on_all_cuda(monkeypatch, capability):
    """Default (auto): FP16 on every CUDA device — T4 and Blackwell alike.

    BF16 is never auto-selected; a 5080 A/B showed it regresses high-magnitude
    heads, so FP16 is the proven default regardless of compute capability.
    """
    monkeypatch.delenv("FF_AMP_DTYPE", raising=False)
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert amp_dtype() is torch.float16


@pytest.mark.unit
@pytest.mark.parametrize(
    "override,expected",
    [
        ("fp16", torch.float16),
        ("fp32", None),  # explicit fp32 disables AMP even on a capable GPU
    ],
)
def test_amp_dtype_simple_overrides(monkeypatch, override, expected):
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setenv("FF_AMP_DTYPE", override)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (12, 0))
    assert amp_dtype() is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "capability,expected",
    [
        ((12, 0), torch.bfloat16),  # Blackwell — opt-in honoured
        ((8, 0), torch.bfloat16),  # Ampere — opt-in honoured
        ((7, 5), torch.float16),  # T4 — REFUSED, falls back to FP16 (no #293 hang)
    ],
)
def test_amp_dtype_bf16_optin_is_sm80_gated(monkeypatch, capability, expected):
    """FF_AMP_DTYPE=bf16 engages BF16 only on sm_80+; on T4 it degrades to FP16
    so the opt-in cannot reintroduce the PR #293/#301 T4 hang."""
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setenv("FF_AMP_DTYPE", "bf16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert amp_dtype() is expected
