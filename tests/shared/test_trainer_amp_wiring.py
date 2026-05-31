"""Unit tests for ``MultiHeadTrainer``'s AMP wiring (dtype selection + GradScaler).

``tests/shared/test_device_selection.py`` covers the ``amp_dtype()`` *helper*;
these cover how the trainer *consumes* it — the `__init__` lines that pick
``self._amp_dtype`` and gate ``GradScaler(enabled=...)``. The per-position
trainer tests all run on ``torch.device("cpu")``, which short-circuits AMP, so
without this file the CUDA fp16-vs-bf16 branch is never exercised. We mock CUDA
availability + compute capability so the test runs on a CPU-only CI host.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from src.shared.training import MultiHeadTrainer


def _make_trainer(device):
    """Construct a minimal trainer (init only stores args + does AMP wiring)."""
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings():
        # GradScaler("cuda", enabled=True) warns if it thinks CUDA is absent;
        # the is_available mock prevents that, but stay quiet regardless.
        warnings.simplefilter("ignore")
        return MultiHeadTrainer(model, opt, None, None, device, ["y"], use_amp=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "ff_amp,capability,expected_dtype",
    [
        (None, (12, 0), torch.float16),  # auto on Blackwell (5080) → FP16 default
        (None, (8, 9), torch.float16),  # auto on Ada (L4 prod host) → FP16 default
        (None, (7, 5), torch.float16),  # auto on Turing (T4) → FP16
        ("bf16", (12, 0), torch.bfloat16),  # opt-in BF16 honoured on sm_80+
        ("bf16", (8, 9), torch.bfloat16),  # opt-in BF16 honoured on Ada (L4)
        ("bf16", (7, 5), torch.float16),  # opt-in BF16 REFUSED on T4 → FP16
        ("fp16", (12, 0), torch.float16),  # explicit fp16
        ("fp32", (12, 0), None),  # fp32 → AMP off
    ],
)
def test_trainer_selects_amp_dtype_on_cuda(monkeypatch, ff_amp, capability, expected_dtype):
    monkeypatch.delenv("FF_DEVICE", raising=False)
    if ff_amp is None:
        monkeypatch.delenv("FF_AMP_DTYPE", raising=False)
    else:
        monkeypatch.setenv("FF_AMP_DTYPE", ff_amp)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)

    tr = _make_trainer(torch.device("cuda"))

    assert tr._amp_dtype is expected_dtype
    assert tr._use_amp is (expected_dtype is not None)
    # GradScaler is enabled ONLY for the FP16 path; BF16 keeps the FP32 exponent
    # range (no scaling) and fp32/None is AMP-off — both must leave it disabled.
    assert tr._scaler.is_enabled() is (expected_dtype is torch.float16)


@pytest.mark.unit
def test_trainer_amp_off_on_cpu(monkeypatch):
    """CPU device → AMP off regardless of FF_AMP_DTYPE or a CUDA-available host.

    The ``device.type == "cuda"`` guard short-circuits before ``amp_dtype()``,
    so CPU/MPS dev + CI runs stay byte-identical to the FP32 path.
    """
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setenv("FF_AMP_DTYPE", "bf16")  # even an explicit opt-in is ignored on CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    tr = _make_trainer(torch.device("cpu"))

    assert tr._amp_dtype is None
    assert tr._use_amp is False
    assert tr._scaler.is_enabled() is False
