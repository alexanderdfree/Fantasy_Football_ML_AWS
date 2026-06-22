"""Unit tests for the ``FF_DEVICE`` / ``--device`` resolver in src.shared.utils.

These guard the contract ``run_pipeline --device`` relies on: ``auto`` reproduces
the historical ``torch.cuda.is_available()`` behaviour (so Linux/macOS/CI are
unchanged when the flag is omitted), ``cpu`` forces the CPU path even on a
CUDA-visible host, ``cuda`` fails loudly when no GPU is visible rather than
silently degrading to CPU, and ``mps`` is an opt-in that ``auto`` never selects
(so the default path stays CUDA-or-CPU and byte-identical to CI).
"""

from __future__ import annotations

import pytest
import torch

from src.shared.utils import (
    amp_dtype,
    cuda_enabled,
    mps_enabled,
    requested_amp_dtype,
    requested_device,
    seed_everything,
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
        ("mps", "mps"),
        ("MPS", "mps"),  # case-insensitive
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


@pytest.mark.unit
def test_seed_everything_cpu_device_does_not_touch_cuda(monkeypatch):
    """CPU-forced diagnostics must be able to fork without inheriting CUDA state."""
    monkeypatch.setenv("FF_DEVICE", "cpu")
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("FF_DEVICE=cpu should short-circuit the CUDA probe"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: pytest.fail("FF_DEVICE=cpu should not seed CUDA"),
    )

    seed_everything(123)


@pytest.mark.unit
def test_seed_everything_auto_cuda_still_seeds_cuda(monkeypatch):
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))

    seed_everything(456)

    assert calls == [456]


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
def test_amp_dtype_auto_is_off_on_all_cuda(monkeypatch, capability):
    """Default (auto): AMP off -> FP32+TF32 on every CUDA device.

    Flipped 2026-06-22 (owner-approved metric-path change): the default no longer
    autocasts to FP16 — the NN trains in FP32 storage with TF32 matmuls. FP16
    remains available via FF_AMP_DTYPE=fp16 (see test_amp_dtype_simple_overrides).
    """
    monkeypatch.delenv("FF_AMP_DTYPE", raising=False)
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert amp_dtype() is None


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


@pytest.mark.unit
def test_cuda_enabled_mps_request_is_not_cuda(monkeypatch):
    """--device mps must keep CUDA off even on a CUDA-visible box, so the NN's
    device and its batcher path can't both fire."""
    monkeypatch.setenv("FF_DEVICE", "mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cuda_enabled() is False


@pytest.mark.unit
@pytest.mark.parametrize("req", ["auto", "cpu", "cuda"])
def test_mps_enabled_false_unless_explicitly_requested(monkeypatch, req):
    """MPS is opt-in: auto never selects it even when MPS is available."""
    monkeypatch.setenv("FF_DEVICE", req)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert mps_enabled() is False


@pytest.mark.unit
def test_mps_enabled_auto_default_ignores_mps(monkeypatch):
    """Unset FF_DEVICE (auto) keeps the default path CUDA-or-CPU, never MPS."""
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert mps_enabled() is False


@pytest.mark.unit
def test_mps_enabled_uses_mps_when_available(monkeypatch):
    monkeypatch.setenv("FF_DEVICE", "mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert mps_enabled() is True


@pytest.mark.unit
def test_mps_enabled_raises_when_unavailable(monkeypatch):
    """--device mps fails loudly rather than silently running on CPU."""
    monkeypatch.setenv("FF_DEVICE", "mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device mps"):
        mps_enabled()
