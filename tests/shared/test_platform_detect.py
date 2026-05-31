"""Unit tests for src.shared.platform_detect.

``detect_platform()`` is the canonical hardware-capability report new
platform-specific optimizations branch off (see CLAUDE.md's *Platform & hardware
targets*). These guard the per-arch fields the matrix depends on: backend
selection (CUDA > MPS > CPU), the ``sm_xx`` / wheel mapping, native-BF16 gating
at the Ampere (sm_80) boundary, and WSL2 vs native-Windows detection.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch

from src.shared import platform_detect
from src.shared.platform_detect import PlatformInfo, detect_platform


def _force_cuda(monkeypatch, capability, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: name)


def _force_no_cuda(monkeypatch, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def _force_os(monkeypatch, system, release="", machine="x86_64"):
    monkeypatch.setattr(platform_detect.platform, "system", lambda: system)
    monkeypatch.setattr(platform_detect.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        platform_detect.platform,
        "uname",
        lambda: types.SimpleNamespace(release=release),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "capability,name,sm,bf16,wheel",
    [
        ((7, 5), "Tesla T4", "sm_75", False, "cu126"),  # g4dn — no native BF16
        ((8, 9), "NVIDIA L4", "sm_89", True, "cu126"),  # g6 — BF16-capable
        ((12, 0), "NVIDIA GeForce RTX 5080", "sm_120", True, "cu128"),  # Blackwell
    ],
)
def test_cuda_backend_fields(monkeypatch, capability, name, sm, bf16, wheel):
    _force_os(monkeypatch, "Linux")
    _force_cuda(monkeypatch, capability, name)
    info = detect_platform()
    assert info.backend == "cuda"
    assert info.gpu_name == name
    assert info.compute_capability == capability
    assert info.sm == sm
    assert info.supports_bf16 is bf16
    assert info.recommended_cuda_wheel == wheel


@pytest.mark.unit
def test_mps_backend(monkeypatch):
    """No CUDA but MPS available → MPS backend, no CUDA-only fields."""
    _force_os(monkeypatch, "Darwin", machine="arm64")
    _force_no_cuda(monkeypatch, mps=True)
    info = detect_platform()
    assert info.backend == "mps"
    assert info.gpu_name == "Apple MPS"
    assert info.os == "macOS"
    assert info.arch == "arm64"
    assert info.compute_capability is None
    assert info.sm is None
    assert info.supports_bf16 is False
    assert info.recommended_cuda_wheel is None


@pytest.mark.unit
def test_cpu_backend(monkeypatch):
    """No CUDA and no MPS → plain CPU, every accelerator field cleared."""
    _force_os(monkeypatch, "Linux")
    _force_no_cuda(monkeypatch, mps=False)
    info = detect_platform()
    assert info.backend == "cpu"
    assert info.gpu_name is None
    assert info.compute_capability is None
    assert info.sm is None
    assert info.supports_bf16 is False
    assert info.recommended_cuda_wheel is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "system,release,expected_os,expected_wsl",
    [
        ("Linux", "5.15.0-microsoft-standard-WSL2", "Linux", True),
        ("Linux", "6.8.0-aws", "Linux", False),  # native Linux (AWS)
        ("Windows", "10.0.22631", "Windows", False),  # native Windows, never WSL
        ("Darwin", "23.5.0", "macOS", False),
    ],
)
def test_os_and_wsl_detection(monkeypatch, system, release, expected_os, expected_wsl):
    _force_os(monkeypatch, system, release=release)
    _force_no_cuda(monkeypatch, mps=False)
    info = detect_platform()
    assert info.os == expected_os
    assert info.is_wsl is expected_wsl


@pytest.mark.unit
def test_summary_is_descriptive(monkeypatch):
    _force_os(monkeypatch, "Linux", release="5.15.0-microsoft-standard-WSL2")
    _force_cuda(monkeypatch, (12, 0), "NVIDIA GeForce RTX 5080")
    summary = detect_platform().summary()
    assert "CUDA" in summary
    assert "sm_120" in summary
    assert "WSL2" in summary


@pytest.mark.unit
def test_platform_info_is_frozen():
    """The matrix relies on a stable snapshot — PlatformInfo must be immutable."""
    info = detect_platform()
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.backend = "cuda"  # type: ignore[misc]


@pytest.mark.unit
def test_detect_platform_on_real_host_is_self_consistent():
    """Unmocked smoke test: whatever the CI/dev host is, the invariants hold."""
    info = detect_platform()
    assert isinstance(info, PlatformInfo)
    assert info.backend in {"cuda", "mps", "cpu"}
    if info.backend == "cuda":
        assert info.sm is not None and info.sm.startswith("sm_")
        assert info.recommended_cuda_wheel in {"cu126", "cu128"}
    else:
        # CUDA-only fields are cleared off-CUDA.
        assert info.compute_capability is None
        assert info.sm is None
        assert info.recommended_cuda_wheel is None
