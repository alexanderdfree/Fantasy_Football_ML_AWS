"""Unit tests for the opt-in, sm_80+-gated ``torch.compile`` in _maybe_compile.

Guard the contract that keeps T4 production and CPU/Mac/CI unchanged: compile is
a no-op unless ``FF_COMPILE`` is truthy AND the GPU is sm_80+. The positive path
is verified by stubbing ``torch.compile`` so no real compilation runs in CI.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.shared.pipeline import _maybe_compile


@pytest.fixture
def model():
    return nn.Linear(3, 2)


@pytest.mark.unit
def test_noop_when_ff_compile_unset(monkeypatch, model):
    """Default (no FF_COMPILE): byte-identical to today — same object back."""
    monkeypatch.delenv("FF_COMPILE", raising=False)
    assert _maybe_compile(model) is model


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
def test_noop_when_ff_compile_falsy(monkeypatch, model, flag):
    monkeypatch.setenv("FF_COMPILE", flag)
    assert _maybe_compile(model) is model


@pytest.mark.unit
def test_noop_when_cuda_disabled(monkeypatch, model):
    """FF_COMPILE=1 on a CPU/forced-CPU box is still a no-op."""
    monkeypatch.setenv("FF_COMPILE", "1")
    monkeypatch.setenv("FF_DEVICE", "cpu")  # forces cuda_enabled() False
    assert _maybe_compile(model) is model


@pytest.mark.unit
def test_noop_on_t4_sm75(monkeypatch, model):
    """T4 (sm_75) stays on the proven no-compile path even with FF_COMPILE=1."""
    monkeypatch.setenv("FF_COMPILE", "1")
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert _maybe_compile(model) is model


@pytest.mark.unit
@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (12, 0)])
def test_compiles_on_sm80plus_when_opted_in(monkeypatch, model, capability):
    """sm_80+ (Ampere/Ada/Blackwell) + FF_COMPILE=1 → torch.compile is invoked."""
    monkeypatch.setenv("FF_COMPILE", "1")
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    calls = {}

    def fake_compile(m, **kwargs):
        calls["model"] = m
        calls["kwargs"] = kwargs
        return "COMPILED"

    monkeypatch.setattr(torch, "compile", fake_compile)
    out = _maybe_compile(model)
    assert out == "COMPILED"
    assert calls["model"] is model
    assert calls["kwargs"].get("dynamic") is True
