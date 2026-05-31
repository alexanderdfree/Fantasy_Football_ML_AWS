"""Smoke tests for src/tuning/ablate_backbone_norm.py.

Cover the loadability + the LayerNorm-backbone swap surface so a future
refactor of the shared backbone (``src/shared/neural_net.py::_build_backbone``,
the monkeypatch target) doesn't break the ablation CLI without warning. The
actual pipeline ``run`` is not invoked here — that is a multi-minute training
run, out of scope for unit tests.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

import src.shared.neural_net as nn_mod
from src.tuning import ablate_backbone_norm

pytestmark = pytest.mark.unit


def test_module_imports_cleanly():
    for attr in ("VARIANTS", "main", "print_summary", "run_variant", "_extract"):
        assert hasattr(ablate_backbone_norm, attr)


def test_variants_dict_is_bn_and_ln():
    """Exactly two variants — adding/removing one requires updating this test."""
    assert set(ablate_backbone_norm.VARIANTS) == {"bn", "ln"}


def test_layernorm_backbone_swaps_only_the_norm():
    """The LN variant must mirror the stock backbone structurally but use
    LayerNorm where the stock uses BatchNorm1d — that one swap is the entire
    ablation, so guard it directly."""
    stock = list(nn_mod._build_backbone(10, [8, 4], 0.1))
    ln = list(ablate_backbone_norm._build_backbone_layernorm(10, [8, 4], 0.1))

    # Same module count (only the norm type differs).
    assert len(ln) == len(stock)
    # Stock uses BatchNorm; LN variant uses LayerNorm and no BatchNorm.
    assert any(isinstance(m, nn.BatchNorm1d) for m in stock)
    assert any(isinstance(m, nn.LayerNorm) for m in ln)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in ln)
    # Per-block order preserved: Linear -> LayerNorm -> ReLU -> Dropout.
    assert isinstance(ln[0], nn.Linear)
    assert isinstance(ln[1], nn.LayerNorm)
    assert isinstance(ln[2], nn.ReLU)
    assert isinstance(ln[3], nn.Dropout)
    # LayerNorm normalizes over the hidden width.
    assert ln[1].normalized_shape == (8,)


def test_monkeypatch_target_exists_and_is_callable():
    """run_variant('ln') patches ``nn_mod._build_backbone`` — fail loudly here
    if that attribute ever moves or is renamed."""
    assert callable(nn_mod._build_backbone)


def test_print_summary_handles_empty_rows():
    """Edge case: empty rows must not raise (no mean-of-empty crash)."""
    ablate_backbone_norm.print_summary([], ["receiving_tds"])


def _row(variant: str, seed: int, attn: float, base: float, ridge: float) -> dict:
    return {
        "variant": variant,
        "seed": seed,
        "attn_fp_mae": attn,
        "base_fp_mae": base,
        "ridge_fp_mae": ridge,
        "attn_targets": {"receiving_tds": 0.29, "receiving_yards": 20.2},
        "base_targets": {"receiving_tds": 0.30, "receiving_yards": 20.5},
    }


def test_print_summary_sentinel_passes_when_ridge_matches(capsys):
    targets = ["receiving_tds", "receiving_yards"]
    rows = [_row("bn", 42, 4.21, 4.15, 4.72), _row("ln", 42, 4.20, 4.14, 4.72)]
    ok = ablate_backbone_norm.print_summary(rows, targets)
    out = capsys.readouterr().out
    assert ok is True  # identical Ridge MAE → clean single-variable comparison
    assert "VERDICT" in out


def test_print_summary_sentinel_fails_when_ridge_differs(capsys):
    """A Ridge-MAE mismatch means the variants saw different data/seed — the
    sentinel must flag it (False) rather than report a bogus NN delta."""
    targets = ["receiving_tds", "receiving_yards"]
    rows = [_row("bn", 42, 4.21, 4.15, 4.72), _row("ln", 42, 4.20, 4.14, 4.99)]
    ok = ablate_backbone_norm.print_summary(rows, targets)
    out = capsys.readouterr().out
    assert ok is False
    assert "MISMATCH" in out
