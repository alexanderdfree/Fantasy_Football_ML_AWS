"""Smoke tests for src/tuning/ablate_rb_gate.py.

Cover the loadability + variant-application surface so a future RB config
refactor doesn't break the ablation CLI without warning. The actual pipeline
``run`` is not invoked here — that would be a ~10 min training run, which is
out of scope for unit tests.

Also exercises the contract between ``ablate_rb_gate.VARIANTS`` /
``print_summary`` and the in-container ``_run_rb_gate_ablation`` in
``src/batch/train.py`` (post-M5 they share the same VARIANTS dict).
"""

from __future__ import annotations

import pytest

from src.tuning import ablate_rb_gate

pytestmark = pytest.mark.unit


def test_module_imports_cleanly():
    assert hasattr(ablate_rb_gate, "VARIANTS")
    assert hasattr(ablate_rb_gate, "main")
    assert hasattr(ablate_rb_gate, "print_summary")
    assert hasattr(ablate_rb_gate, "run_variant")


def test_variants_dict_has_all_six_variants():
    """A/B/C/D/E/Bf — adding/removing a variant requires updating this test."""
    assert set(ablate_rb_gate.VARIANTS) == {"A", "B", "C", "D", "E", "Bf"}


def test_each_variant_returns_a_dict_with_gated_targets():
    """Apply every variant fn to a stub cfg; each must return a fresh dict
    with non-empty ``gated_targets`` (a copy-paste regression would yield a
    reference to the shared base config and let one variant mutate another).
    """
    base = {
        "head_losses": {
            "rushing_tds": "poisson_nll",
            "receiving_tds": "poisson_nll",
            "rushing_yards": "huber",
            "receiving_yards": "huber",
            "receptions": "huber",
            "fumbles_lost": "poisson_nll",
        },
        "gated_targets": ["receptions"],
        "loss_weights": {},
        "huber_deltas": {},
        "nn_head_hidden_overrides": {},
    }
    for variant, (_label, fn) in ablate_rb_gate.VARIANTS.items():
        out = fn(base)
        assert out is not base, f"Variant {variant} returned the base dict (no deepcopy)"
        assert isinstance(out, dict)
        assert out.get("gated_targets"), f"Variant {variant} returned empty gated_targets"
        # Sanity: variant identifier survives the deepcopy and the cfg is non-empty.
        assert "head_losses" in out


def test_variant_a_uses_huber_on_all_targets():
    """Variant A is the pre-PR-2 baseline: every head on Huber."""
    base = {
        "head_losses": {
            "rushing_tds": "poisson_nll",
            "receiving_tds": "poisson_nll",
        },
        "gated_targets": [],
        "loss_weights": {},
        "huber_deltas": {},
        "nn_head_hidden_overrides": {},
    }
    _label, fn = ablate_rb_gate.VARIANTS["A"]
    out = fn(base)
    assert all(loss == "huber" for loss in out["head_losses"].values())


def test_variant_d_uses_hurdle_poisson_on_tds():
    """Variant D switches both TD heads to hurdle_poisson."""
    base = {
        "head_losses": {
            "rushing_tds": "poisson_nll",
            "receiving_tds": "poisson_nll",
            "rushing_yards": "huber",
        },
        "gated_targets": ["receptions"],
        "loss_weights": {},
        "huber_deltas": {},
        "nn_head_hidden_overrides": {},
    }
    _label, fn = ablate_rb_gate.VARIANTS["D"]
    out = fn(base)
    assert out["head_losses"]["rushing_tds"] == "hurdle_poisson"
    assert out["head_losses"]["receiving_tds"] == "hurdle_poisson"
    # Yards heads untouched.
    assert out["head_losses"]["rushing_yards"] == "huber"


def test_print_summary_handles_empty_rows():
    """Edge case: empty variants list shouldn't crash print_summary."""
    # No assertion on output content — just that it doesn't raise.
    ablate_rb_gate.print_summary([])


def test_print_summary_runs_with_synthetic_rows(capsys):
    """Print summary on a 2-row fake input — checks the format path executes
    and ranks rows by ``count_target_mae_sum``.
    """
    rows = [
        {
            "variant": "B",
            "label": "B label",
            "seed": 42,
            "fp_mae": 7.20,
            "fp_rmse": 9.10,
            "rushing_tds_mae": 0.40,
            "receiving_tds_mae": 0.30,
            "fumbles_lost_mae": 0.10,
            "receptions_mae": 1.0,
            "rushing_yards_mae": 24.0,
            "receiving_yards_mae": 16.0,
            "count_target_mae_sum": 0.80,
            "gate_aucs": {},
        },
        {
            "variant": "C",
            "label": "C label",
            "seed": 42,
            "fp_mae": 7.18,
            "fp_rmse": 9.08,
            "rushing_tds_mae": 0.35,
            "receiving_tds_mae": 0.28,
            "fumbles_lost_mae": 0.09,
            "receptions_mae": 1.0,
            "rushing_yards_mae": 24.0,
            "receiving_yards_mae": 16.0,
            "count_target_mae_sum": 0.72,
            "gate_aucs": {},
        },
    ]
    ablate_rb_gate.print_summary(rows)
    out = capsys.readouterr().out
    # C should win on count_target_mae_sum (0.72 < 0.80).
    assert "Count-target decision: variant C wins" in out
