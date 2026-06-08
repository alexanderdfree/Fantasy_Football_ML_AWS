"""Unit tests for src/tuning/ablate_rb_gate.py.

Cover the loadability + variant-application surface so a future RB config
refactor doesn't break the ablation CLI without warning.  The actual pipeline
``run`` is not invoked here — that would be a ~10 min training run, which is
out of scope for unit tests.

Tests cover:
- module-level API surface (attributes present, importable without pipeline)
- variant cfg-mutators: deep-copy isolation + gated_targets non-empty
- per-variant semantics (A=huber, D=hurdle_poisson)
- decision-table helper (``print_summary``) on synthetic rows
- ``_results_to_rows`` aggregation from ``AblationResult`` objects
"""

from __future__ import annotations

import pytest

from src.tuning import ablate_rb_gate
from src.tuning.ablation_runner import AblationResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Module API surface
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    assert hasattr(ablate_rb_gate, "VARIANTS")
    assert hasattr(ablate_rb_gate, "main")
    assert hasattr(ablate_rb_gate, "print_summary")
    # run_variant is no longer exported (replaced by _execute_rb_gate_job);
    # check the new public surface instead.
    assert hasattr(ablate_rb_gate, "_build_jobs")
    assert hasattr(ablate_rb_gate, "_results_to_rows")


def test_variants_dict_has_all_six_variants():
    """A/B/C/D/E/Bf — adding/removing a variant requires updating this test."""
    assert set(ablate_rb_gate.VARIANTS) == {"A", "B", "C", "D", "E", "Bf"}


def test_variants_are_label_fn_tuples():
    """Each entry must be a (str label, callable) pair."""
    for key, entry in ablate_rb_gate.VARIANTS.items():
        assert isinstance(entry, tuple) and len(entry) == 2, (
            f"Variant {key}: expected (label, fn) tuple, got {type(entry)}"
        )
        label, fn = entry
        assert isinstance(label, str), f"Variant {key}: label must be str"
        assert callable(fn), f"Variant {key}: second element must be callable"


# ---------------------------------------------------------------------------
# Variant cfg-mutators: deepcopy isolation + gated_targets
# ---------------------------------------------------------------------------

_BASE_CFG = {
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


def test_each_variant_returns_a_dict_with_gated_targets():
    """Apply every variant fn to a stub cfg; each must return a fresh dict
    with non-empty ``gated_targets`` (a copy-paste regression would yield a
    reference to the shared base config and let one variant mutate another).
    """
    for variant, (_label, fn) in ablate_rb_gate.VARIANTS.items():
        out = fn(_BASE_CFG)
        assert out is not _BASE_CFG, f"Variant {variant} returned the base dict (no deepcopy)"
        assert isinstance(out, dict)
        assert out.get("gated_targets"), f"Variant {variant} returned empty gated_targets"
        assert "head_losses" in out


def test_variant_deepcopy_isolation():
    """Applying variant A then B on the same base must not cross-contaminate."""
    _, fn_a = ablate_rb_gate.VARIANTS["A"]
    _, fn_b = ablate_rb_gate.VARIANTS["B"]
    out_a = fn_a(_BASE_CFG)
    out_b = fn_b(_BASE_CFG)
    # A sets gated_targets=["rushing_tds","receiving_tds"]
    # B sets gated_targets=["receptions"]
    assert "rushing_tds" in out_a["gated_targets"]
    assert "rushing_tds" not in out_b["gated_targets"]
    # Mutating the A output must not affect the base or the B output.
    out_a["gated_targets"].append("__sentinel__")
    assert "__sentinel__" not in _BASE_CFG.get("gated_targets", [])
    assert "__sentinel__" not in out_b["gated_targets"]


# ---------------------------------------------------------------------------
# Per-variant semantics
# ---------------------------------------------------------------------------


def test_variant_a_uses_huber_on_all_targets():
    """Variant A is the pre-PR-2 baseline: every head on Huber."""
    _, fn = ablate_rb_gate.VARIANTS["A"]
    out = fn(_BASE_CFG)
    assert all(loss == "huber" for loss in out["head_losses"].values())


def test_variant_b_gates_only_receptions():
    """Variant B pins gated_targets to ['receptions'] regardless of base."""
    _, fn = ablate_rb_gate.VARIANTS["B"]
    out = fn(_BASE_CFG)
    assert out["gated_targets"] == ["receptions"]


def test_variant_c_gates_tds_and_receptions():
    """Variant C adds TD gates on top of the reception gate."""
    _, fn = ablate_rb_gate.VARIANTS["C"]
    out = fn(_BASE_CFG)
    gated = set(out["gated_targets"])
    assert "receptions" in gated
    assert "rushing_tds" in gated
    assert "receiving_tds" in gated


def test_variant_d_uses_hurdle_poisson_on_tds():
    """Variant D switches both TD heads to hurdle_poisson."""
    _, fn = ablate_rb_gate.VARIANTS["D"]
    out = fn(_BASE_CFG)
    assert out["head_losses"]["rushing_tds"] == "hurdle_poisson"
    assert out["head_losses"]["receiving_tds"] == "hurdle_poisson"
    # Yards heads untouched.
    assert out["head_losses"]["rushing_yards"] == "huber"


def test_variant_e_adds_fumbles_hurdle_poisson():
    """Variant E extends D by also setting fumbles_lost to hurdle_poisson."""
    _, fn = ablate_rb_gate.VARIANTS["E"]
    out = fn(_BASE_CFG)
    assert out["head_losses"]["rushing_tds"] == "hurdle_poisson"
    assert out["head_losses"]["receiving_tds"] == "hurdle_poisson"
    assert out["head_losses"]["fumbles_lost"] == "hurdle_poisson"
    assert "fumbles_lost" in out["gated_targets"]


def test_variant_bf_gates_fumbles_but_keeps_poisson_nll():
    """Variant Bf adds fumbles_lost to gated_targets but leaves head_losses unchanged."""
    _, fn = ablate_rb_gate.VARIANTS["Bf"]
    out = fn(_BASE_CFG)
    # fumbles_lost is now gated
    assert "fumbles_lost" in out["gated_targets"]
    # head_losses is a deepcopy of base — fumbles_lost stays poisson_nll
    assert out["head_losses"]["fumbles_lost"] == "poisson_nll"


# ---------------------------------------------------------------------------
# print_summary: edge cases + decision-rule output
# ---------------------------------------------------------------------------


def test_print_summary_handles_empty_rows():
    """Edge case: empty variants list shouldn't crash print_summary."""
    ablate_rb_gate.print_summary([])


def test_print_summary_runs_with_synthetic_rows_single_seed(capsys):
    """Print summary on a 2-row fake input (single-seed path) — checks that
    the format path executes and ranks rows by ``count_target_mae_sum``."""
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


def test_print_summary_abc_decision_keep_gate(capsys):
    """When A and C both beat B by >= 0.05, the original rule says keep gate."""
    rows = [
        {
            "variant": "A",
            "label": "A label",
            "seed": 42,
            "fp_mae": 7.10,
            "fp_rmse": 9.0,
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
            "variant": "B",
            "label": "B label",
            "seed": 42,
            "fp_mae": 7.30,  # B is worse than A and C by >0.05
            "fp_rmse": 9.2,
            "rushing_tds_mae": 0.42,
            "receiving_tds_mae": 0.32,
            "fumbles_lost_mae": 0.11,
            "receptions_mae": 1.0,
            "rushing_yards_mae": 24.0,
            "receiving_yards_mae": 16.0,
            "count_target_mae_sum": 0.85,
            "gate_aucs": {},
        },
        {
            "variant": "C",
            "label": "C label",
            "seed": 42,
            "fp_mae": 7.12,
            "fp_rmse": 9.0,
            "rushing_tds_mae": 0.38,
            "receiving_tds_mae": 0.29,
            "fumbles_lost_mae": 0.09,
            "receptions_mae": 1.0,
            "rushing_yards_mae": 24.0,
            "receiving_yards_mae": 16.0,
            "count_target_mae_sum": 0.76,
            "gate_aucs": {},
        },
    ]
    ablate_rb_gate.print_summary(rows)
    out = capsys.readouterr().out
    assert "keep gate on TDs" in out


def test_print_summary_abc_decision_drop_gate(capsys):
    """When neither A nor C beats B by >= 0.05, the original rule says drop gate."""
    rows = [
        {
            "variant": "A",
            "label": "A",
            "seed": 42,
            "fp_mae": 7.22,
            "fp_rmse": 9.0,
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
            "variant": "B",
            "label": "B",
            "seed": 42,
            "fp_mae": 7.24,  # margin < 0.05
            "fp_rmse": 9.1,
            "rushing_tds_mae": 0.41,
            "receiving_tds_mae": 0.31,
            "fumbles_lost_mae": 0.10,
            "receptions_mae": 1.0,
            "rushing_yards_mae": 24.0,
            "receiving_yards_mae": 16.0,
            "count_target_mae_sum": 0.82,
            "gate_aucs": {},
        },
        {
            "variant": "C",
            "label": "C",
            "seed": 42,
            "fp_mae": 7.21,
            "fp_rmse": 9.0,
            "rushing_tds_mae": 0.39,
            "receiving_tds_mae": 0.30,
            "fumbles_lost_mae": 0.10,
            "receptions_mae": 1.0,
            "rushing_yards_mae": 24.0,
            "receiving_yards_mae": 16.0,
            "count_target_mae_sum": 0.79,
            "gate_aucs": {},
        },
    ]
    ablate_rb_gate.print_summary(rows)
    out = capsys.readouterr().out
    assert "drop gate on TDs" in out


# ---------------------------------------------------------------------------
# _results_to_rows: AblationResult → legacy row dict aggregation
# ---------------------------------------------------------------------------


def _make_result(variant: str, seed: int, fp_mae: float, cnt_sum: float) -> AblationResult:
    label = ablate_rb_gate.VARIANTS[variant][0]
    return AblationResult(
        position="RB",
        seed=seed,
        variant=variant,
        metrics={
            "fp_mae": fp_mae,
            "fp_rmse": fp_mae + 1.0,
            "rushing_tds_mae": 0.35,
            "receiving_tds_mae": 0.28,
            "fumbles_lost_mae": cnt_sum - 0.63,
            "receptions_mae": 1.0,
            "rushing_yards_mae": 24.0,
            "receiving_yards_mae": 16.0,
            "count_target_mae_sum": cnt_sum,
            "gate_aucs": {},
        },
        timings={},
        metadata={"run_kind": "experiment", "variant_label": label},
        error=None,
    )


def test_results_to_rows_single_seed():
    """Single-seed: rows preserve exact metric values, no _stats suffix."""
    results = [
        _make_result("B", 42, 7.20, 0.80),
        _make_result("C", 42, 7.18, 0.72),
    ]
    rows = ablate_rb_gate._results_to_rows(results)
    assert len(rows) == 2
    by_var = {r["variant"]: r for r in rows}
    assert abs(by_var["B"]["fp_mae"] - 7.20) < 1e-9
    assert abs(by_var["C"]["fp_mae"] - 7.18) < 1e-9
    # Single-seed: no _stats keys expected
    assert "fp_mae_stats" not in by_var["B"]


def test_results_to_rows_multi_seed_averages():
    """Multi-seed: numeric metrics are averaged across seeds."""
    results = [
        _make_result("C", 42, 7.18, 0.72),
        _make_result("C", 43, 7.22, 0.76),
    ]
    rows = ablate_rb_gate._results_to_rows(results)
    assert len(rows) == 1
    row = rows[0]
    assert row["variant"] == "C"
    assert abs(row["fp_mae"] - 7.20) < 1e-9  # (7.18+7.22)/2
    assert "fp_mae_stats" in row
    stats = row["fp_mae_stats"]
    assert abs(stats["mean"] - 7.20) < 1e-9
    assert stats["n"] == 2


def test_results_to_rows_skips_errors():
    """Error results must be excluded from the aggregation."""
    good = _make_result("B", 42, 7.20, 0.80)
    bad = AblationResult(
        position="RB",
        seed=43,
        variant="B",
        metrics={},
        timings={},
        metadata={},
        error="RuntimeError: something failed",
    )
    rows = ablate_rb_gate._results_to_rows([good, bad])
    assert len(rows) == 1
    assert abs(rows[0]["fp_mae"] - 7.20) < 1e-9


def test_results_to_rows_variant_order():
    """Rows must follow _VARIANT_ORDER (A,B,C,D,E,Bf) regardless of input order."""
    results = [
        _make_result("E", 42, 7.15, 0.70),
        _make_result("A", 42, 7.25, 0.85),
        _make_result("C", 42, 7.18, 0.72),
    ]
    rows = ablate_rb_gate._results_to_rows(results)
    assert [r["variant"] for r in rows] == ["A", "C", "E"]


# ---------------------------------------------------------------------------
# _build_jobs: job construction smoke test (no pipeline run)
# ---------------------------------------------------------------------------


def test_build_jobs_count():
    """_build_jobs produces exactly positions × seeds × variants jobs."""
    from unittest.mock import patch

    stub_cfg = dict(_BASE_CFG)
    with patch("src.tuning.ablate_rb_gate.get_config", return_value=stub_cfg):
        jobs = ablate_rb_gate._build_jobs(
            positions=["RB"],
            seeds=[42, 43],
            variants=["A", "B"],
        )
    assert len(jobs) == 4  # 1 pos × 2 seeds × 2 variants
    variants_seen = {j.variant for j in jobs}
    assert variants_seen == {"A", "B"}
    seeds_seen = {j.seed for j in jobs}
    assert seeds_seen == {42, 43}


def test_build_jobs_run_fn_is_callable():
    """Every job must carry a callable run_fn."""
    from unittest.mock import patch

    stub_cfg = dict(_BASE_CFG)
    with patch("src.tuning.ablate_rb_gate.get_config", return_value=stub_cfg):
        jobs = ablate_rb_gate._build_jobs(positions=["RB"], seeds=[42], variants=["C"])
    assert all(callable(j.run_fn) for j in jobs)
