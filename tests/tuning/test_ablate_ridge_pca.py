"""Unit tests for src.tuning.ablate_ridge_pca (parallel-runner shape).

Tests the PCA cfg-mutator, the decision/comparison table helpers, and the
module's public API on synthetic rows.  No real pipeline is invoked — all
tests use synthetic AblationResult fixtures or inspect config mutations.

pytest.mark.unit — runs in the fast unit shard (no data/splits needed).
"""

from __future__ import annotations

import pytest

from src.tuning import ablate_ridge_pca as abl
from src.tuning.ablation_runner import AblationResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_cfg(**updates):
    """Minimal cfg dict mirroring what get_config() returns for QB."""
    cfg = {
        "targets": ["passing_yards", "passing_tds"],
        "train_ridge": True,
        "train_base_nn": True,
        "train_attention_nn": True,
        "train_elasticnet": True,
        "train_lightgbm": True,
        "ridge_alpha_grids": {"passing_yards": [0.1, 1.0, 10.0]},
        "ridge_cv_folds": 5,
        "ridge_refine_points": 3,
    }
    cfg.update(updates)
    return cfg


def _result(
    position,
    seed,
    variant,
    val_mae,
    test_mae,
    *,
    run_kind="experiment",
    error=None,
):
    return AblationResult(
        position=position,
        seed=seed,
        variant=variant,
        metrics={"val_mae": val_mae, "test_mae": test_mae, "pca_n": abl.VARIANTS[variant]},
        timings={"ridge_train_sec": None},
        metadata={"run_kind": run_kind, "pca_n": abl.VARIANTS[variant]},
        error=error,
    )


# ---------------------------------------------------------------------------
# Module-level API smoke
# ---------------------------------------------------------------------------


def test_module_and_public_api_importable():
    """Guard against import/signature drift."""
    for name in (
        "VARIANTS",
        "BASELINE_VARIANT",
        "DEFAULT_POSITIONS",
        "DEFAULT_SEEDS",
        "DEFAULT_VARIANTS",
        "_make_cfg",
        "_execute_ridge_pca_job",
        "_build_jobs",
        "summarize_results",
        "print_summary",
        "main",
    ):
        assert hasattr(abl, name), f"missing {name}"


def test_baseline_variant_is_none_pca():
    """'none' variant must map to pca_n=None (the no-PCA baseline)."""
    assert abl.BASELINE_VARIANT == "none"
    assert abl.VARIANTS[abl.BASELINE_VARIANT] is None


def test_default_variants_all_present_in_variants_dict():
    for key in abl.DEFAULT_VARIANTS:
        assert key in abl.VARIANTS, f"DEFAULT_VARIANTS key {key!r} missing from VARIANTS"


def test_default_variants_baseline_is_first():
    assert abl.DEFAULT_VARIANTS[0] == abl.BASELINE_VARIANT


# ---------------------------------------------------------------------------
# _make_cfg — PCA cfg mutator
# ---------------------------------------------------------------------------


def test_make_cfg_baseline_sets_pca_none_and_disables_nonridge():
    cfg = abl._make_cfg(_base_cfg(), pca_n=None)
    assert cfg["ridge_pca_components"] is None
    assert cfg["train_ridge"] is True
    assert cfg["train_base_nn"] is False
    assert cfg["train_attention_nn"] is False
    assert cfg["train_lightgbm"] is False
    assert cfg["train_elasticnet"] is False


def test_make_cfg_sets_pca_n_for_pca_variants():
    cfg = abl._make_cfg(_base_cfg(), pca_n=55)
    assert cfg["ridge_pca_components"] == 55
    assert cfg["train_ridge"] is True
    assert cfg["train_base_nn"] is False
    assert cfg["train_attention_nn"] is False
    assert cfg["train_lightgbm"] is False
    assert cfg["train_elasticnet"] is False


def test_make_cfg_deep_copies_base():
    base = _base_cfg(ridge_alpha_grids={"passing_yards": [1.0, 10.0]})
    cfg = abl._make_cfg(base, pca_n=40)
    # Mutating the returned cfg must not affect base.
    cfg["ridge_alpha_grids"]["passing_yards"].append(999.0)
    assert 999.0 not in base["ridge_alpha_grids"]["passing_yards"]


def test_make_cfg_all_pca_variant_values():
    """Every entry in VARIANTS maps to the expected int or None."""
    for _key, pca_n in abl.VARIANTS.items():
        cfg = abl._make_cfg(_base_cfg(), pca_n=pca_n)
        assert cfg["ridge_pca_components"] == pca_n


# ---------------------------------------------------------------------------
# _build_jobs
# ---------------------------------------------------------------------------


def test_build_jobs_count_and_structure(monkeypatch):
    monkeypatch.setattr(abl, "get_config", lambda pos: _base_cfg())
    jobs = abl._build_jobs(
        positions=["QB", "TE"],
        seeds=[42, 43],
        variants=["none", "pca55", "pca40"],
    )
    # 2 positions × 2 seeds × 3 variants = 12 jobs
    assert len(jobs) == 12
    assert all(j.run_fn is abl._execute_ridge_pca_job for j in jobs)
    assert all(j.metadata["run_kind"] == "experiment" for j in jobs)


def test_build_jobs_injects_pca_n_sentinel(monkeypatch):
    monkeypatch.setattr(abl, "get_config", lambda pos: _base_cfg())
    jobs = abl._build_jobs(
        positions=["QB"],
        seeds=[42],
        variants=["none", "pca80"],
    )
    pca_values = {j.variant: j.base_cfg["_job_pca_n"] for j in jobs}
    assert pca_values["none"] is None
    assert pca_values["pca80"] == 80


def test_build_jobs_does_not_mutate_base_cfg(monkeypatch):
    base = _base_cfg()
    monkeypatch.setattr(abl, "get_config", lambda pos: base)
    abl._build_jobs(positions=["QB"], seeds=[42], variants=["none", "pca55"])
    # _job_pca_n must not leak back into the original cfg dict.
    assert "_job_pca_n" not in base


# ---------------------------------------------------------------------------
# summarize_results — decision table
# ---------------------------------------------------------------------------


def test_summarize_results_consistent_improver_beats_both_seasons():
    """pca55 beats None on BOTH val and test -> consistent improver."""
    results = [
        _result("QB", 42, "none", val_mae=5.00, test_mae=6.00),
        _result("QB", 42, "pca55", val_mae=4.90, test_mae=5.90),
        _result("QB", 42, "pca40", val_mae=4.80, test_mae=6.10),  # test worse
    ]
    summary = abl.summarize_results(results, variants=["none", "pca55", "pca40"])
    assert summary["QB"]["variants"]["pca55"]["consistent_improver"] is True
    assert summary["QB"]["variants"]["pca40"]["consistent_improver"] is False
    assert "pca_n=55" in summary["QB"]["recommendation"]
    assert summary["QB"]["consistent_improvers"] == ["pca55"]


def test_summarize_results_no_consistent_improver():
    """No pca_n beats None on BOTH seasons -> recommendation says not worth shipping."""
    results = [
        _result("QB", 42, "none", val_mae=5.00, test_mae=6.00),
        _result("QB", 42, "pca80", val_mae=4.90, test_mae=6.10),  # test worse
        _result("QB", 42, "pca60", val_mae=5.10, test_mae=5.90),  # val worse
    ]
    summary = abl.summarize_results(results, variants=["none", "pca80", "pca60"])
    assert summary["QB"]["consistent_improvers"] == []
    assert "NO pca_n" in summary["QB"]["recommendation"]


def test_summarize_results_best_by_test_mae():
    """When multiple consistent improvers exist, the best by test MAE is highlighted."""
    results = [
        _result("QB", 42, "none", val_mae=5.00, test_mae=6.00),
        _result("QB", 42, "pca80", val_mae=4.95, test_mae=5.95),
        _result("QB", 42, "pca55", val_mae=4.90, test_mae=5.85),  # better test
    ]
    summary = abl.summarize_results(results, variants=["none", "pca80", "pca55"])
    assert set(summary["QB"]["consistent_improvers"]) == {"pca80", "pca55"}
    assert "pca_n=55" in summary["QB"]["recommendation"]


def test_summarize_results_delta_vs_baseline_sign():
    """Deltas should be improvement direction (val/test - baseline)."""
    results = [
        _result("QB", 42, "none", val_mae=5.00, test_mae=6.00),
        _result("QB", 42, "pca55", val_mae=4.80, test_mae=5.70),
    ]
    summary = abl.summarize_results(results, variants=["none", "pca55"])
    rec = summary["QB"]["variants"]["pca55"]
    assert rec["val_delta_vs_baseline"]["mean"] == pytest.approx(-0.20)
    assert rec["test_delta_vs_baseline"]["mean"] == pytest.approx(-0.30)


def test_summarize_results_multi_seed_mean():
    """Multi-seed: summary uses mean across seeds, std≈0 for deterministic Ridge."""
    results = [
        _result("QB", 42, "none", val_mae=5.00, test_mae=6.00),
        _result("QB", 43, "none", val_mae=5.00, test_mae=6.00),
        _result("QB", 42, "pca55", val_mae=4.80, test_mae=5.80),
        _result("QB", 43, "pca55", val_mae=4.80, test_mae=5.80),
    ]
    summary = abl.summarize_results(results, variants=["none", "pca55"])
    assert summary["QB"]["variants"]["pca55"]["val_mae"]["mean"] == pytest.approx(4.80)
    assert summary["QB"]["variants"]["pca55"]["val_mae"]["std"] == pytest.approx(0.0)


def test_summarize_results_skips_error_rows():
    """Error rows must not contribute to metrics."""
    results = [
        _result("QB", 42, "none", val_mae=5.00, test_mae=6.00),
        _result("QB", 42, "pca55", val_mae=0.0, test_mae=0.0, error="RuntimeError: crash"),
    ]
    summary = abl.summarize_results(results, variants=["none", "pca55"])
    # pca55 errored — val/test lists are empty, consistent_improver must be False
    assert summary["QB"]["variants"]["pca55"]["consistent_improver"] is False


def test_summarize_results_baseline_never_consistent_improver():
    """The 'none' baseline itself is never a consistent improver by definition."""
    results = [_result("QB", 42, "none", val_mae=5.00, test_mae=6.00)]
    summary = abl.summarize_results(results, variants=["none"])
    assert summary["QB"]["variants"]["none"]["consistent_improver"] is False


# ---------------------------------------------------------------------------
# CLI dry-run
# ---------------------------------------------------------------------------


def test_cli_dry_run_prints_plan_without_training(monkeypatch, capsys):
    monkeypatch.setattr(abl, "get_config", lambda pos: _base_cfg())

    def fail_run_grid(*args, **kwargs):
        raise AssertionError("dry-run must not execute jobs")

    monkeypatch.setattr(abl, "run_grid", fail_run_grid)

    abl.main(["--dry-run", "--positions", "QB", "TE"])
    out = capsys.readouterr().out

    # 2 positions × len(DEFAULT_VARIANTS) variants × 1 default seed
    expected = 2 * len(abl.DEFAULT_VARIANTS) * len(abl.DEFAULT_SEEDS)
    assert f"Planned ablation jobs: {expected}" in out
    assert "Experiment workers:" in out
    assert "QB" in out
    assert "TE" in out


def test_cli_dry_run_single_position(monkeypatch, capsys):
    monkeypatch.setattr(abl, "get_config", lambda pos: _base_cfg())
    monkeypatch.setattr(abl, "run_grid", lambda *a, **kw: (_ for _ in ()).throw(AssertionError()))

    abl.main(["--dry-run", "--positions", "QB", "--variants", "none,pca55"])
    out = capsys.readouterr().out
    # 1 position × 2 variants × 1 seed = 2
    assert "Planned ablation jobs: 2" in out


def test_cli_errors_if_baseline_variant_excluded(monkeypatch):
    monkeypatch.setattr(abl, "get_config", lambda pos: _base_cfg())
    with pytest.raises(SystemExit):
        abl.main(["--positions", "QB", "--variants", "pca55,pca40"])
