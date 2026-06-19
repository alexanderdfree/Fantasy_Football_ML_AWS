"""Unit tests for src/tuning/ablate_attn_arch.py.

Cover loadability, the variant→cfg-key mapping, the config-build surface, and the
decision-table sentinel — without invoking the real pipeline ``run`` (a
multi-minute training run, out of scope for unit tests).

The most important guard here is ``test_flag_keys_are_read_by_the_factory``: a
variant that toggles a cfg key the attention factory does not read is a silent
no-op (the key falls through to its OFF default with no error) — exactly the
failure mode this project warns about — so we assert every key the variants touch
still appears in ``neural_net.py``'s factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.shared.neural_net as nn_mod
from src.tuning import ablate_attn_arch as aaa
from src.tuning.ablation_runner import AblationJob, AblationResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Module surface checks
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    for attr in (
        "VARIANTS",
        "FLAG_VARIANTS",
        "KNOWN_FLAG_KEYS",
        "BASELINE",
        "DEFAULT_VARIANTS",
        "main",
        "print_summary",
        "_execute_attn_arch_job",
        "_extract_run_payload",
        "_make_cfg",
        "_build_jobs",
    ):
        assert hasattr(aaa, attr)


def test_variants_cover_the_seven_prs_plus_baseline_and_alibi_split():
    """Baseline + one variant per PR, with ALiBi split into stacked/replacing PE.

    Adding or removing a variant requires updating this test — it is the
    inventory of what the ablation claims to cover.
    """
    assert aaa.BASELINE == "baseline"
    assert set(aaa.VARIANTS) == {
        "baseline",
        "temp",
        "seqdrop",
        "swiglu",
        "entropy",
        "alibi",
        "alibi_only",
        "selfattn",
        "condq",
    }
    assert aaa.BASELINE not in aaa.FLAG_VARIANTS
    assert set(aaa.FLAG_VARIANTS) == set(aaa.VARIANTS) - {"baseline"}


def test_default_variants_is_baseline_plus_all_flags():
    assert aaa.DEFAULT_VARIANTS[0] == aaa.BASELINE
    assert set(aaa.DEFAULT_VARIANTS) == set(aaa.VARIANTS)


def test_baseline_override_is_empty():
    """The baseline must be the all-OFF production reference (no overrides)."""
    assert aaa.VARIANTS[aaa.BASELINE][1] == {}


def test_variant_overrides_match_documented_flags():
    """Lock the exact cfg key each variant toggles, including types (a float flag
    set to ``True`` would be a different experiment)."""
    o = {k: v[1] for k, v in aaa.VARIANTS.items()}
    assert o["temp"] == {"attn_learn_temperature": True}
    assert o["seqdrop"] == {"attn_history_dropout": 0.1}
    assert isinstance(o["seqdrop"]["attn_history_dropout"], float)
    assert o["swiglu"] == {"attn_use_swiglu_encoder": True}
    assert o["entropy"] == {"attn_entropy_coeff": 0.01}
    assert isinstance(o["entropy"]["attn_entropy_coeff"], float)
    assert o["alibi"] == {"attn_use_alibi_bias": True}
    # alibi_only must disable the production positional encoding so ALiBi replaces
    # it rather than stacking on top.
    assert o["alibi_only"] == {"attn_use_alibi_bias": True, "attn_positional_encoding": False}
    assert o["selfattn"] == {"attn_self_layers": 1}
    assert isinstance(o["selfattn"]["attn_self_layers"], int)
    assert o["condq"] == {"attn_condition_queries_on_static": True}


def test_every_override_key_is_declared_known():
    """No variant may touch a key outside KNOWN_FLAG_KEYS (typo guard)."""
    for v in aaa.FLAG_VARIANTS:
        assert set(aaa.VARIANTS[v][1]) <= aaa.KNOWN_FLAG_KEYS, v


def test_flag_keys_are_read_by_the_factory():
    """Every KNOWN_FLAG_KEYS entry must still be consumed by the attention factory
    as ``cfg.get("<key>"...)``. If a flag is renamed/dropped in neural_net.py this
    fails loudly instead of silently producing a no-op variant."""
    source = Path(nn_mod.__file__).read_text()
    for key in aaa.KNOWN_FLAG_KEYS:
        assert f'cfg.get("{key}"' in source, key


# ---------------------------------------------------------------------------
# _make_cfg
# ---------------------------------------------------------------------------


def test_make_cfg_disables_lightgbm_applies_override_and_deep_copies():
    base = {
        "targets": ["rushing_yards", "rushing_tds"],
        "train_lightgbm": True,
        "huber_deltas": {"rushing_yards": 15.0},
    }
    cfg = aaa._make_cfg(base, {"attn_use_alibi_bias": True})

    assert cfg["train_lightgbm"] is False
    assert cfg["attn_use_alibi_bias"] is True
    # Base config is untouched (deep copy, no leak).
    assert base["train_lightgbm"] is True
    assert "attn_use_alibi_bias" not in base
    # Nested containers are copied, not shared.
    cfg["huber_deltas"]["rushing_yards"] = 1.0
    assert base["huber_deltas"]["rushing_yards"] == 15.0


def test_make_cfg_baseline_only_disables_lightgbm():
    """Baseline override is empty — only LightGBM is flipped."""
    base = {"targets": ["rushing_yards"], "train_lightgbm": True, "train_ridge": True}
    cfg = aaa._make_cfg(base, {})
    assert cfg["train_lightgbm"] is False
    assert cfg["train_ridge"] is True


# ---------------------------------------------------------------------------
# _extract_run_payload
# ---------------------------------------------------------------------------


def test_extract_run_payload_raises_when_metrics_missing():
    with pytest.raises(RuntimeError, match="missing metrics"):
        aaa._extract_run_payload(
            {"attn_nn_metrics": None, "nn_metrics": None, "ridge_metrics": None},
            targets=["rushing_yards"],
            metadata={},
        )


def test_extract_run_payload_raises_when_only_attn_present():
    result = {
        "attn_nn_metrics": {"total": {"mae": 4.0}, "rushing_yards": {"mae": 20.0}},
        # nn_metrics and ridge_metrics missing
    }
    with pytest.raises(RuntimeError, match="missing metrics"):
        aaa._extract_run_payload(result, targets=["rushing_yards"], metadata={})


def test_extract_run_payload_maps_all_fields():
    result = {
        "attn_nn_metrics": {
            "total": {"mae": 4.21, "rmse": 5.90},
            "rushing_yards": {"mae": 20.5, "rmse": 28.1},
            "rushing_tds": {"mae": 0.3, "rmse": 0.5},
        },
        "nn_metrics": {"total": {"mae": 4.50, "rmse": 6.10}},
        "ridge_metrics": {"total": {"mae": 4.72, "rmse": 6.30}},
    }
    payload = aaa._extract_run_payload(
        result,
        targets=["rushing_yards", "rushing_tds"],
        metadata={"variant_label": "alibi"},
    )

    assert payload["metrics"]["attn_fp_mae"] == pytest.approx(4.21)
    assert payload["metrics"]["base_fp_mae"] == pytest.approx(4.50)
    assert payload["metrics"]["ridge_fp_mae"] == pytest.approx(4.72)
    assert payload["metrics"]["attn_fp_rmse"] == pytest.approx(5.90)
    assert payload["metrics"]["base_fp_rmse"] == pytest.approx(6.10)
    assert payload["metrics"]["ridge_fp_rmse"] == pytest.approx(6.30)
    assert payload["metrics"]["attn_targets"] == {
        "rushing_yards": pytest.approx(20.5),
        "rushing_tds": pytest.approx(0.3),
    }
    assert payload["metrics"]["attn_targets_rmse"] == {
        "rushing_yards": pytest.approx(28.1),
        "rushing_tds": pytest.approx(0.5),
    }
    assert payload["timings"] == {}
    assert payload["metadata"]["variant_label"] == "alibi"


# ---------------------------------------------------------------------------
# _build_jobs
# ---------------------------------------------------------------------------


def test_build_jobs_produces_correct_job_count_and_shape(monkeypatch):
    monkeypatch.setattr(
        aaa,
        "get_config",
        lambda pos: {"targets": ["rushing_yards"], "train_lightgbm": True},
    )
    jobs = aaa._build_jobs(
        position="RB",
        seeds=[42, 43, 44],
        variants=["baseline", "alibi", "seqdrop"],
    )
    # 3 seeds × 3 variants = 9 jobs
    assert len(jobs) == 9
    assert all(isinstance(j, AblationJob) for j in jobs)
    assert all(j.position == "RB" for j in jobs)
    assert all(j.run_fn is aaa._execute_attn_arch_job for j in jobs)
    assert all(j.metadata["run_kind"] == "experiment" for j in jobs)
    # Every (seed, variant) pair appears exactly once.
    pairs = [(j.seed, j.variant) for j in jobs]
    assert len(set(pairs)) == 9


def test_build_jobs_label_matches_variants_dict(monkeypatch):
    monkeypatch.setattr(
        aaa,
        "get_config",
        lambda pos: {"targets": ["rushing_yards"], "train_lightgbm": True},
    )
    jobs = aaa._build_jobs(position="RB", seeds=[42], variants=["alibi"])
    job = jobs[0]
    assert job.label == aaa.VARIANTS["alibi"][0]


# ---------------------------------------------------------------------------
# print_summary (decision table + sentinel)
# ---------------------------------------------------------------------------


def _make_result(
    variant: str,
    seed: int,
    attn: float,
    base: float,
    ridge: float,
    targets: dict[str, float] | None = None,
    attn_rmse: float | None = None,
) -> AblationResult:
    return AblationResult(
        position="RB",
        seed=seed,
        variant=variant,
        metrics={
            "attn_fp_mae": attn,
            "base_fp_mae": base,
            "ridge_fp_mae": ridge,
            "attn_fp_rmse": attn_rmse if attn_rmse is not None else attn + 1.5,
            "base_fp_rmse": base + 1.5,
            "ridge_fp_rmse": ridge + 1.5,
            "attn_targets": targets or {"rushing_yards": 20.2, "rushing_tds": 0.29},
            "attn_targets_rmse": {"rushing_yards": 28.0, "rushing_tds": 0.5},
        },
        timings={},
        metadata={"run_kind": "experiment"},
    )


def test_print_summary_handles_empty_results():
    """Edge case: empty results list must not raise."""
    ok = aaa.print_summary([], ["rushing_yards"], [aaa.BASELINE])
    assert ok is False


def test_print_summary_sentinel_passes_when_ridge_matches(capsys):
    targets = ["rushing_yards", "rushing_tds"]
    results = [
        _make_result("baseline", 42, 4.21, 4.15, 4.72),
        _make_result("alibi", 42, 4.18, 4.15, 4.72),
    ]
    ok = aaa.print_summary(results, targets, ["baseline", "alibi"])
    out = capsys.readouterr().out
    assert ok is True
    assert "VERDICT" in out
    assert "Δ vs baseline" in out


def test_print_summary_sentinel_fails_when_ridge_differs(capsys):
    """A Ridge-MAE mismatch means the variants saw different data/seed — flag it
    (False) rather than report a bogus flag delta."""
    targets = ["rushing_yards", "rushing_tds"]
    results = [
        _make_result("baseline", 42, 4.21, 4.15, 4.72),
        _make_result("alibi", 42, 4.18, 4.15, 4.99),
    ]
    ok = aaa.print_summary(results, targets, ["baseline", "alibi"])
    out = capsys.readouterr().out
    assert ok is False
    assert "MISMATCH" in out


def test_print_summary_verdict_labels_promising_and_flat(capsys):
    """A variant better than FLAT_NOISE_THRESHOLD gets 'PROMISING'; a tiny delta
    stays 'FLAT'. Uses two seeds so stdev is defined."""
    targets = ["rushing_yards"]
    results = [
        # seed 42
        _make_result("baseline", 42, 4.50, 4.00, 4.72),
        _make_result("alibi", 42, 4.40, 4.00, 4.72),  # Δ = -0.10 (promising)
        _make_result("temp", 42, 4.495, 4.00, 4.72),  # Δ = -0.005 (flat)
        # seed 43
        _make_result("baseline", 43, 4.50, 4.00, 4.72),
        _make_result("alibi", 43, 4.40, 4.00, 4.72),
        _make_result("temp", 43, 4.496, 4.00, 4.72),
    ]
    ok = aaa.print_summary(results, targets, ["baseline", "alibi", "temp"])
    out = capsys.readouterr().out
    assert ok is True
    assert "PROMISING" in out
    assert "FLAT" in out


def test_print_summary_verdict_single_seed_note(capsys):
    """Single-seed run prints the 'Directional only' caveat."""
    targets = ["rushing_yards"]
    results = [
        _make_result("baseline", 42, 4.50, 4.00, 4.72),
        _make_result("alibi", 42, 4.30, 4.00, 4.72),  # Δ = -0.20 (clearly promising)
    ]
    aaa.print_summary(results, targets, ["baseline", "alibi"])
    out = capsys.readouterr().out
    assert "Directional only" in out


def test_print_summary_errors_are_skipped(capsys):
    """Results with error set are excluded from the table gracefully."""
    targets = ["rushing_yards"]
    results = [
        _make_result("baseline", 42, 4.50, 4.00, 4.72),
        AblationResult(
            position="RB",
            seed=42,
            variant="alibi",
            metrics={},
            timings={},
            metadata={},
            error="RuntimeError: something went wrong",
        ),
    ]
    # Only baseline is present → not enough for a paired delta.
    ok = aaa.print_summary(results, targets, ["baseline", "alibi"])
    # Sentinel sees only 1 value → skips the spread check → remains True.
    assert ok is True


def test_print_summary_shows_rmse_row(capsys):
    """The RMSE headline row appears when results carry ``attn_fp_rmse`` — a flag
    can be MAE-flat but RMSE-better (the condq case), which only RMSE surfaces."""
    targets = ["rushing_yards"]
    results = [
        _make_result("baseline", 42, 4.50, 4.00, 4.72, attn_rmse=6.00),
        _make_result("condq", 42, 4.50, 4.00, 4.72, attn_rmse=5.90),  # MAE flat, RMSE -0.10
        _make_result("baseline", 43, 4.50, 4.00, 4.72, attn_rmse=6.00),
        _make_result("condq", 43, 4.50, 4.00, 4.72, attn_rmse=5.92),
    ]
    aaa.print_summary(results, targets, ["baseline", "condq"])
    out = capsys.readouterr().out
    assert "FP RMSE" in out


def test_print_summary_tolerates_missing_rmse(capsys):
    """Older MAE-only result JSONs (no ``attn_fp_rmse``) must still print — the
    RMSE row is skipped, no KeyError (the optional-key guard in _vals)."""
    targets = ["rushing_yards"]
    mae_only = {
        "attn_fp_mae": 4.5,
        "base_fp_mae": 4.0,
        "ridge_fp_mae": 4.72,
        "attn_targets": {"rushing_yards": 20.2},
    }
    results = [
        AblationResult("RB", 42, "baseline", dict(mae_only), {}, {"run_kind": "experiment"}),
        AblationResult("RB", 42, "condq", dict(mae_only), {}, {"run_kind": "experiment"}),
    ]
    aaa.print_summary(results, targets, ["baseline", "condq"])
    out = capsys.readouterr().out
    assert "FP MAE" in out
    assert "FP RMSE" not in out  # skipped gracefully


# ---------------------------------------------------------------------------
# CLI dry-run
# ---------------------------------------------------------------------------


def test_cli_dry_run_does_not_train(monkeypatch, capsys):
    monkeypatch.setattr(
        aaa,
        "get_config",
        lambda pos: {"targets": ["rushing_yards"], "train_lightgbm": True},
    )

    def fail_run_grid(*args, **kwargs):
        raise AssertionError("dry-run should not execute jobs")

    monkeypatch.setattr(aaa, "run_grid", fail_run_grid)

    aaa.main(["--dry-run", "--position", "RB", "--seeds", "42,43"])
    out = capsys.readouterr().out

    # 2 seeds × len(DEFAULT_VARIANTS) variants
    expected_jobs = 2 * len(aaa.DEFAULT_VARIANTS)
    assert f"Planned ablation jobs: {expected_jobs}" in out
    assert "Experiment workers:" in out
    assert "RB" in out


def test_cli_dry_run_variant_subset(monkeypatch, capsys):
    monkeypatch.setattr(
        aaa,
        "get_config",
        lambda pos: {"targets": ["rushing_yards"], "train_lightgbm": True},
    )
    monkeypatch.setattr(aaa, "run_grid", lambda *a, **kw: (_ for _ in ()).throw(AssertionError()))

    aaa.main(["--dry-run", "--position", "RB", "--seeds", "42", "--variants", "alibi,seqdrop"])
    out = capsys.readouterr().out
    # baseline is prepended automatically when absent: 1 seed × 3 variants = 3
    assert "Planned ablation jobs: 3" in out
