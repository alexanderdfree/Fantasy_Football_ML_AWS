"""Unit tests for src/tuning/ablate_backbone_norm.py.

Cover the loadability + the LayerNorm-backbone swap surface so a future
refactor of the shared backbone (``src/shared/neural_net.py::_build_backbone``,
the monkeypatch target) doesn't break the ablation CLI without warning.

The monkeypatch is applied INSIDE ``_execute_backbone_norm_job`` (the run_fn
passed to ``run_grid``), not at module level — this keeps each spawn worker
isolated. Tests verify that invariant without invoking the real pipeline.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

import src.shared.neural_net as nn_mod
from src.tuning import ablate_backbone_norm as abn
from src.tuning.ablation_runner import AblationJob, AblationResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Module-level smoke
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    for attr in (
        "VARIANTS",
        "main",
        "print_summary",
        "_build_backbone_layernorm",
        "_execute_backbone_norm_job",
        "_build_jobs",
    ):
        assert hasattr(abn, attr)


def test_variants_dict_is_bn_and_ln():
    """Exactly two variants — adding/removing one requires updating this test."""
    assert set(abn.VARIANTS) == {"bn", "ln"}


# ---------------------------------------------------------------------------
# LayerNorm backbone shape and norm-type
# ---------------------------------------------------------------------------


def test_layernorm_backbone_swaps_only_the_norm():
    """The LN variant must mirror the stock backbone structurally but use
    LayerNorm where the stock uses BatchNorm1d — that one swap is the entire
    ablation, so guard it directly."""
    stock = list(nn_mod._build_backbone(10, [8, 4], 0.1))
    ln = list(abn._build_backbone_layernorm(10, [8, 4], 0.1))

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
    """_execute_backbone_norm_job patches ``nn_mod._build_backbone`` — fail loudly
    here if that attribute ever moves or is renamed."""
    assert callable(nn_mod._build_backbone)


# ---------------------------------------------------------------------------
# Monkeypatch is applied/reverted inside the job, not at module level
# ---------------------------------------------------------------------------


def _make_dummy_job(variant: str) -> AblationJob:
    return AblationJob(
        position="WR",
        seed=42,
        variant=variant,
        label=abn.VARIANTS[variant],
        run_fn=abn._execute_backbone_norm_job,
        base_cfg={"targets": ["receiving_yards"], "train_lightgbm": True},
        metadata={"run_kind": "experiment"},
    )


def test_ln_job_patches_and_reverts_backbone(monkeypatch):
    """_execute_backbone_norm_job patches nn_mod._build_backbone for 'ln' and
    reverts it after the run — even if the run raises — so the module is clean."""
    original = nn_mod._build_backbone
    patched_during = []

    def fake_run_fn(seed, config):
        patched_during.append(nn_mod._build_backbone is abn._build_backbone_layernorm)
        raise RuntimeError("fake pipeline error")

    monkeypatch.setattr("src.tuning.ablate_backbone_norm.get_runner", lambda pos: fake_run_fn)
    monkeypatch.setattr(
        "src.tuning.ablate_backbone_norm.get_config",
        lambda pos: {"targets": ["receiving_yards"], "train_lightgbm": True},
    )

    job = _make_dummy_job("ln")
    with pytest.raises(RuntimeError, match="fake pipeline error"):
        abn._execute_backbone_norm_job(job)

    # Patch was active during the run.
    assert patched_during == [True]
    # Reverted after the run (try/finally).
    assert nn_mod._build_backbone is original


def test_bn_job_does_not_patch_backbone(monkeypatch):
    """The 'bn' variant must NOT touch nn_mod._build_backbone."""
    original = nn_mod._build_backbone
    seen_during = []

    def fake_run_fn(seed, config):
        seen_during.append(nn_mod._build_backbone is original)
        raise RuntimeError("fake pipeline error")

    monkeypatch.setattr("src.tuning.ablate_backbone_norm.get_runner", lambda pos: fake_run_fn)
    monkeypatch.setattr(
        "src.tuning.ablate_backbone_norm.get_config",
        lambda pos: {"targets": ["receiving_yards"], "train_lightgbm": True},
    )

    job = _make_dummy_job("bn")
    with pytest.raises(RuntimeError, match="fake pipeline error"):
        abn._execute_backbone_norm_job(job)

    assert seen_during == [True]
    assert nn_mod._build_backbone is original


# ---------------------------------------------------------------------------
# _build_jobs
# ---------------------------------------------------------------------------


def test_build_jobs_count():
    jobs = abn._build_jobs(positions=["WR"], seeds=[42, 43], variants=["bn", "ln"])
    # 1 position × 2 seeds × 2 variants = 4 jobs
    assert len(jobs) == 4


def test_build_jobs_structure():
    jobs = abn._build_jobs(positions=["QB"], seeds=[7], variants=["bn"])
    assert len(jobs) == 1
    job = jobs[0]
    assert job.position == "QB"
    assert job.seed == 7
    assert job.variant == "bn"
    assert job.run_fn is abn._execute_backbone_norm_job
    # base_cfg is the raw registry config; _make_cfg (which sets train_lightgbm=False)
    # is applied inside _execute_backbone_norm_job, not at job-build time.
    assert "targets" in job.base_cfg


def test_build_jobs_base_cfg_train_lightgbm_reset(monkeypatch):
    """_make_cfg inside the run_fn deep-copies and sets train_lightgbm=False;
    the original base_cfg must be untouched."""
    captured_cfg = {}

    def fake_run_fn(seed, config):
        captured_cfg.update(config)
        raise RuntimeError("stop")

    monkeypatch.setattr("src.tuning.ablate_backbone_norm.get_runner", lambda pos: fake_run_fn)
    monkeypatch.setattr(
        "src.tuning.ablate_backbone_norm.get_config",
        lambda pos: {"targets": [], "train_lightgbm": True},
    )

    job = abn._build_jobs(positions=["WR"], seeds=[0], variants=["bn"])[0]
    # base_cfg still has the original value
    assert job.base_cfg.get("train_lightgbm") is True

    with pytest.raises(RuntimeError):
        abn._execute_backbone_norm_job(job)

    # run_fn received a deep-copy with train_lightgbm=False
    assert captured_cfg.get("train_lightgbm") is False
    # original base_cfg unmodified
    assert job.base_cfg.get("train_lightgbm") is True


# ---------------------------------------------------------------------------
# print_summary — sentinel and verdict logic on synthetic AblationResult rows
# ---------------------------------------------------------------------------


def _result(variant: str, seed: int, attn: float, base: float, ridge: float) -> AblationResult:
    return AblationResult(
        position="WR",
        seed=seed,
        variant=variant,
        metrics={
            "attn_fp_mae": attn,
            "base_fp_mae": base,
            "ridge_fp_mae": ridge,
            "attn_targets": {"receiving_tds": 0.29, "receiving_yards": 20.2},
            "base_targets": {"receiving_tds": 0.30, "receiving_yards": 20.5},
        },
        timings={},
        metadata={},
    )


def _dict_row(variant: str, seed: int, attn: float, base: float, ridge: float) -> dict:
    """Plain-dict form — kept for backward-compat with the old test shape."""
    return {
        "variant": variant,
        "seed": seed,
        "attn_fp_mae": attn,
        "base_fp_mae": base,
        "ridge_fp_mae": ridge,
        "attn_targets": {"receiving_tds": 0.29, "receiving_yards": 20.2},
        "base_targets": {"receiving_tds": 0.30, "receiving_yards": 20.5},
    }


def test_print_summary_handles_empty_rows():
    """Edge case: empty rows must not raise (no mean-of-empty crash)."""
    abn.print_summary([], ["receiving_tds"])


def test_print_summary_sentinel_passes_when_ridge_matches(capsys):
    targets = ["receiving_tds", "receiving_yards"]
    rows = [_result("bn", 42, 4.21, 4.15, 4.72), _result("ln", 42, 4.20, 4.14, 4.72)]
    ok = abn.print_summary(rows, targets)
    out = capsys.readouterr().out
    assert ok is True  # identical Ridge MAE → clean single-variable comparison
    assert "VERDICT" in out


def test_print_summary_sentinel_fails_when_ridge_differs(capsys):
    """A Ridge-MAE mismatch means the variants saw different data/seed — the
    sentinel must flag it (False) rather than report a bogus NN delta."""
    targets = ["receiving_tds", "receiving_yards"]
    rows = [_result("bn", 42, 4.21, 4.15, 4.72), _result("ln", 42, 4.20, 4.14, 4.99)]
    ok = abn.print_summary(rows, targets)
    out = capsys.readouterr().out
    assert ok is False
    assert "MISMATCH" in out


def test_print_summary_accepts_plain_dicts(capsys):
    """print_summary normalises plain-dict rows for backward compat."""
    targets = ["receiving_tds", "receiving_yards"]
    rows = [_dict_row("bn", 42, 4.21, 4.15, 4.72), _dict_row("ln", 42, 4.20, 4.14, 4.72)]
    ok = abn.print_summary(rows, targets)
    out = capsys.readouterr().out
    assert ok is True
    assert "VERDICT" in out


def test_verdict_flat_when_within_noise(capsys):
    """When |mean Δ| <= std AND < FLAT_NOISE_THRESHOLD the verdict must say FLAT."""
    targets = ["receiving_tds"]
    # Two seeds where Δ is tiny and std > |mean|.
    rows = [
        _result("bn", 42, 4.20, 4.15, 4.72),
        _result("ln", 42, 4.201, 4.151, 4.72),  # Δ = +0.001
        _result("bn", 43, 4.20, 4.15, 4.72),
        _result("ln", 43, 4.199, 4.149, 4.72),  # Δ = -0.001
    ]
    abn.print_summary(rows, targets)
    out = capsys.readouterr().out
    assert "FLAT" in out


def test_verdict_layernorm_winner(capsys):
    """When LN consistently beats BN by > noise threshold the verdict names LayerNorm."""
    targets = ["receiving_tds"]
    rows = [
        _result("bn", 42, 4.30, 4.15, 4.72),
        _result("ln", 42, 4.20, 4.10, 4.72),  # Δ = -0.10 (LN better)
        _result("bn", 43, 4.32, 4.17, 4.72),
        _result("ln", 43, 4.22, 4.12, 4.72),  # Δ = -0.10
        _result("bn", 44, 4.28, 4.13, 4.72),
        _result("ln", 44, 4.18, 4.03, 4.72),  # Δ = -0.10
    ]
    abn.print_summary(rows, targets)
    out = capsys.readouterr().out
    assert "LayerNorm" in out


def test_verdict_batchnorm_winner(capsys):
    """When BN consistently beats LN by > noise threshold the verdict names BatchNorm."""
    targets = ["receiving_tds"]
    rows = [
        _result("bn", 42, 4.10, 4.05, 4.72),
        _result("ln", 42, 4.20, 4.15, 4.72),  # Δ = +0.10 (BN better)
        _result("bn", 43, 4.12, 4.07, 4.72),
        _result("ln", 43, 4.22, 4.17, 4.72),
        _result("bn", 44, 4.08, 4.03, 4.72),
        _result("ln", 44, 4.18, 4.13, 4.72),
    ]
    abn.print_summary(rows, targets)
    out = capsys.readouterr().out
    assert "BatchNorm" in out


# ---------------------------------------------------------------------------
# CLI smoke (no real pipeline)
# ---------------------------------------------------------------------------


def test_dry_run_does_not_train(monkeypatch, capsys):
    """--dry-run must print the plan and return without calling run_grid."""
    called = []
    monkeypatch.setattr(
        "src.tuning.ablate_backbone_norm.run_grid", lambda *a, **kw: called.append(1) or []
    )
    monkeypatch.setattr(
        "src.tuning.ablate_backbone_norm.get_config",
        lambda pos: {"targets": [], "train_lightgbm": True},
    )
    monkeypatch.setattr("src.tuning.ablate_backbone_norm.resolve_max_workers", lambda *a, **kw: 1)
    abn.main(["--positions", "WR", "--seeds", "42", "--dry-run"])
    out = capsys.readouterr().out
    assert "Planned ablation jobs" in out
    assert called == []


def test_main_error_exit_on_bad_variant():
    """An unknown variant name must cause a parser error (SystemExit)."""
    with pytest.raises(SystemExit):
        abn.main(["--variants", "nonexistent_variant"])
