"""Unit tests for src.tuning.ablate_batch_lr."""

from __future__ import annotations

import pytest

from src.tuning import ablate_batch_lr as abl
from src.tuning.ablation_runner import AblationResult

pytestmark = pytest.mark.unit


def _base_cfg(**updates):
    cfg = {
        "targets": ["passing_yards", "passing_tds"],
        "nn_batch_size": 128,
        "attn_batch_size": 256,
        "nn_lr": 5e-4,
        "attn_lr": 1e-3,
        "scheduler_type": "cosine_warm_restarts",
        "cosine_t0": 40,
        "cosine_t_mult": 2,
        "cosine_eta_min": 1e-5,
        "train_ridge": True,
        "train_base_nn": True,
        "train_elasticnet": True,
        "train_lightgbm": True,
        "train_attention_nn": True,
        "huber_deltas": {"passing_yards": 15.0},
    }
    cfg.update(updates)
    return cfg


def test_make_cfg_deep_copies_and_scales_cosine_scheduler():
    base = _base_cfg()
    cfg, meta = abl._make_cfg(base, abl.VARIANTS["b2_lrsqrt"], ridge_sentinel=False)

    assert cfg["attn_batch_size"] == 512
    assert cfg["attn_lr"] == pytest.approx(1e-3 * 2**0.5)
    assert cfg["cosine_eta_min"] == pytest.approx(1e-5)
    assert cfg["attn_cosine_eta_min"] == pytest.approx(1e-5 * 2**0.5)
    assert cfg["cosine_t0"] == 40
    assert cfg["cosine_t_mult"] == 2
    assert cfg["train_ridge"] is False
    assert cfg["train_base_nn"] is False
    assert cfg["train_elasticnet"] is False
    assert cfg["train_lightgbm"] is False
    assert cfg["train_attention_nn"] is True
    assert meta["base_attn_batch_size"] == 256
    assert meta["effective_attn_batch_size"] == 512
    assert base["attn_batch_size"] == 256
    cfg["huber_deltas"]["passing_yards"] = 1.0
    assert base["huber_deltas"]["passing_yards"] == 15.0


def test_make_cfg_caps_batch_at_1024_and_keeps_ridge_for_sentinel():
    cfg, meta = abl._make_cfg(
        _base_cfg(attn_batch_size=512),
        abl.VARIANTS["b4_lrsqrt"],
        ridge_sentinel=True,
    )

    assert cfg["attn_batch_size"] == 1024
    assert cfg["attn_lr"] == pytest.approx(2e-3)
    assert cfg["train_ridge"] is True
    assert meta["effective_attn_batch_size"] == 1024
    assert meta["ridge_sentinel"] is True


def test_make_cfg_scales_onecycle_max_lr():
    base = _base_cfg(
        scheduler_type="onecycle",
        onecycle_max_lr=2e-3,
        onecycle_pct_start=0.3,
    )
    base.pop("cosine_t0")
    base.pop("cosine_t_mult")
    base.pop("cosine_eta_min")

    cfg, meta = abl._make_cfg(base, abl.VARIANTS["b2_lrlin"], ridge_sentinel=False)

    assert cfg["attn_lr"] == pytest.approx(2e-3)
    assert cfg["onecycle_max_lr"] == pytest.approx(2e-3)
    assert cfg["attn_onecycle_max_lr"] == pytest.approx(4e-3)
    assert cfg["onecycle_pct_start"] == 0.3
    assert meta["effective_scheduler"]["attn_onecycle_max_lr"] == pytest.approx(4e-3)


def test_build_jobs_adds_one_seed_ridge_preflight(monkeypatch):
    monkeypatch.setattr(abl, "get_config", lambda position: _base_cfg())

    preflight, experiment = abl._build_jobs(
        positions=["QB"],
        seeds=[42, 43],
        variants=["baseline", "b2_lrsqrt"],
        ridge_sentinel_preflight=True,
    )

    assert [(j.seed, j.variant, j.metadata["run_kind"]) for j in preflight] == [
        (42, "baseline", "sentinel_preflight"),
        (42, "b2_lrsqrt", "sentinel_preflight"),
    ]
    assert all(j.metadata["ridge_sentinel"] is True for j in preflight)
    assert len(experiment) == 4
    assert all(j.metadata["run_kind"] == "experiment" for j in experiment)
    assert all(j.metadata["ridge_sentinel"] is False for j in experiment)


def test_extract_run_payload_records_metrics_and_timings():
    result = {
        "attn_nn_metrics": {
            "total": {"mae": 5.25},
            "passing_yards": {"mae": 20.0},
            "passing_tds": {"mae": 0.4},
        },
        "ridge_metrics": {"total": {"mae": 6.0}},
        "attn_history": {
            "val_loss": [9.0, 4.0, 5.0],
            "epoch_sec": [1.0, 2.0, 3.0],
            "peak_mem_gb": [0.1, 0.3, 0.2],
        },
        "phase_seconds": {"attn_nn_train": 8.5},
    }

    payload = abl._extract_run_payload(
        result,
        targets=["passing_yards", "passing_tds"],
        metadata={"variant_label": "x"},
    )

    assert payload["metrics"]["attn_fp_mae"] == 5.25
    assert payload["metrics"]["attn_target_mae"] == {
        "passing_yards": 20.0,
        "passing_tds": 0.4,
    }
    assert payload["metrics"]["min_attn_val_loss"] == 4.0
    assert payload["metrics"]["epochs"] == 3
    assert payload["metrics"]["ridge_fp_mae"] == 6.0
    assert payload["timings"]["epoch_sec_median"] == 2.0
    assert payload["timings"]["epoch_sec_sum"] == 6.0
    assert payload["timings"]["peak_mem_gb_max"] == 0.3
    assert payload["timings"]["attn_nn_train_sec"] == 8.5


def _result(position, seed, variant, mae, train_sec, *, run_kind="experiment", ridge=None):
    return AblationResult(
        position=position,
        seed=seed,
        variant=variant,
        metrics={"attn_fp_mae": mae, "ridge_fp_mae": ridge},
        timings={"attn_nn_train_sec": train_sec},
        metadata={"run_kind": run_kind},
    )


def test_summarize_results_marks_fast_accurate_variant_eligible():
    results = [
        _result("QB", 42, "baseline", 5.00, 100.0),
        _result("QB", 43, "baseline", 5.10, 100.0),
        _result("QB", 42, "b2_lrsqrt", 4.99, 80.0),
        _result("QB", 43, "b2_lrsqrt", 5.09, 80.0),
        _result("QB", 42, "b2_lrlin", 5.40, 50.0),
        _result("QB", 43, "b2_lrlin", 5.50, 50.0),
    ]

    summary = abl.summarize_results(
        results,
        variants=["baseline", "b2_lrsqrt", "b2_lrlin"],
        sentinels={"QB": {"ok": True, "max_spread": 0.0, "n": 2}},
    )

    assert summary["QB"]["sentinel"]["ok"] is True
    assert summary["QB"]["variants"]["b2_lrsqrt"]["delta_vs_baseline"]["mean"] == pytest.approx(
        -0.01
    )
    assert summary["QB"]["variants"]["b2_lrsqrt"]["speedup_vs_baseline"] == pytest.approx(0.2)
    assert summary["QB"]["variants"]["b2_lrsqrt"]["eligible"] is True
    assert summary["QB"]["variants"]["b2_lrlin"]["eligible"] is False
    assert "b2_lrsqrt" in summary["QB"]["recommendation"]


def test_ridge_sentinel_by_position_compares_preflight_rows():
    results = [
        _result("QB", 42, "baseline", 5.0, 100.0, run_kind="sentinel_preflight", ridge=6.0),
        _result("QB", 42, "b2_lrsqrt", 5.0, 90.0, run_kind="sentinel_preflight", ridge=6.0),
        _result("WR", 42, "baseline", 5.0, 100.0, run_kind="sentinel_preflight", ridge=6.0),
        _result("WR", 42, "b2_lrsqrt", 5.0, 90.0, run_kind="sentinel_preflight", ridge=6.1),
    ]

    sentinels = abl.ridge_sentinel_by_position(results)

    assert sentinels["QB"]["ok"] is True
    assert sentinels["WR"]["ok"] is False
    assert sentinels["WR"]["max_spread"] == pytest.approx(0.1)


def test_cli_dry_run_all_six_positions_does_not_train(monkeypatch, capsys):
    monkeypatch.setattr(abl, "get_config", lambda position: _base_cfg())

    def fail_run_grid(*args, **kwargs):
        raise AssertionError("dry-run should not execute jobs")

    monkeypatch.setattr(abl, "run_grid", fail_run_grid)

    abl.main(["--dry-run", "--positions", "QB", "RB", "WR", "TE", "K", "DST"])
    out = capsys.readouterr().out

    assert "Planned ablation jobs: 240" in out
    assert "Experiment workers:" in out
    for position in ("QB", "RB", "WR", "TE", "K", "DST"):
        assert position in out
