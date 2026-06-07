"""Unit tests for src/tuning/ablate_scheduler_type.py.

No training happens — ``run_position`` is driven with a fake ``run_fn`` and a
fake ``base_cfg`` so we test the config-swap, the data-identity sentinel, and the
verdict logic without a GPU or the real pipeline.
"""

from __future__ import annotations

import pytest

from src.tuning import ablate_scheduler_type as ast

pytestmark = pytest.mark.unit


# A cosine-production base config (QB/RB/WR/DST shape).
def _cosine_base():
    return {
        "scheduler_type": "cosine_warm_restarts",
        "cosine_t0": 40,
        "cosine_t_mult": 2,
        "cosine_eta_min": 1e-5,
        "nn_lr": 1e-3,
        "targets": ["a", "b"],
        "train_lightgbm": True,
    }


def _onecycle_base():
    return {
        "scheduler_type": "onecycle",
        "onecycle_max_lr": 1e-3,
        "onecycle_pct_start": 0.3,
        "nn_lr": 3e-4,
        "targets": ["a", "b"],
        "train_lightgbm": True,
    }


def test_canonical_params_per_type():
    cfg = _cosine_base()
    assert ast._canonical_params(cfg, "cosine_warm_restarts") == {
        "cosine_t0": 40,
        "cosine_t_mult": 2,
        "cosine_eta_min": 1e-5,
    }
    # onecycle peaks at 4x nn_lr.
    oc = ast._canonical_params(cfg, "onecycle")
    assert oc["onecycle_max_lr"] == pytest.approx(4e-3)
    assert oc["onecycle_pct_start"] == 0.3
    assert ast._canonical_params(cfg, "plateau") == {
        "plateau_factor": 0.5,
        "plateau_patience": 8,
    }


def test_scheduler_params_production_keeps_tuned_alt_canonical():
    base = _onecycle_base()  # production = onecycle, tuned max_lr 1e-3
    # Production type -> the position's own tuned params.
    assert ast._scheduler_params(base, "onecycle") == {
        "onecycle_max_lr": 1e-3,
        "onecycle_pct_start": 0.3,
    }
    # Alternative type -> canonical (cosine family-standard).
    assert ast._scheduler_params(base, "cosine_warm_restarts") == {
        "cosine_t0": 40,
        "cosine_t_mult": 2,
        "cosine_eta_min": 1e-5,
    }


def test_make_cfg_swaps_scheduler_and_drops_other_type_keys():
    base = _cosine_base()
    cfg = ast._make_cfg(base, "onecycle")
    assert cfg["scheduler_type"] == "onecycle"
    assert cfg["onecycle_max_lr"] == pytest.approx(4e-3)
    assert cfg["onecycle_pct_start"] == 0.3
    # Stale cosine keys removed so the active config is unambiguous.
    assert "cosine_t0" not in cfg
    assert "cosine_eta_min" not in cfg
    # LightGBM disabled (scheduler-free); base untouched (deep copy).
    assert cfg["train_lightgbm"] is False
    assert base["train_lightgbm"] is True
    assert base["scheduler_type"] == "cosine_warm_restarts"


def _fake_run_factory(attn_by_sched, ridge_by_sched=None):
    """Build a fake run_fn returning per-scheduler metrics in the pipeline shape."""

    def fake_run(*frames, seed=42, config=None):
        sched = config["scheduler_type"]
        targets = config["targets"]
        attn = attn_by_sched[sched]
        ridge = (ridge_by_sched or {}).get(sched, 5.0)
        return {
            "attn_nn_metrics": {"total": {"mae": attn}, **{t: {"mae": attn / 2} for t in targets}},
            "nn_metrics": {
                "total": {"mae": attn + 0.05},
                **{t: {"mae": attn / 2} for t in targets},
            },
            "ridge_metrics": {"total": {"mae": ridge}},
        }

    return fake_run


def test_run_position_returns_summary_and_picks_best():
    base = _cosine_base()
    run_fn = _fake_run_factory({"onecycle": 4.2, "cosine_warm_restarts": 4.0, "plateau": 4.1})
    out = ast.run_position("RB", [42], run_fn=run_fn, base_cfg=base)
    summary = out["summary"]
    assert summary["sentinel_ok"] is True  # ridge constant across types
    assert len(out["rows"]) == 3
    # cosine (4.0) is best AND production → verdict winner == production.
    assert summary["verdict"]["winner"] == "cosine_warm_restarts"
    assert summary["production_type"] == "cosine_warm_restarts"
    # Aggregated means recorded for each scheduler.
    assert summary["aggregated"]["onecycle"]["attn_fp_mae_mean"] == pytest.approx(4.2)


def test_run_position_flags_alt_candidate_single_seed():
    base = _onecycle_base()  # production onecycle
    # cosine (alt) clearly beats production onecycle by > flat threshold.
    run_fn = _fake_run_factory({"onecycle": 4.30, "cosine_warm_restarts": 4.00, "plateau": 4.25})
    out = ast.run_position("K", [42], run_fn=run_fn, base_cfg=base)
    v = out["summary"]["verdict"]
    assert v["winner"] == "cosine_warm_restarts"
    assert v["margin_vs_production"] == pytest.approx(0.30)
    assert v["note"] == "alt_candidate_single_seed"


def test_run_position_sentinel_fails_on_ridge_mismatch():
    base = _cosine_base()
    run_fn = _fake_run_factory(
        {"onecycle": 4.2, "cosine_warm_restarts": 4.0, "plateau": 4.1},
        ridge_by_sched={"onecycle": 5.0, "cosine_warm_restarts": 5.5, "plateau": 5.0},
    )
    out = ast.run_position("RB", [42], run_fn=run_fn, base_cfg=base)
    assert out["summary"]["sentinel_ok"] is False
    assert out["summary"]["verdict"]["note"] == "sentinel_failed"


def test_result_is_json_serializable():
    import json

    base = _cosine_base()
    run_fn = _fake_run_factory({"onecycle": 4.2, "cosine_warm_restarts": 4.0, "plateau": 4.1})
    out = ast.run_position("RB", [42], run_fn=run_fn, base_cfg=base)
    # Must round-trip — the Batch container uploads this dict as JSON to S3.
    json.dumps(out)
