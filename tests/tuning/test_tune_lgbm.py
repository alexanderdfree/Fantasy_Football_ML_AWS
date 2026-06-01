"""Smoke tests for src/tuning/tune_lgbm.py.

Cover the cheap-but-load-bearing pieces of the module so a future rename or
refactor doesn't silently break the on-EC2 retune flow:

  * the module imports without dragging in optuna data prep / boto3 init,
  * ``main()`` accepts ``--print-best`` for unknown positions without
    crashing (the argparse boundary is the cheapest integration point),
  * ``_format_config_lines`` emits the lowercase ``lgbm_*=...`` kwarg form
    that ``build_position_config()`` consumes (regression guard for M6).
"""

from __future__ import annotations

import argparse
import contextlib
import os
from unittest.mock import patch

import numpy as np
import optuna
import pytest

from src.tuning import tune_lgbm

pytestmark = pytest.mark.unit


def test_module_imports_cleanly():
    """Module-level imports succeed without env vars or S3 contact."""
    assert hasattr(tune_lgbm, "main")
    assert hasattr(tune_lgbm, "_format_config_lines")
    assert hasattr(tune_lgbm, "_prepare_cv_folds")


def test_format_config_lines_emits_lowercase_kwarg_form():
    """Regression guard for M6: output must match build_position_config kwargs.

    Pre-M6 the helper emitted uppercase ``{POS}_LGBM_*`` constants targeting
    a config-file layout that never shipped. The factory currently consumes
    lowercase ``lgbm_*`` kwargs; any drift here breaks the manual-paste flow
    documented in retune-lgbm.yml.
    """
    best = {
        "n_estimators": 1500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "objective": "huber",
    }
    out = tune_lgbm._format_config_lines("QB", best)
    # Lowercase form, comma-terminated for direct paste.
    assert "lgbm_n_estimators=1500," in out
    assert "lgbm_num_leaves=31," in out
    assert 'lgbm_objective="huber",' in out
    # Float formatting uses %g (no trailing zero noise).
    assert "lgbm_learning_rate=0.05," in out
    # No uppercase constants left over.
    assert "QB_LGBM_" not in out
    assert "N_ESTIMATORS" not in out


def test_format_config_lines_skips_missing_params():
    """Params not present in best_params should be silently omitted."""
    out = tune_lgbm._format_config_lines("RB", {"n_estimators": 800})
    assert "lgbm_n_estimators=800," in out
    # max_depth not supplied → not in output
    assert "lgbm_max_depth" not in out


def test_trial_to_params_drops_use_max_depth_flag():
    """``use_max_depth=False`` should collapse to ``max_depth=-1`` (LGBM no-cap)."""
    fake_trial = argparse.Namespace(
        params={"use_max_depth": False, "n_estimators": 500, "max_depth": 7}
    )
    cleaned = tune_lgbm._trial_to_params(fake_trial)
    assert cleaned["max_depth"] == -1
    assert cleaned["n_estimators"] == 500
    assert "use_max_depth" not in cleaned


def test_parse_seeds_accepts_commas_and_rejects_bad_values():
    assert tune_lgbm._parse_seeds("42,43,44") == (42, 43, 44)
    assert tune_lgbm._parse_seeds("42 43") == (42, 43)

    for raw in ["", "   ", "42,42", "-1", "x"]:
        with pytest.raises(argparse.ArgumentTypeError):
            tune_lgbm._parse_seeds(raw)


def test_seed_versioned_study_names_do_not_match_legacy_names():
    seeds = (42, 43, 44)
    assert tune_lgbm._seed_key(seeds) == "s42-43-44"
    assert tune_lgbm._study_name("RB", seeds) == "lgbm_seedavg_v1_s42-43-44_rb"
    assert tune_lgbm._study_db_path("RB", seeds) == "tune_lgbm_seedavg_v1_s42-43-44_rb.db"
    assert tune_lgbm._study_name("RB", seeds) != "lgbm_rb"
    assert tune_lgbm._study_db_path("RB", seeds) != "tune_lgbm_rb.db"


def test_print_best_handles_missing_study(capsys, monkeypatch, tmp_path):
    """``main --print-best <POS>`` for an unknown position should print a
    soft error instead of crashing — the per-position loop continues so a
    follow-up position with a valid study still prints.

    Skip ``_ensure_data_from_s3`` so the test doesn't reach for boto3.
    ``chdir`` to ``tmp_path`` so the relative ``sqlite:///tune_lgbm_qb.db``
    storage path resolves to an empty tmp dir rather than the project root
    (which may carry a real tuning study from a prior interactive
    ``tune_lgbm.py`` run — that file would make this test see real trial
    data and fail the "no saved study" assertion).
    """
    monkeypatch.setattr(tune_lgbm, "_ensure_data_from_s3", lambda: None)
    monkeypatch.delenv("LGBM_N_JOBS", raising=False)
    monkeypatch.chdir(tmp_path)
    with patch("sys.argv", ["tune_lgbm.py", "QB", "--print-best"]):
        tune_lgbm.main()
    out = capsys.readouterr().out
    assert "No saved study for QB" in out
    assert "LGBM_N_JOBS" not in os.environ


def test_objective_scores_every_fold_seed_and_passes_leased_n_jobs(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def fake_lease(stage):
        assert stage == "tune_lgbm_cv"
        yield 7

    class FakeLGBM:
        def __init__(self, target_names, seed, n_jobs=None, **params):
            self.seed = seed
            self.n_jobs = n_jobs
            self.params = params
            calls.append({"seed": seed, "n_jobs": n_jobs, "params": params})

        def fit(self, X_train, y_train_dict, X_val, y_val_dict, feature_names=None):
            return None

        def predict(self, X_val):
            # Fold marker lives in X_val[0, 0], so expected MAEs are:
            # fold 10: seeds 1,3 -> 11,13 mean 12
            # fold 20: seeds 1,3 -> 21,23 mean 22
            return {"points": np.array([float(X_val[0, 0]) + self.seed])}

    class FakeTrial:
        def __init__(self):
            self.reports = []

        def suggest_categorical(self, name, choices):
            return choices[0]

        def suggest_int(self, name, low, high, step=1):
            return low

        def suggest_float(self, name, low, high, log=False):
            return low

        def report(self, value, step):
            self.reports.append((step, value))

        def should_prune(self):
            return False

    monkeypatch.setattr(tune_lgbm, "_lease_lgbm_cores", fake_lease)
    monkeypatch.setattr(tune_lgbm, "LightGBMMultiTarget", FakeLGBM)

    folds_data = [
        (
            np.array([[0.0]]),
            np.array([[10.0]]),
            {"points": np.array([0.0])},
            {"points": np.array([0.0])},
            ["feature"],
        ),
        (
            np.array([[0.0]]),
            np.array([[20.0]]),
            {"points": np.array([0.0])},
            {"points": np.array([0.0])},
            ["feature"],
        ),
    ]
    trial = FakeTrial()
    objective = tune_lgbm._make_objective(folds_data, ["points"], "huber", seeds=(1, 3))

    assert objective(trial) == pytest.approx(17.0)
    assert [(c["seed"], c["n_jobs"]) for c in calls] == [(1, 7), (3, 7), (1, 7), (3, 7)]
    assert trial.reports == [(0, pytest.approx(12.0)), (1, pytest.approx(17.0))]


def test_maybe_local_core_pool_disabled_preserves_thread_guard(monkeypatch):
    monkeypatch.delenv("LGBM_N_JOBS", raising=False)
    with tune_lgbm._maybe_local_core_pool(4, disabled=True) as (activity, status):
        assert activity is None
        assert status == "disabled"
        assert os.environ["LGBM_N_JOBS"] == "1"


def test_multiseed_comparison_returns_per_seed_and_aggregate(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def fake_lease(stage):
        assert stage == "tune_lgbm_compare"
        yield 5

    class FakeLGBM:
        def __init__(self, target_names, seed, n_jobs=None, **params):
            self.seed = seed
            self.n_jobs = n_jobs
            self.tuned = params.get("num_leaves") == 99
            calls.append({"seed": seed, "n_jobs": n_jobs, "tuned": self.tuned})

        def fit(self, X_train, y_train_dict, X_val, y_val_dict, feature_names=None):
            return None

        def predict(self, X_test):
            base = 10.0 if self.tuned else 0.0
            return {"points": np.array([base + self.seed])}

    def fake_read_parquet(path):
        return "df"

    def fake_prepare(pos, cfg, train_df, val_df, test_df):
        import pandas as pd

        return (
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            {"points": np.array([0.0])},
            {"points": np.array([0.0])},
            {"points": np.array([0.0])},
            None,
            None,
            pd.DataFrame({"fantasy_points": [0.0]}),
            ["feature"],
        )

    def fake_target_metrics(y_true, preds, targets):
        mae = float(preds["points"][0])
        return {
            "total": {"mae": mae, "r2": mae * 2},
            "points": {"mae": mae + 1, "r2": mae * 3},
        }

    def fake_ranking(df, pred_col):
        val = float(df[pred_col].iloc[0])
        return {"season_avg_hit_rate": val + 100, "season_avg_spearman": val + 200}

    monkeypatch.setattr(tune_lgbm, "_lease_lgbm_cores", fake_lease)
    monkeypatch.setattr(tune_lgbm, "LightGBMMultiTarget", FakeLGBM)
    monkeypatch.setattr(tune_lgbm.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(tune_lgbm, "_prepare_position_data", fake_prepare)
    monkeypatch.setattr(tune_lgbm, "compute_target_metrics", fake_target_metrics)
    monkeypatch.setattr(tune_lgbm, "compute_ranking_metrics", fake_ranking)

    cfg = {
        "targets": ["points"],
        "lgbm_objective": "huber",
        "lgbm_num_leaves": 31,
        "aggregate_fn": lambda preds: preds["points"],
    }
    result = tune_lgbm._run_comparison(
        "QB", cfg, {"num_leaves": 99, "objective": "huber"}, seeds=(1, 2)
    )

    assert result["seeds"] == [1, 2]
    assert [row["seed"] for row in result["per_seed"]] == [1, 2]
    assert result["aggregate"]["old_metrics"]["total"]["mae"] == {
        "mean": pytest.approx(1.5),
        "std": pytest.approx(0.70710678),
    }
    assert result["aggregate"]["new_metrics"]["total"]["mae"]["mean"] == pytest.approx(11.5)
    assert result["aggregate"]["delta_metrics"]["total"]["mae"]["mean"] == pytest.approx(10.0)
    assert result["aggregate"]["old_ranking"]["hit_rate"]["mean"] == pytest.approx(101.5)
    assert all(call["n_jobs"] == 5 for call in calls)


@pytest.mark.parametrize("logical,expected", [(32, 16), (16, 16), (8, 8), (4, 4), (1, 1)])
def test_default_n_jobs_caps_at_16_and_scales_down(monkeypatch, logical, expected):
    """``--n-jobs`` default is ``min(cpu_count, 16)``: all cores on small/medium
    boxes (16 on a 16-core workstation), capped at 16 on big ones (measured
    diminishing-returns point), so it never oversubscribes a CI / AWS-Batch host."""
    monkeypatch.setattr(os, "cpu_count", lambda: logical)
    assert tune_lgbm._default_n_jobs() == expected


def test_default_n_jobs_handles_none_cpu_count(monkeypatch):
    """os.cpu_count() may return None; the default must still be a valid >=1 int."""
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert tune_lgbm._default_n_jobs() == 1


def test_guard_lgbm_threads_pins_one_thread_when_parallel(monkeypatch):
    """>1 parallel trial → LGBM_N_JOBS defaulted to 1 so trials stay single-
    threaded (prevents the n_jobs x per-trial-threads oversubscription footgun)."""
    monkeypatch.delenv("LGBM_N_JOBS", raising=False)
    tune_lgbm._guard_lgbm_threads(16)
    assert os.environ["LGBM_N_JOBS"] == "1"


def test_guard_lgbm_threads_respects_explicit_setting(monkeypatch):
    """An explicitly-set LGBM_N_JOBS is preserved (setdefault, not override)."""
    monkeypatch.setenv("LGBM_N_JOBS", "4")
    tune_lgbm._guard_lgbm_threads(16)
    assert os.environ["LGBM_N_JOBS"] == "4"


def test_guard_lgbm_threads_noop_when_serial(monkeypatch):
    """n_jobs<=1 (no parallelism) → leave LGBM_N_JOBS untouched."""
    monkeypatch.delenv("LGBM_N_JOBS", raising=False)
    tune_lgbm._guard_lgbm_threads(1)
    assert "LGBM_N_JOBS" not in os.environ
