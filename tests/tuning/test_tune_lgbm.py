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
import os
from unittest.mock import patch

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
    monkeypatch.chdir(tmp_path)
    with patch("sys.argv", ["tune_lgbm.py", "QB", "--print-best"]):
        tune_lgbm.main()
    out = capsys.readouterr().out
    assert "No saved study for QB" in out


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
