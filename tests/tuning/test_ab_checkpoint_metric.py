"""The selector comparison changes only selection and must keep early stopping."""

import copy

import pandas as pd
import pytest

from src.shared.registry import get_config
from src.tuning.ab_checkpoint_metric import VARIANTS, metric_fn
from src.tuning.tune_nn_storage import SCOPE_ROOTS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_production_and_ab_selectors(position, monkeypatch):
    monkeypatch.delenv("FF_NN_FIXED_EPOCHS", raising=False)
    original = get_config(position)
    assert original["nn_selection_metric"] == "fantasy_rmse_ppr"
    for variant, metric in zip(
        VARIANTS, ("weighted_mae", "weighted_rmse", "fantasy_rmse_ppr"), strict=True
    ):
        cfg = copy.deepcopy(original)
        expected = dict(cfg)
        cfg = variant.cfg_mutator(cfg)
        assert cfg.pop("nn_selection_metric") == metric
        expected.pop("nn_selection_metric")
        assert cfg == expected


def test_fixed_epoch_modes_cannot_silently_disable_the_experiment(monkeypatch):
    monkeypatch.setenv("FF_NN_FIXED_EPOCHS", "30")
    with pytest.raises(ValueError, match="requires early stopping"):
        VARIANTS[0].cfg_mutator({})


def test_both_tuning_scopes_use_a_new_metric_namespace():
    assert SCOPE_ROOTS["full"] == "scheduler_v2_fp_rmse_ppr_v1"
    assert SCOPE_ROOTS["history"] == "history_v2_fp_rmse_ppr_v1"


def test_report_uses_shared_stat_truth_in_every_scoring_format():
    targets = get_config("RB")["targets"]
    frame = pd.DataFrame({t: [0.0, 1.0] for t in targets})
    frame["fantasy_points"] = [999.0, 999.0]  # incompatible full-total fallback
    for t in targets:
        frame[f"pred_nn_{t}"] = frame[t] + (1.0 if t == "receptions" else 0.0)
    scores = metric_fn({"test_df": frame, "nn_metrics": {}}, "RB")["NN"]
    assert scores["rmse"] == 1.0
    assert scores["half_ppr_rmse"] == 0.5
    assert scores["standard_rmse"] == 0.0


def test_cuda_auto_mode_keeps_checkpoint_spec_eager(monkeypatch, tmp_path):
    from src.tuning import ab_harness

    monkeypatch.setattr("src.shared.utils.cuda_enabled", lambda: True)
    calls = []
    monkeypatch.setattr(ab_harness, "run_sequential", lambda *args: calls.append("eager") or [])
    monkeypatch.setattr(ab_harness, "run_sequential_stacked", lambda *args: pytest.fail("stacked"))
    ab_harness.run_ab(
        "src.tuning.ab_checkpoint_metric",
        positions=["RB"],
        seeds=[42],
        data_dir=str(tmp_path),
        jobs=1,
    )
    assert calls == ["eager"]
    with pytest.raises(ValueError, match="does not support stacked"):
        ab_harness.run_ab(
            "src.tuning.ab_checkpoint_metric",
            positions=["RB"],
            seeds=[42],
            data_dir=str(tmp_path),
            jobs=1,
            stacked_seeds=True,
        )


def test_batch_rejects_stacked_checkpoint_spec_before_submission(monkeypatch):
    import sys

    from src.tuning import launch_ab

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_ab",
            "--spec",
            "src.tuning.ab_checkpoint_metric",
            "--positions",
            "RB",
            "--stacked-seeds",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit, match="does not support stacked"):
        launch_ab.main()
