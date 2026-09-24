"""No fitting: exercise numerical functions, switching and fold orchestration."""

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.shared import training as candidate
from src.tuning import correctness_development as audit

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_switches():
    yield
    audit.select_changes("combined")
    audit.STATE.clear()


def test_observed_reference_value_and_gradient():
    from src.analysis.correctness_count_diagnostics import stable_ztnb_reference

    outputs = []
    for dtype, function in [
        (torch.float32, candidate.ztnb2_log_prob),
        (torch.float64, stable_ztnb_reference),
    ]:
        y = torch.tensor([1, 2, 3, 8, 16, 30], dtype=dtype)
        mu = torch.tensor([2e-6, 0.002, 0.2, 2, 12, 30], dtype=dtype, requires_grad=True)
        dispersion = torch.tensor([-5, -3, -1, 0, 1, 3], dtype=dtype, requires_grad=True)
        value = function(y, mu, dispersion)
        outputs.append(
            [
                v.detach().double().numpy()
                for v in (value, *torch.autograd.grad(value.sum(), (mu, dispersion)))
            ]
        )
    for actual, expected in zip(*outputs, strict=True):
        assert np.all(np.abs(actual - expected) / (1 + np.abs(expected)) < 1e-4)


def test_switch_reset_without_fit():
    from src.shared import training
    from src.tuning import correctness_legacy_count

    audit.select_changes("combined")
    candidate_function = training.ztnb2_log_prob
    assert candidate_function.__module__ == "src.shared.training"
    audit.select_changes("baseline")
    assert training.ztnb2_log_prob is correctness_legacy_count.ztnb2_log_prob
    audit.select_changes("precision")
    assert training.ztnb2_log_prob is candidate_function
    audit.select_changes("mean")
    assert training.ztnb2_log_prob is correctness_legacy_count.ztnb2_log_prob


def test_nonfinite_independent_oracle_cannot_pass(monkeypatch):
    from src.analysis import correctness_count_diagnostics as diagnostics

    monkeypatch.setattr(
        diagnostics,
        "stable_ztnb_reference",
        lambda y, mu, log_alpha: (mu + log_alpha) * float("nan"),
    )
    with pytest.raises(ValueError, match="Independent count oracle returned nonfinite"):
        diagnostics.observed_likelihood_check([1.0], [1.0], [0.0], device="cpu")


@pytest.mark.parametrize("position,floor", [("skill", 2013), ("K", 2015), ("DST", 2013)])
@pytest.mark.parametrize("origin", [2022, 2023])
def test_split_boundaries(position, floor, origin):
    frame = pd.DataFrame({"season": range(2012, 2026), "season_type": "REG"})
    train, val, test = audit.split_origin((frame, None), origin, position=position)
    assert train.season.min() == floor
    assert train.season.max() == origin - 2
    assert list(val.season) == [origin - 1]
    assert list(test.season) == [origin]


@pytest.mark.parametrize("position", ["K", "DST"])
def test_native_provider_defers_imputation_and_preserves_bindings(monkeypatch, position):
    from src.training.contracts import DatasetSplits

    module = importlib.import_module(f"src.{position.lower()}.run_pipeline")
    history = object()
    calls = []

    def provider(config, *, cross_validation=False):
        calls.append(cross_validation)
        frame = pd.DataFrame({"season": range(2013, 2026), "season_type": "REG"})
        return DatasetSplits(frame, None, None, {"attn_history_builder_fn": history})

    monkeypatch.setattr(module, "provide_dataset", provider)
    audit.STATE.update(origin=2022)
    audit.install_native_origin(position)
    result = module.provide_dataset({})
    assert calls == [True]
    assert result.bindings["attn_history_builder_fn"] is history
    assert result.train.season.max() == 2020
    assert list(result.val.season) == [2021]
    assert list(result.test.season) == [2022]


def test_local_training_and_confirmation_are_rejected(monkeypatch):
    monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="AWS Batch"):
        audit.configure({}, arm="baseline")
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "fake-for-no-fit-test")
    monkeypatch.setenv("FF_CORRECTNESS_ORIGIN", "2024")
    with pytest.raises(ValueError, match="confirmation"):
        audit.configure({}, arm="baseline")


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_configured_position_uses_each_native_filter(position):
    module = importlib.import_module(f"src.{position.lower()}.run_pipeline")
    assert audit.configured_position(module.CONFIG) == position


def test_wr_and_te_shared_targets_keep_distinct_observer_identities(monkeypatch):
    wr = importlib.import_module("src.wr.run_pipeline").CONFIG
    te = importlib.import_module("src.te.run_pipeline").CONFIG
    assert set(wr["targets"]) == set(te["targets"])
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "fake-for-no-fit-test")
    monkeypatch.setenv("FF_CORRECTNESS_ORIGIN", "2022")
    monkeypatch.setenv("FF_AMP_DTYPE", "fp32")
    monkeypatch.setattr(audit, "select_changes", lambda arm: None)
    monkeypatch.setattr(audit, "install_observer", lambda: None)
    audit.configure(dict(wr), arm="precision")
    assert audit.STATE["position"] == "WR"
    audit.configure(dict(te), arm="precision")
    assert audit.STATE["position"] == "TE"


def test_unknown_filter_identity_fails_before_installing_any_fit_observer(monkeypatch):
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "fake-for-no-fit-test")
    monkeypatch.setenv("FF_CORRECTNESS_ORIGIN", "2022")
    monkeypatch.setenv("FF_AMP_DTYPE", "fp32")
    with monkeypatch.context() as guard:
        guard.setattr(audit, "select_changes", lambda arm: pytest.fail("switches installed"))
        guard.setattr(audit, "install_observer", lambda: pytest.fail("observer installed"))
        with pytest.raises(ValueError, match="Unknown production position filter identity"):
            audit.configure({"filter_fn": lambda frame: frame}, arm="baseline")


@pytest.mark.parametrize(
    "suffix,native,positions,arms",
    [
        ("affected", False, {"RB", "WR", "TE"}, ["baseline", "precision", "mean", "combined"]),
        ("qb", False, {"QB"}, ["baseline", "combined"]),
        ("native", True, {"K", "DST"}, ["baseline", "combined"]),
    ],
)
def test_specs_are_eager_and_correct_provider(suffix, native, positions, arms):
    spec = importlib.import_module(f"src.tuning.ab_correctness_{suffix}")
    assert set(spec.POSITIONS) == positions
    assert spec.SEEDS == [42, 123, 7]
    assert spec.SUPPORTS_STACKED is False
    assert [v.name for v in spec.VARIANTS] == arms
    assert all((v.frame_injector is None) == native for v in spec.VARIANTS)


def test_saved_inference_parity_rejects_changed_values(monkeypatch):
    from src.analysis import artifact_eval

    frame = pd.DataFrame({"player_id": ["p"], "season": [2022], "week": [1]})
    for family in audit.FAMILIES:
        frame[f"pred_{family}_total"] = 1.0
        frame[f"pred_{family}_x"] = 1.0
    replay = frame.copy()
    replay["pred_nn_x"] = 2.0
    monkeypatch.setattr(artifact_eval, "build_test_df_from_artifacts", lambda *a, **kw: replay)
    monkeypatch.setattr(
        audit,
        "current_context",
        lambda: SimpleNamespace(output_dir=lambda position: __import__("pathlib").Path("/tmp")),
    )
    audit.STATE["frames"] = (frame, frame, frame)

    class Result(dict):
        recipe = {"targets": ["x"]}

    result = Result(test_df=frame, per_target_preds={})
    with pytest.raises(AssertionError):
        audit.inference_parity(result, "WR")


def test_observer_rejects_numerical_defect_in_corrected_arm_without_fitting(monkeypatch):
    from src.analysis import correctness_count_diagnostics
    from src.shared import training

    class FakeTrainer:
        device = torch.device("cpu")
        _use_amp = False
        _graphed_step = None

        def train(self, train_loader, val_loader, n_epochs):
            return {
                "checkpoint_selection": {"epoch": 1},
                "val_fantasy_mae_ppr": [1.0],
                "val_fantasy_rmse_ppr": [1.0],
                "val_selection_metric": [1.0],
            }

    predictions = {
        "receptions": np.array([1.0, 2.0]),
        "receptions_value_mu": np.array([0.1, 0.2]),
        "receptions_value_log_alpha": np.array([0.0, 0.0]),
        "receptions_gate_logit": np.array([0.0, 0.0]),
    }
    monkeypatch.setattr(training, "MultiHeadTrainer", FakeTrainer)
    monkeypatch.setattr(
        audit, "_validation_predictions", lambda *args: (predictions, {"receptions": np.ones(2)})
    )
    monkeypatch.setattr(audit, "_checkpoint_metrics", lambda *args: {"mae": 1.0, "rmse": 1.0})
    monkeypatch.setattr(
        correctness_count_diagnostics,
        "observed_likelihood_check",
        lambda *args: {"active_numerical_defect": True},
    )
    saved = []
    monkeypatch.setattr(audit, "evidence", lambda name, payload: saved.append(name))
    audit.STATE.update(arm="combined", position="WR", trainers=[])
    audit.install_observer()
    with pytest.raises(ValueError, match="fails observed reference"):
        FakeTrainer().train(None, None, 1)
    assert saved == ["failed-count-check.json"]
