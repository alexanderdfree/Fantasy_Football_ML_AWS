"""No fitting: exercise numerical functions, switching and fold orchestration."""

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.tuning import audit_count_candidate as candidate
from src.tuning import audit_development as audit

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_switches():
    yield
    audit.select_changes("baseline")
    audit.STATE.clear()


def test_observed_reference_value_and_gradient():
    from src.analysis.repair_count_diagnostics import stable_ztnb_reference

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


def test_switch_reset_and_bagging_parameters_without_fit():
    from src.shared import training
    from src.shared.models import LightGBMMultiTarget

    original = training.ztnb2_log_prob
    audit.select_changes("count_precision")
    assert training.ztnb2_log_prob is candidate.ztnb2_log_prob
    audit.select_changes("bagging")
    assert training.ztnb2_log_prob is original
    fitted = LightGBMMultiTarget(["x"], subsample=0.5)
    assert fitted._params["subsample_freq"] == 1
    assert fitted._models["x"].get_params()["subsample_freq"] == 1
    assert LightGBMMultiTarget(["x"], subsample=1)._params["subsample_freq"] == 0
    audit.select_changes("baseline_rep")
    assert "subsample_freq" not in LightGBMMultiTarget(["x"])._params


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
    monkeypatch.setenv("FF_AUDIT_ORIGIN", "2024")
    with pytest.raises(ValueError, match="confirmation"):
        audit.configure({}, arm="baseline")


@pytest.mark.parametrize(
    "suffix,native,positions",
    [
        ("count", False, {"RB", "WR", "TE"}),
        ("bagging", False, {"QB", "RB", "WR", "TE"}),
        ("native_bagging", True, {"K", "DST"}),
    ],
)
def test_specs_are_eager_and_correct_provider(suffix, native, positions):
    spec = importlib.import_module(f"src.tuning.ab_audit_{suffix}_development")
    assert set(spec.POSITIONS) == positions
    assert spec.SEEDS == [42, 123, 7]
    assert spec.SUPPORTS_STACKED is False
    assert [v.name for v in spec.VARIANTS][:2] == ["baseline", "baseline_rep"]
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
