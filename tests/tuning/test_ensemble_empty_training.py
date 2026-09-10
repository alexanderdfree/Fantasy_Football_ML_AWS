"""Captured trainers cannot turn an empty epoch into a successful tuning result."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from src.tuning import ab_ensemble_seeds as ensemble
from src.tuning import tune_nn
from tests.tuning.test_ab_ensemble_seeds import (
    _build_models,
    _captures_for,
    _criterion,
    _synthetic,
    _tiny_cfg,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("FF_NN_NORM", "layer")

    def build(rows, kind="resident"):
        cfg = _tiny_cfg()
        models = _build_models(cfg, 2)
        features, targets, train = _synthetic(n=rows, batch_size=4)
        if kind == "dataloader":
            dataset = [
                (
                    *[feature[i] for feature in features],
                    {key: value[i] for key, value in targets.items()},
                )
                for i in range(rows)
            ]
            train = DataLoader(dataset, batch_size=4, drop_last=True)
        _, _, val = _synthetic(n=8, batch_size=4)
        captures = _captures_for(models, _criterion(cfg), train)
        for capture in captures:
            capture["val_loader"] = val
            capture["trainer"].device = torch.device("cpu")
        return captures, cfg

    return build


@pytest.mark.parametrize("routine", ["train_stacked", "train_sequential"])
@pytest.mark.parametrize("kind", ["resident", "dataloader"])
@pytest.mark.parametrize("rows", [0, 2])
def test_sized_empty_loader_fails_before_setup(setup, monkeypatch, routine, kind, rows):
    captures, cfg = setup(rows, kind)
    before = [
        {key: value.clone() for key, value in c["trainer"].model.state_dict().items()}
        for c in captures
    ]

    def forbidden(*args, **kwargs):
        pytest.fail("Empty training reached model/optimizer/scheduler setup")

    monkeypatch.setattr(ensemble, "stack_models", forbidden)
    monkeypatch.setattr(ensemble.torch.optim, "AdamW", forbidden)
    monkeypatch.setattr("src.shared.pipeline._build_scheduler", forbidden)
    with pytest.raises(ValueError, match="Training loader.*no batches"):
        getattr(ensemble, routine)(captures, cfg, torch.device("cpu"), 1)
    for capture, original in zip(captures, before, strict=True):
        for key, value in capture["trainer"].model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        assert not capture["trainer"].optimizer.state


@pytest.mark.parametrize("routine", ["train_stacked", "train_sequential"])
@pytest.mark.parametrize("initial_batches", [0, 1])
def test_unsized_empty_or_exhausted_epoch_cannot_return_success(setup, routine, initial_batches):
    captures, cfg = setup(4)
    loader = iter(captures[0]["train_loader"] if initial_batches else ())
    for capture in captures:
        capture["train_loader"] = loader
    reports = []
    kwargs = (
        {"epoch_callback": lambda *args: reports.append(args)} if routine == "train_stacked" else {}
    )
    with pytest.raises(ValueError, match="Training loader.*no batches"):
        getattr(ensemble, routine)(captures, cfg, torch.device("cpu"), 2, **kwargs)
    if routine == "train_stacked":
        assert len(reports) == initial_batches


@pytest.mark.parametrize("routine", ["train_stacked", "train_sequential"])
@pytest.mark.parametrize("kind", ["resident", "dataloader"])
def test_one_batch_preserves_real_parameter_updates(setup, routine, kind):
    captures, cfg = setup(4, kind)
    before = [
        {key: value.clone() for key, value in c["trainer"].model.named_parameters()}
        for c in captures
    ]
    result = getattr(ensemble, routine)(captures, cfg, torch.device("cpu"), 1)
    if routine == "train_stacked":
        params, _, _ = result
        changed = [
            any(not torch.equal(value[i], before[i][key]) for key, value in params.items())
            for i in range(2)
        ]
    else:
        changed = [
            any(not torch.equal(value, before[i][key]) for key, value in model.named_parameters())
            for i, model in enumerate(result)
        ]
    assert all(changed)


def test_actual_stacked_tuner_objective_rejects_underfilled_capture(setup, monkeypatch):
    captures, cfg = setup(2)
    monkeypatch.setattr(ensemble, "capture_seeds", lambda *args, **kwargs: (captures, {}))
    monkeypatch.setattr(tune_nn, "_sample_overrides", lambda *args: {})
    monkeypatch.setattr(tune_nn, "_validate_overrides", lambda *args: None)
    reports = []
    trial = SimpleNamespace(
        number=0, report=lambda *args: reports.append(args), should_prune=lambda: False
    )
    objective = tune_nn._make_stacked_objective("QB", cfg, 42, 2, 1)
    with pytest.raises(ValueError, match="Training loader.*no batches"):
        objective(trial)
    assert reports == []
