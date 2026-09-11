"""Behavioral checks for RMSE-based NN checkpoint selection."""

import numpy as np
import pytest
import torch
from torch import nn

from src.rb.config import POSITION_CONFIG as RB_CONFIG
from src.shared.training import MultiHeadTrainer, MultiTargetLoss, _GPUResidentBatcher

pytestmark = pytest.mark.unit


class ScriptedPredictions(nn.Module):
    """Expose known validation predictions and save the selected epoch in state."""

    def __init__(self, predictions):
        super().__init__()
        self.predictions = {name: torch.tensor(values) for name, values in predictions.items()}
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("epoch", torch.tensor(-1))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.epoch.add_(1)
        return self

    def forward(self, x):
        if self.training:
            return {name: self.anchor.expand(len(x)) for name in self.predictions}
        rows = x[:, 0].long()
        return {name: values[self.epoch, rows] for name, values in self.predictions.items()}


def run_trajectory(
    predictions,
    *,
    weights=None,
    patience=20,
    batch_size=2,
    empty_val=False,
    selection_metric="weighted_rmse",
    epoch_callback=None,
    plateau=False,
):
    model = ScriptedPredictions(predictions)
    epochs, rows = next(iter(model.predictions.values())).shape
    targets = list(predictions)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        if plateau
        else torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    )
    trainer = MultiHeadTrainer(
        model,
        optimizer,
        scheduler,
        MultiTargetLoss(target_names=targets, loss_weights=weights or {}),
        torch.device("cpu"),
        targets,
        patience=patience,
        log_every=1,
        selection_metric=selection_metric,
        epoch_callback=epoch_callback,
    )
    # Exercise the production resident batcher's full batches plus ragged tail.
    x = torch.arange(rows, dtype=torch.float32).reshape(-1, 1)
    y = {name: torch.zeros(rows) for name in targets}
    loader = _GPUResidentBatcher((x,), y, batch_size=batch_size, shuffle=False, drop_last=False)
    history = trainer.train(loader, [] if empty_val else loader, epochs)
    return trainer, history


@pytest.mark.parametrize("patience", [1, 20], ids=["early-stop", "epoch-limit"])
def test_restores_rmse_winner_when_mae_prefers_another_epoch(patience, monkeypatch, capsys):
    monkeypatch.delenv("FF_NN_FIXED_EPOCHS", raising=False)
    trainer, history = run_trajectory(
        {"yards": [[0.0, 0.0, 6.0], [2.5, 2.5, 2.5], [4.0, 4.0, 4.0]]},
        patience=patience,
    )
    assert np.argmin(history["val_mae_yards"]) == 0
    assert np.argmin(history["val_rmse_weighted"]) == 1
    assert history["val_rmse_yards"] == pytest.approx([np.sqrt(12), 2.5, 4.0])
    assert trainer.best_val_metric == pytest.approx(2.5)
    assert trainer.model.epoch.item() == 1
    assert trainer.epochs_without_improvement == 1
    assert "weighted_rmse:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("weights", "expected_epoch", "expected_scores"),
    [(None, 0, [3.0, 4.5]), ({"yards": 0.1, "tds": 1.0}, 1, [4.2 / 1.1, 1.8 / 1.1])],
)
def test_checkpoint_metric_preserves_per_target_loss_weights(
    weights, expected_epoch, expected_scores
):
    trainer, history = run_trajectory(
        {"yards": [[2.0] * 3, [8.0] * 3], "tds": [[4.0] * 3, [1.0] * 3]},
        weights=weights,
    )
    assert history["val_rmse_weighted"] == pytest.approx(expected_scores)
    assert trainer.model.epoch.item() == expected_epoch


@pytest.mark.parametrize("batch_size", [2, 3])
def test_rmse_pools_validation_rows_before_taking_root(batch_size):
    _, history = run_trajectory({"yards": [[0.0, 0.0, 6.0]]}, batch_size=batch_size)
    assert history["val_rmse_yards"] == pytest.approx([np.sqrt(12)])
    assert history["val_mae_yards"] == pytest.approx([2.0])


def test_fixed_epoch_diagnostic_keeps_last_weights(monkeypatch):
    monkeypatch.setenv("FF_NN_FIXED_EPOCHS", "3")
    trainer, history = run_trajectory({"yards": [[1.0] * 3, [2.0] * 3, [3.0] * 3]}, patience=1)
    assert len(history["val_rmse_weighted"]) == 3
    assert trainer.best_val_metric == pytest.approx(1.0)
    assert trainer.model.epoch.item() == 2


@pytest.mark.parametrize("empty_val", [False, True], ids=["non-finite", "empty"])
def test_invalid_validation_never_saves_a_checkpoint(empty_val):
    trainer, history = run_trajectory({"yards": [[float("nan")] * 3]}, empty_val=empty_val)
    assert not np.isfinite(history["val_rmse_weighted"][0])
    assert trainer.best_model_state is None
    assert trainer.best_val_metric == float("inf")


def test_perfect_validation_has_zero_rmse():
    trainer, history = run_trajectory({"yards": [[0.0] * 3]})
    assert history["val_rmse_weighted"] == [0.0]
    assert trainer.best_model_state is not None


@pytest.mark.parametrize("patience", [1, 20], ids=["early-stop", "epoch-limit"])
def test_fantasy_selector_callback_and_scheduler_agree(patience):
    # The old per-stat score favors one TD error over two reception errors;
    # PPR correctly prefers the two-point miss over the six-point miss.
    predictions = {t: [[0.0] * 3 for _ in range(3)] for t in RB_CONFIG.targets}
    predictions["rushing_tds"][0] = [1.0] * 3
    predictions["receptions"][1:] = [[2.0] * 3, [3.0] * 3]
    reports = []
    trainer, history = run_trajectory(
        predictions,
        weights=RB_CONFIG.loss_weights,
        patience=patience,
        selection_metric="fantasy_rmse_ppr",
        plateau=True,
        epoch_callback=lambda epoch, value: reports.append((epoch, value)),
    )
    assert np.argmin(history["val_rmse_weighted"]) == 0
    assert reports == [(0, 6.0), (1, 2.0), (2, 3.0)]
    assert trainer.scheduler.best == 2.0
    assert trainer.best_val_metric == 2.0
    assert trainer.model.epoch.item() == 1
    selection = history["checkpoint_selection"]
    assert selection["metric"] == "fantasy_rmse_ppr"
    assert selection["epoch"] == 2
    assert selection["score"] == 2.0
    assert selection["validation_metrics"]["val_fantasy_rmse_standard"] == 0.0
    assert selection["validation_metrics"]["val_fantasy_rmse_half_ppr"] == 1.0


def test_fantasy_rmse_aggregates_errors_before_reduction():
    predictions = {t: [[0.0] * 3] for t in RB_CONFIG.targets}
    predictions["rushing_yards"] = [[60.0] * 3]
    predictions["fumbles_lost"] = [[3.0] * 3]
    _, history = run_trajectory(predictions, selection_metric="fantasy_rmse_ppr")
    assert history["val_rmse_weighted"][0] > 0.0
    assert history["val_fantasy_rmse_ppr"] == [0.0]


def test_fantasy_selector_rejects_incomplete_targets():
    with pytest.raises(ValueError, match="complete known position"):
        run_trajectory({"yards": [[0.0] * 3]}, selection_metric="fantasy_rmse_ppr")


def test_legacy_mae_selector_is_available_for_comparison():
    trainer, history = run_trajectory(
        {"yards": [[0.0, 0.0, 6.0], [2.5] * 3]}, selection_metric="weighted_mae"
    )
    assert trainer.model.epoch.item() == 0
    assert history["checkpoint_selection"]["metric"] == "weighted_mae"
