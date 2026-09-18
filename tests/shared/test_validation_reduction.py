"""Validation losses describe observations, independently of batch boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.shared.aggregate_targets import POSITION_TARGET_MAP, predictions_to_fantasy_points
from src.shared.training import (
    MultiHeadTrainer,
    MultiTargetDataset,
    MultiTargetLoss,
    _GPUResidentBatcher,
    _GraphedValPass,
)
from src.tuning.ab_ensemble_seeds import stack_models, stacked_val_rmse, train_stacked

pytestmark = pytest.mark.unit

# The stacked report scores PPR fantasy points, so its probes emit a real
# position's target set; the trainer probes below keep synthetic heads.
_RB_TARGETS = tuple(POSITION_TARGET_MAP["RB"])


class _ProbeModel(nn.Module):
    def __init__(self, targets=("a",), gain=1.0):
        super().__init__()
        self.targets = targets
        self.gain = nn.Parameter(torch.full((len(targets),), gain))

    def forward(self, x):
        return {name: x[:, 0] * (i + 1) * self.gain[i] for i, name in enumerate(self.targets)}


def _loader(values, batch_size, targets=("a",), kind="resident"):
    x = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    y = {name: torch.zeros(len(x)) for name in targets}
    if kind == "dataloader":
        dataset = MultiTargetDataset(x.numpy(), {name: value.numpy() for name, value in y.items()})
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return _GPUResidentBatcher((x,), y, batch_size, shuffle=False, drop_last=False)


def _ppr_points_per_unit(targets):
    """PPR points one unit of probe input scores when target ``i`` predicts ``i + 1``."""
    unit = {name: np.array([float(i + 1)]) for i, name in enumerate(targets)}
    return abs(float(predictions_to_fantasy_points("RB", unit)[0]))


def _stacked_cfg():
    return {
        "scheduler_type": "cosine_warm_restarts",
        "cosine_t0": 10,
        "cosine_t_mult": 1,
        "cosine_eta_min": 0.0,
    }


def _stacked_captures(selection_metric="fantasy_rmse_ppr"):
    criterion = MultiTargetLoss(
        target_names=list(_RB_TARGETS),
        loss_weights=dict.fromkeys(_RB_TARGETS, 1.0),
        head_losses=dict.fromkeys(_RB_TARGETS, "mse"),
    )
    captures = []
    for gain in (1.0, 2.0):
        model = _ProbeModel(_RB_TARGETS, gain=gain)
        captures.append(
            {
                "trainer": SimpleNamespace(
                    model=model,
                    criterion=criterion,
                    optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0),
                    selection_metric=selection_metric,
                    selection_position="RB",
                ),
                "train_loader": _loader([0, 0, 0, 0], 2, _RB_TARGETS),
                "val_loader": _loader([1, 1, 1, 10], 3, _RB_TARGETS),
            }
        )
    return captures


def _trainer(weights, callback=None):
    model = _ProbeModel(tuple(weights))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = MultiTargetLoss(
        target_names=list(weights), loss_weights=weights, head_losses=dict.fromkeys(weights, "mse")
    )
    trainer = MultiHeadTrainer(
        model,
        optimizer,
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        criterion,
        torch.device("cpu"),
        list(weights),
        patience=10,
        epoch_callback=callback,
        log_every=100,
    )
    return trainer


@pytest.mark.parametrize("kind", ["dataloader", "resident"])
@pytest.mark.parametrize("batch_size", [2, 3, 4, 8])
@pytest.mark.parametrize("weights", [{"a": 1.0}, {"a": 2.0, "b": 0.5}])
def test_actual_trainer_history_callback_and_plateau_use_sample_mean(kind, batch_size, weights):
    callbacks = []
    trainer = _trainer(weights, lambda epoch, value: callbacks.append((epoch, value)))
    train = _loader([0, 0, 0, 0], 2, tuple(weights), kind)
    val = _loader([1, 1, 1, 10], batch_size, tuple(weights), kind)
    history = trainer.train(train, val, 1)
    # Per-observation squared errors are 1, 1, 1, 100 for head a.
    expected = sum(weight * 25.75 * (i + 1) ** 2 for i, weight in enumerate(weights.values()))
    assert history["val_loss"] == pytest.approx([expected])
    assert callbacks == [(0, pytest.approx(expected))]
    assert trainer.scheduler.best == pytest.approx(expected)
    for i, target in enumerate(weights):
        assert history[f"val_loss_{target}"] == pytest.approx([25.75 * (i + 1) ** 2])
        assert history[f"val_mae_{target}"] == pytest.approx([3.25 * (i + 1)])
    weighted_mae = sum(weight * 3.25 * (i + 1) for i, weight in enumerate(weights.values()))
    assert trainer.best_val_metric == pytest.approx(weighted_mae / sum(weights.values()))


@pytest.mark.parametrize("batch_size", [2, 3, 4])
def test_graph_prefix_and_eager_tail_use_sample_counts_on_cpu(batch_size):
    """Execute the real graph body; this verifies math, not CUDA acceptance."""
    weights = {"a": 2.0, "b": 0.5}
    trainer = _trainer(weights)
    val = _loader([1, 1, 1, 10], batch_size, tuple(weights))
    graph = _GraphedValPass(trainer.model, trainer.criterion, val, list(weights), trainer.device)
    first = next(iter(val))
    _, components = trainer.criterion._compute_loss_components_capturable(
        trainer.model(first[0]), first[1]
    )
    graph.comp_sums = {key: torch.zeros(()) for key in components}
    graph._graph = SimpleNamespace(replay=graph._run_body)
    trainer._graphed_val = graph
    history = trainer.train(_loader([0, 0, 0, 0], 2, tuple(weights)), val, 1)
    assert history["val_loss"] == pytest.approx([103.0])
    assert history["val_loss_a"] == pytest.approx([25.75])
    assert history["val_loss_b"] == pytest.approx([103.0])
    assert history["val_mae_a"] == pytest.approx([3.25])


@pytest.mark.parametrize("batch_size", [2, 3, 4, 8])
def test_stacked_validation_pools_rows_for_each_member(batch_size):
    """Each member's PPR RMSE pools every validation row before the root, so the
    stacked report does not depend on how the loader partitions the rows."""
    models = [_ProbeModel(_RB_TARGETS, gain=1.0), _ProbeModel(_RB_TARGETS, gain=2.0)]
    template, params, buffers = stack_models(models, torch.device("cpu"))
    val = _loader([1, 1, 1, 10], batch_size, _RB_TARGETS)
    actual = stacked_val_rmse(template, params, buffers, "RB", val, torch.device("cpu"))
    # Per-observation fantasy-point errors are gain * unit * [1, 1, 1, 10].
    unit = _ppr_points_per_unit(_RB_TARGETS)
    assert actual == pytest.approx([np.sqrt(25.75) * unit, 2 * np.sqrt(25.75) * unit])
    assert template.training
    assert all(parameter.grad is None for parameter in params.values())


def test_empty_validation_preserves_ordinary_and_stacked_contracts():
    callbacks = []
    trainer = _trainer({"a": 1.0}, lambda epoch, value: callbacks.append((epoch, value)))
    val = _loader([], 3)
    history = trainer.train(_loader([0, 0], 2), val, 1)
    assert history["val_loss"] == history["val_loss_a"] == [0.0]
    assert callbacks == [(0, 0.0)]
    assert np.isinf(history["val_mae_a"][0])
    template, params, buffers = stack_models([_ProbeModel()], torch.device("cpu"))
    with pytest.raises(RuntimeError, match="empty val loader"):
        stacked_val_rmse(template, params, buffers, "RB", val, torch.device("cpu"))
    assert template.training


def test_stacked_epoch_callback_reports_mean_of_member_rmses():
    callbacks = []
    train_stacked(
        _stacked_captures(),
        _stacked_cfg(),
        torch.device("cpu"),
        1,
        epoch_callback=lambda epoch, value: callbacks.append((epoch, value)),
    )
    # Each member's RMSE pools all four rows (a batch of 3 plus a batch of 1);
    # the per-epoch report is the mean of the member RMSEs, not a pooled RMSE.
    unit = _ppr_points_per_unit(_RB_TARGETS)
    assert callbacks == [(0, pytest.approx((1.0 + 2.0) * np.sqrt(25.75) * unit / 2))]


def test_stacked_epoch_callback_requires_fantasy_rmse_selection():
    """The stacked report is PPR RMSE, so the captured trainers must select on it."""
    with pytest.raises(ValueError, match="nn_selection_metric"):
        train_stacked(
            _stacked_captures(selection_metric="weighted_mae"),
            _stacked_cfg(),
            torch.device("cpu"),
            1,
            epoch_callback=lambda epoch, value: None,
        )


def test_validation_partition_does_not_change_training_gradients_or_checkpoint():
    runs = []
    for batch_size in (2, 3):
        trainer = _trainer({"a": 2.0, "b": 0.5})
        trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=0.9)
        history = trainer.train(
            _loader([1, 2, 3, 4], 2, ("a", "b")),
            _loader([1, 1, 1, 10], batch_size, ("a", "b")),
            3,
        )
        runs.append((trainer, history))
    baseline, changed = runs
    assert baseline[1]["train_loss"] == changed[1]["train_loss"]
    assert baseline[1]["val_mae_a"] == changed[1]["val_mae_a"]
    assert baseline[0].best_val_metric == changed[0].best_val_metric
    torch.testing.assert_close(baseline[0].model.gain, changed[0].model.gain, rtol=0, atol=0)
    torch.testing.assert_close(
        baseline[0].model.gain.grad, changed[0].model.gain.grad, rtol=0, atol=0
    )


def test_acceptance_probe_refuses_a_cpu_only_host():
    if torch.cuda.is_available():
        pytest.skip("This control checks the CPU-only refusal")
    from src.analysis.verify_validation_reduction import verify_validation_reduction

    with pytest.raises(RuntimeError, match="requires real CUDA"):
        verify_validation_reduction()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA capture")
def test_actual_cuda_validation_reduction_acceptance():
    from src.analysis.verify_validation_reduction import verify_validation_reduction

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Graph validation is supported on sm_80+ only")
    proof = verify_validation_reduction()
    assert proof["passed_cases"] == 3.0
    assert proof["batch3_prefix_rows"] == 3.0
    assert proof["batch3_tail_rows"] == 1.0
