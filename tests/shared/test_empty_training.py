"""A run must perform training before it can report a trained checkpoint."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.shared.neural_net import MultiHeadNet
from src.shared.training import (
    MultiHeadTrainer,
    MultiTargetDataset,
    MultiTargetLoss,
    _GPUResidentBatcher,
    make_dataloaders,
)

pytestmark = pytest.mark.unit


def _trainer():
    model = MultiHeadNet(1, ["yards"], [4], head_hidden=4, dropout=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.heads["yards"][-1].bias.fill_(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    return MultiHeadTrainer(
        model,
        optimizer,
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        MultiTargetLoss(["yards"], {"yards": 1.0}, head_losses={"yards": "mse"}),
        torch.device("cpu"),
        ["yards"],
    )


def _loader(rows, batch_size, kind, *, drop_last=True):
    x = np.arange(rows, dtype=np.float32).reshape(-1, 1)
    y = {"yards": np.full(rows, 2.0, dtype=np.float32)}
    if kind == "resident":
        return _GPUResidentBatcher(
            (torch.from_numpy(x),),
            {"yards": torch.from_numpy(y["yards"])},
            batch_size,
            shuffle=False,
            drop_last=drop_last,
        )
    return DataLoader(MultiTargetDataset(x, y), batch_size=batch_size, drop_last=drop_last)


@pytest.mark.parametrize("kind", ["dataloader", "resident"])
@pytest.mark.parametrize("rows", [0, 2, 3])
def test_empty_training_rejects_before_graph_setup_validation_or_checkpoint(
    kind, rows, monkeypatch
):
    trainer = _trainer()
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    setup_calls = []
    monkeypatch.setattr(
        trainer, "_maybe_graph_full_step", lambda loader: setup_calls.append(loader)
    )
    with pytest.raises(ValueError, match="Training loader.*no batches"):
        trainer.train(_loader(rows, 4, kind), _loader(4, 4, kind), n_epochs=1)
    assert not setup_calls
    assert trainer.best_model_state is None
    assert not trainer.optimizer.state
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["dataloader", "resident"])
@pytest.mark.parametrize("rows,drop_last", [(4, True), (3, False)])
def test_one_training_batch_still_updates_model_and_saves_checkpoint(kind, rows, drop_last):
    trainer = _trainer()
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    history = trainer.train(
        _loader(rows, 4, kind, drop_last=drop_last), _loader(4, 4, kind), n_epochs=1
    )
    assert history["train_loss"][0] > 0
    assert trainer.best_model_state is not None
    assert trainer.optimizer.state
    assert any(
        not torch.equal(value, before[key]) for key, value in trainer.model.state_dict().items()
    )


def test_real_cpu_loader_factory_does_not_silently_train_an_underfilled_cohort():
    x = np.zeros((2, 1), dtype=np.float32)
    y = {"yards": np.full(2, 2.0, dtype=np.float32)}
    train, val = make_dataloaders(x, y, x, y, batch_size=256, device=torch.device("cpu"))
    assert len(train) == 0 and len(val) == 1
    with pytest.raises(ValueError, match="Training loader.*no batches"):
        _trainer().train(train, val, n_epochs=1)


def test_unsized_empty_iterator_cannot_save_a_checkpoint():
    trainer = _trainer()
    with pytest.raises(ValueError, match="Training loader.*no batches"):
        trainer.train(iter(()), _loader(4, 4, "resident"), n_epochs=1)
    assert trainer.best_model_state is None
    assert not trainer.optimizer.state


def test_unsized_nonempty_iterator_performs_training():
    trainer = _trainer()
    history = trainer.train(iter(_loader(4, 4, "resident")), _loader(4, 4, "resident"), n_epochs=1)
    assert history["train_loss"][0] > 0
    assert trainer.best_model_state is not None
    assert trainer.optimizer.state
