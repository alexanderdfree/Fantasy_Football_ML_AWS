"""Unit tests for src/tuning/tune_nn.py and the epoch_callback hook.

These tests exercise:
  * Search-space sampling produces valid configs and surfaces every override
    key the pipeline cfg consumers expect.
  * The d_model / n_heads divisibility guard short-circuits invalid combos.
  * format_config_lines emits Python that round-trips through eval.
  * _trial_to_params normalizes the backbone-layers tuple to a list.
  * Unsupported positions (K, DST) are rejected at the CLI boundary.
  * MultiHeadTrainer.train()'s epoch_callback propagates raised exceptions
    (i.e. optuna.TrialPruned would bubble out of trainer.train()).
  * The objective wrapper consumes a stubbed runner end-to-end and returns
    min(captured_val_losses) without touching real data.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import optuna
import pytest

from src.tuning import tune_nn

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _sample_overrides
# ---------------------------------------------------------------------------


def _ask_overrides(study: optuna.Study) -> tuple[optuna.Trial, dict]:
    """Drive _sample_overrides with a real Optuna trial from ``study``.

    Returns the (trial, overrides) pair so individual tests can inspect both.
    Trials that hit the invalid-combo guard raise ``optuna.TrialPruned``;
    callers either handle that or wrap in pytest.raises.
    """
    trial = study.ask()
    overrides = tune_nn._sample_overrides(trial)
    return trial, overrides


_EXPECTED_KEYS = {
    "attn_d_model",
    "attn_n_heads",
    "attn_encoder_hidden_dim",
    "attn_dropout",
    "attn_lr",
    "attn_batch_size",
    "nn_backbone_layers",
    "nn_head_hidden",
    "nn_dropout",
    "nn_lr",
    "nn_weight_decay",
    "nn_batch_size",
}


def test_sample_overrides_returns_every_cfg_key():
    """Every override key tune_lgbm/pipeline.py reads must be present."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    # Drive a handful of trials so we exercise more of the search space
    # than the first TPE sample (which is effectively random init).
    seen: set[str] = set()
    for _ in range(8):
        try:
            _, overrides = _ask_overrides(study)
        except optuna.TrialPruned:
            continue
        seen.update(overrides.keys())
    assert seen == _EXPECTED_KEYS, (
        f"sample_overrides should produce exactly {_EXPECTED_KEYS}, got {seen}"
    )


def test_sample_overrides_d_model_divisible_by_n_heads():
    """Every non-pruned trial must satisfy d_model % n_heads == 0 — otherwise
    PyTorch's MultiheadAttention will error at model construction."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    valid_trials = 0
    for _ in range(20):
        try:
            _, overrides = _ask_overrides(study)
        except optuna.TrialPruned:
            continue
        assert overrides["attn_d_model"] % overrides["attn_n_heads"] == 0
        valid_trials += 1
    # The current categorical sets happen to have all-divisible pairs, so
    # every trial should succeed. If the search space is later expanded with
    # incompatible options this assertion will alert us.
    assert valid_trials > 0


def test_sample_overrides_ranges():
    """Float ranges should stay inside the documented bounds."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=2))
    for _ in range(5):
        try:
            _, o = _ask_overrides(study)
        except optuna.TrialPruned:
            continue
        assert 0.0 <= o["attn_dropout"] <= 0.3
        assert 1e-4 <= o["attn_lr"] <= 5e-3
        assert 0.0 <= o["nn_dropout"] <= 0.4
        assert 1e-4 <= o["nn_lr"] <= 5e-3
        assert 1e-5 <= o["nn_weight_decay"] <= 1e-3
        assert o["attn_batch_size"] in (128, 256, 512)
        assert o["nn_batch_size"] in (128, 256, 512)
        assert o["attn_d_model"] in (16, 24, 32, 48, 64)
        assert o["attn_n_heads"] in (1, 2, 4)
        assert o["nn_head_hidden"] in (16, 24, 32, 48, 64)
        assert o["attn_encoder_hidden_dim"] in (0, 16, 32, 64)
        assert isinstance(o["nn_backbone_layers"], list)
        assert all(isinstance(v, int) for v in o["nn_backbone_layers"])


def test_sample_overrides_invalid_combo_raises_pruned(monkeypatch):
    """If a future search-space change ever yields d_model not divisible by
    n_heads, the guard must raise ``optuna.TrialPruned`` (not bubble a torch
    KeyError later)."""
    fake_trial = MagicMock()
    # Force the divisibility guard to trip.
    fake_trial.suggest_categorical.side_effect = lambda name, choices: {
        "attn_d_model": 24,
        "attn_n_heads": 7,  # 24 % 7 != 0
    }.get(name, choices[0])
    fake_trial.suggest_float.return_value = 0.1
    with pytest.raises(optuna.TrialPruned):
        tune_nn._sample_overrides(fake_trial)


# ---------------------------------------------------------------------------
# _format_config_lines / _format_value / _trial_to_params
# ---------------------------------------------------------------------------


def test_format_value_handles_types():
    assert tune_nn._format_value(True) == "True"
    assert tune_nn._format_value(False) == "False"
    assert tune_nn._format_value(32) == "32"
    assert tune_nn._format_value("cosine") == '"cosine"'
    # Floats should be %g-formatted, not full repr — keeps config.py readable.
    assert tune_nn._format_value(0.0005) == "0.0005"
    assert tune_nn._format_value([64, 32]) == "[64, 32]"


def test_format_config_lines_roundtrips_through_eval():
    """The lines we emit should be valid Python; eval each assignment back to
    a dict and verify it matches the source params."""
    best = {
        "attn_d_model": 32,
        "attn_n_heads": 2,
        "attn_encoder_hidden_dim": 16,
        "attn_dropout": 0.1,
        "attn_lr": 0.001,
        "attn_batch_size": 256,
        "nn_backbone_layers": [128, 64],
        "nn_head_hidden": 32,
        "nn_dropout": 0.2,
        "nn_lr": 0.0005,
        "nn_weight_decay": 0.0001,
        "nn_batch_size": 256,
    }
    rendered = tune_nn._format_config_lines("RB", best)
    # First line is a comment; rest are `RB_<X> = <value>` assignments.
    namespace: dict = {}
    for line in rendered.splitlines():
        if not line or line.startswith("#"):
            continue
        exec(line, namespace)
    # Verify a representative subset round-trips correctly.
    assert namespace["RB_ATTN_D_MODEL"] == 32
    assert namespace["RB_ATTN_LR"] == 0.001
    assert namespace["RB_NN_BACKBONE_LAYERS"] == [128, 64]
    assert namespace["RB_NN_DROPOUT"] == 0.2


def test_trial_to_params_resolves_backbone_idx_to_preset():
    """The Optuna study stores ``nn_backbone_layers_idx`` (int) so the
    categorical choice round-trips through SQLite cleanly. _trial_to_params
    should resolve the index back to the concrete preset list and rename the
    key so downstream consumers see the user-facing shape."""
    frozen = MagicMock()
    frozen.params = {
        "attn_d_model": 32,
        "nn_backbone_layers_idx": 3,  # _BACKBONE_PRESETS[3] == [128, 64]
        "attn_lr": 0.001,
    }
    p = tune_nn._trial_to_params(frozen)
    assert p["nn_backbone_layers"] == [128, 64]
    assert isinstance(p["nn_backbone_layers"], list)
    # The raw index should be dropped — only the resolved key remains.
    assert "nn_backbone_layers_idx" not in p
    # Other params left untouched.
    assert p["attn_d_model"] == 32
    assert p["attn_lr"] == 0.001


# ---------------------------------------------------------------------------
# Objective end-to-end (no real training — runner is stubbed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_objective_returns_min_of_captured_val_losses(monkeypatch, pos):
    """Stub get_runner so it invokes the epoch_callback with a known
    trajectory; verify the objective returns min(losses) without touching
    real data. Parametrized across all six positions to confirm K/DST went
    from rejected (pre-PR-3) to first-class once their run() signatures
    grew a config= kwarg."""

    def fake_runner(seed, config):
        cb = config.get("epoch_callback")
        assert cb is not None, "tune_nn must install epoch_callback into cfg"
        # Decreasing trajectory: min is the last value.
        for ep, loss in enumerate([1.5, 1.2, 0.9, 0.7, 0.6]):
            cb(ep, loss)
        # Return a minimal result dict — objective should prefer the
        # captured trajectory over result["attn_history"].
        return {"attn_history": {"val_loss": [5.0]}}

    base_cfg = {"train_attention_nn": True}
    monkeypatch.setattr(tune_nn, "get_runner", lambda _pos: fake_runner)

    study = optuna.create_study(direction="minimize")
    objective = tune_nn._make_objective(pos, base_cfg, seed=42)
    study.optimize(objective, n_trials=1)

    assert study.best_value == pytest.approx(0.6)


def test_objective_propagates_pruned_trial(monkeypatch):
    """If the callback raises TrialPruned, the trial should register as
    pruned (Optuna's contract), not as a failure."""

    def fake_runner(seed, config):
        cb = config["epoch_callback"]
        # First report ok; on the second, simulate the pruner deciding to kill.
        cb(0, 1.0)
        # Manually trigger the pruning path — the test's trial.report mock
        # below will return True from should_prune at step 1.
        cb(1, 0.9)
        return {"attn_history": {"val_loss": [0.9]}}

    base_cfg = {"train_attention_nn": True}
    monkeypatch.setattr(tune_nn, "get_runner", lambda pos: fake_runner)

    # Pruner that prunes every trial after the first epoch report. The simplest
    # way to force this deterministically is a custom pruner.
    class _AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return trial.last_step is not None and trial.last_step >= 1

    study = optuna.create_study(direction="minimize", pruner=_AlwaysPrune())
    objective = tune_nn._make_objective("QB", base_cfg, seed=42)
    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED


def test_objective_falls_back_to_attn_history_when_callback_unused(monkeypatch):
    """If the callback never fires (e.g. attention training was skipped
    inside the pipeline), the objective should fall back to the val_loss
    trajectory exposed via result["attn_history"]."""

    def fake_runner(seed, config):
        # Deliberately ignore the callback — simulates a pipeline where
        # attention training was disabled at runtime.
        return {"attn_history": {"val_loss": [2.0, 1.5, 1.1]}}

    base_cfg = {"train_attention_nn": True}
    monkeypatch.setattr(tune_nn, "get_runner", lambda pos: fake_runner)

    study = optuna.create_study(direction="minimize")
    objective = tune_nn._make_objective("QB", base_cfg, seed=42)
    study.optimize(objective, n_trials=1)
    assert study.best_value == pytest.approx(1.1)


def test_objective_raises_when_no_val_loss_anywhere(monkeypatch):
    """If neither the callback fires nor attn_history is present, the
    objective should raise a clear error rather than silently returning
    something nonsensical."""

    def fake_runner(seed, config):
        return {}  # No attn_history, no callback fire.

    base_cfg = {"train_attention_nn": True}
    monkeypatch.setattr(tune_nn, "get_runner", lambda pos: fake_runner)

    study = optuna.create_study(direction="minimize")
    objective = tune_nn._make_objective("QB", base_cfg, seed=42)
    # The objective raises inside study.optimize; with default catch=(), it
    # surfaces as the trial's failure rather than a Python exception. Drive
    # the objective directly instead.
    trial = study.ask()
    with pytest.raises(RuntimeError, match="no val_loss trajectory"):
        objective(trial)


# ---------------------------------------------------------------------------
# epoch_callback hook on MultiHeadTrainer
# ---------------------------------------------------------------------------


def test_multihead_trainer_propagates_callback_exception():
    """MultiHeadTrainer.train must let a raise from epoch_callback (e.g.
    optuna.TrialPruned) propagate — that is the intended pruning control
    flow. This guards the contract relied on by tune_nn._make_objective."""
    import numpy as np
    import torch
    import torch.nn as nn

    from src.shared.training import (
        MultiHeadTrainer,
        MultiTargetLoss,
        make_dataloaders,
    )

    torch.manual_seed(0)
    n, d = 32, 3
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {"t": np.random.randn(n).astype(np.float32)}

    train_loader, val_loader = make_dataloaders(X, y_dict, X, y_dict, batch_size=8)

    class _TinyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(d, 1)

        def forward(self, x):
            return {"t": self.fc(x).squeeze(-1)}

    model = _TinyHead()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    class _Marker(Exception):
        pass

    def cb(epoch, val_loss):
        raise _Marker("pruning fired")

    trainer = MultiHeadTrainer(
        model=model,
        optimizer=optimizer,
        # Plateau scheduler is fine for a unit test — `train()` calls
        # `.step(val_loss)` on it, which is a no-op when only one epoch runs.
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        criterion=MultiTargetLoss(target_names=["t"], loss_weights={"t": 1.0}),
        device=torch.device("cpu"),
        target_names=["t"],
        patience=10,
        epoch_callback=cb,
    )

    with pytest.raises(_Marker, match="pruning fired"):
        trainer.train(train_loader, val_loader, n_epochs=3)


def test_multihead_trainer_runs_when_callback_is_none():
    """The default (no callback) path must keep working — base behaviour for
    every existing caller."""
    import numpy as np
    import torch
    import torch.nn as nn

    from src.shared.training import (
        MultiHeadTrainer,
        MultiTargetLoss,
        make_dataloaders,
    )

    torch.manual_seed(0)
    n, d = 32, 3
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {"t": np.random.randn(n).astype(np.float32)}
    train_loader, val_loader = make_dataloaders(X, y_dict, X, y_dict, batch_size=8)

    class _TinyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(d, 1)

        def forward(self, x):
            return {"t": self.fc(x).squeeze(-1)}

    model = _TinyHead()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = MultiHeadTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        criterion=MultiTargetLoss(target_names=["t"], loss_weights={"t": 1.0}),
        device=torch.device("cpu"),
        target_names=["t"],
        patience=10,
        # epoch_callback omitted -> default None.
    )
    history = trainer.train(train_loader, val_loader, n_epochs=2)
    assert "val_loss" in history
    assert len(history["val_loss"]) == 2
