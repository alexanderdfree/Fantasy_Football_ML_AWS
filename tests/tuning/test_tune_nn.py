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

import os
import sqlite3
from unittest.mock import MagicMock

import optuna
import pytest
import torch

from src.shared.neural_net import build_multihead_net_with_history
from src.shared.platform_detect import PlatformInfo
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


_EXPECTED_BASE_KEYS = {
    "attn_d_model",
    "attn_n_heads",
    "attn_encoder_hidden_dim",
    "attn_dropout",
    "attn_lr",
    "attn_batch_size",
    "scheduler_type",
    "nn_backbone_layers",
    "nn_head_hidden",
    "nn_dropout",
    "nn_lr",
    "nn_weight_decay",
    "nn_batch_size",
}
_COSINE_KEYS = {"cosine_t0", "cosine_t_mult", "cosine_eta_min"}
_ONECYCLE_KEYS = {"onecycle_max_lr", "onecycle_pct_start"}


def _base_valid_overrides(*, scheduler_type: str = "cosine_warm_restarts") -> dict:
    overrides = {
        "attn_d_model": 32,
        "attn_n_heads": 2,
        "attn_encoder_hidden_dim": 0,
        "attn_dropout": 0.1,
        "attn_lr": 1e-3,
        "attn_batch_size": 128,
        "scheduler_type": scheduler_type,
        "nn_backbone_layers": [64],
        "nn_head_hidden": 32,
        "nn_dropout": 0.2,
        "nn_lr": 1e-3,
        "nn_weight_decay": 1e-4,
        "nn_batch_size": 128,
    }
    if scheduler_type == "cosine_warm_restarts":
        overrides.update(
            {
                "cosine_t0": 40,
                "cosine_t_mult": 2,
                "cosine_eta_min": 1e-5,
            }
        )
    elif scheduler_type == "onecycle":
        overrides.update(
            {
                "onecycle_max_lr": 2e-3,
                "onecycle_pct_start": 0.3,
            }
        )
    return overrides


def _platform_info(
    *,
    backend: str,
    os_name: str,
    is_wsl: bool = False,
    gpu_name: str | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> PlatformInfo:
    return PlatformInfo(
        os=os_name,
        is_wsl=is_wsl,
        arch="x86_64" if os_name != "macOS" else "arm64",
        backend=backend,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        sm=f"sm_{compute_capability[0]}{compute_capability[1]}" if compute_capability else None,
        supports_bf16=bool(compute_capability and compute_capability >= (8, 0)),
        cpu_count=4,
        recommended_cuda_wheel="cu126" if compute_capability else None,
    )


def test_sample_overrides_returns_every_cfg_key():
    """Every override key tune_lgbm/pipeline.py reads must be present."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    # Drive a handful of trials so we exercise more of the search space
    # than the first TPE sample (which is effectively random init).
    valid_trials = 0
    for _ in range(30):
        try:
            _, overrides = _ask_overrides(study)
        except optuna.TrialPruned:
            continue
        assert overrides.keys() >= _EXPECTED_BASE_KEYS
        if overrides["scheduler_type"] == "cosine_warm_restarts":
            assert overrides.keys() >= _COSINE_KEYS
            assert not (_ONECYCLE_KEYS & overrides.keys())
        elif overrides["scheduler_type"] == "onecycle":
            assert overrides.keys() >= _ONECYCLE_KEYS
            assert not (_COSINE_KEYS & overrides.keys())
        else:  # pragma: no cover - validation below would fail first
            raise AssertionError(overrides["scheduler_type"])
        tune_nn._validate_overrides(overrides)
        valid_trials += 1
    assert valid_trials > 0


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
        assert o["scheduler_type"] in ("cosine_warm_restarts", "onecycle")
        if o["scheduler_type"] == "cosine_warm_restarts":
            assert o["cosine_t0"] in (10, 20, 30, 40, 60)
            assert o["cosine_t_mult"] in (1, 2)
            assert 1e-6 <= o["cosine_eta_min"] <= 5e-5
            assert o["cosine_eta_min"] < o["attn_lr"]
            assert o["cosine_eta_min"] < o["nn_lr"]
            assert not (_ONECYCLE_KEYS & o.keys())
        else:
            assert 1e-4 <= o["onecycle_max_lr"] <= 1e-2
            assert 0.1 <= o["onecycle_pct_start"] <= 0.4
            assert not (_COSINE_KEYS & o.keys())
        assert 0.0 <= o["nn_dropout"] <= 0.4
        assert 1e-4 <= o["nn_lr"] <= 5e-3
        assert 1e-5 <= o["nn_weight_decay"] <= 1e-3
        assert o["attn_batch_size"] in (128, 256, 512)
        assert o["nn_batch_size"] in (128, 256, 512)
        assert o["attn_d_model"] in (16, 24, 32, 48, 64)
        assert o["attn_d_model"] > 0
        assert o["attn_n_heads"] in (1, 2, 4)
        assert o["attn_n_heads"] > 0
        assert o["nn_head_hidden"] in (16, 24, 32, 48, 64)
        assert o["nn_head_hidden"] > 0
        assert o["attn_encoder_hidden_dim"] in (0, 16, 32, 64)
        assert o["attn_encoder_hidden_dim"] == 0 or o["attn_encoder_hidden_dim"] > 0
        assert isinstance(o["nn_backbone_layers"], list)
        assert all(isinstance(v, int) for v in o["nn_backbone_layers"])
        assert all(v > 0 for v in o["nn_backbone_layers"])
        tune_nn._validate_overrides(o)


@pytest.mark.parametrize(
    ("bad_update", "match"),
    [
        ({"nn_head_hidden": 0}, "nn_head_hidden"),
        ({"nn_backbone_layers": [0]}, "nn_backbone_layers"),
    ],
)
def test_validate_overrides_rejects_nonpositive_model_dimensions(bad_update, match):
    overrides = _base_valid_overrides()
    overrides.update(bad_update)

    with pytest.raises(ValueError, match=match):
        tune_nn._validate_overrides(overrides)


def test_validate_overrides_allows_only_encoder_hidden_zero_sentinel():
    overrides = _base_valid_overrides()
    tune_nn._validate_overrides(overrides)

    for key in ("attn_d_model", "attn_n_heads", "nn_head_hidden", "attn_batch_size"):
        bad = _base_valid_overrides()
        bad[key] = 0
        with pytest.raises(ValueError, match=key):
            tune_nn._validate_overrides(bad)


@pytest.mark.parametrize(
    ("bad_update", "match"),
    [
        ({"scheduler_type": "plateau"}, "scheduler_type"),
        ({"cosine_t0": 0}, "cosine_t0"),
        ({"cosine_t_mult": 0}, "cosine_t_mult"),
        ({"cosine_eta_min": 1e-2}, "cosine_eta_min"),
        ({"onecycle_max_lr": 0.0}, "irrelevant scheduler keys"),
        ({"unexpected": 1}, "unknown keys"),
    ],
)
def test_validate_overrides_rejects_bad_cosine_scheduler_configs(bad_update, match):
    overrides = _base_valid_overrides()
    overrides.update(bad_update)

    with pytest.raises(ValueError, match=match):
        tune_nn._validate_overrides(overrides)


@pytest.mark.parametrize(
    ("bad_update", "match"),
    [
        ({"onecycle_max_lr": 0.0}, "onecycle_max_lr"),
        ({"onecycle_pct_start": 0.0}, "onecycle_pct_start"),
        ({"cosine_t0": 40}, "irrelevant scheduler keys"),
    ],
)
def test_validate_overrides_rejects_bad_onecycle_scheduler_configs(bad_update, match):
    overrides = _base_valid_overrides(scheduler_type="onecycle")
    overrides.update(bad_update)

    with pytest.raises(ValueError, match=match):
        tune_nn._validate_overrides(overrides)


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


def test_sampled_overrides_build_attention_model_and_forward():
    """Sampled configs should construct the attention model and produce finite
    outputs before a real tuning run spends epochs on them."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=3))
    sampled: list[dict] = []
    while len(sampled) < 4:
        try:
            _, overrides = _ask_overrides(study)
        except optuna.TrialPruned:
            continue
        tune_nn._validate_overrides(overrides)
        sampled.append(overrides)

    targets = ["target_a", "target_b"]
    x_static = torch.randn(3, 5)
    x_history = torch.randn(3, 4, 2)
    history_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, False, False, False],
        ]
    )

    for overrides in sampled:
        model = build_multihead_net_with_history(
            overrides,
            static_dim=x_static.shape[1],
            game_dim=x_history.shape[2],
            targets=targets,
        )
        model.eval()
        with torch.no_grad():
            preds = model(x_static, x_history, history_mask)

        assert set(preds) == set(targets)
        for pred in preds.values():
            assert pred.shape == (len(x_static),)
            assert torch.isfinite(pred).all()


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
        "scheduler_type": "onecycle",
        "onecycle_max_lr": 0.002,
        "onecycle_pct_start": 0.3,
        "nn_backbone_layers": [128, 64],
        "nn_head_hidden": 32,
        "nn_dropout": 0.2,
        "nn_lr": 0.0005,
        "nn_weight_decay": 0.0001,
        "nn_batch_size": 256,
    }
    rendered = tune_nn._format_config_lines("RB", best)
    # Comment lines start with '#'; the rest are `    <kwarg>=<value>,` lines
    # meant to paste into a PositionConfig(...) call. Wrap them in dict(...) so
    # they round-trip back to a params dict (and prove they're valid Python).
    kwarg_lines = [
        ln for ln in rendered.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    cfg = eval("dict(\n" + "\n".join(kwarg_lines) + "\n)")  # noqa: S307 — test-only
    # Verify a representative subset round-trips correctly, attention-namespaced
    # (attn_*, not the shared scheduler_*/onecycle_* the regular NN reads, #792).
    assert cfg["attn_d_model"] == 32
    assert cfg["attn_lr"] == 0.001
    assert cfg["attn_scheduler_type"] == "onecycle"
    assert cfg["attn_onecycle_max_lr"] == 0.002
    assert cfg["attn_onecycle_pct_start"] == 0.3
    assert cfg["nn_backbone_layers"] == [128, 64]
    assert cfg["nn_dropout"] == 0.2
    # The retired UPPER_CASE module-constant form must be gone (#826).
    assert "RB_ATTN_D_MODEL" not in rendered
    assert "attn_d_model=32," in rendered


def test_trial_to_params_resolves_backbone_idx_to_preset():
    """The Optuna study stores ``nn_backbone_layers_idx`` (int) so the
    categorical choice round-trips through SQLite cleanly. _trial_to_params
    should resolve the index back to the concrete preset list and rename the
    key so downstream consumers see the user-facing shape."""
    frozen = MagicMock()
    frozen.params = _base_valid_overrides()
    frozen.params.pop("nn_backbone_layers")
    frozen.params["nn_backbone_layers_idx"] = 3  # _BACKBONE_PRESETS[3] == [128, 64]
    p = tune_nn._trial_to_params(frozen)
    assert p["nn_backbone_layers"] == [128, 64]
    assert isinstance(p["nn_backbone_layers"], list)
    # The raw index should be dropped — only the resolved key remains.
    assert "nn_backbone_layers_idx" not in p
    # Other params left untouched.
    assert p["attn_d_model"] == 32
    assert p["attn_lr"] == 0.001


def test_trial_to_params_rejects_stale_invalid_best_trial():
    frozen = MagicMock()
    frozen.params = _base_valid_overrides()
    frozen.params["nn_head_hidden"] = 0

    with pytest.raises(ValueError, match="nn_head_hidden"):
        tune_nn._trial_to_params(frozen)


def test_trial_to_params_rejects_unknown_backbone_preset_index():
    frozen = MagicMock()
    frozen.params = _base_valid_overrides()
    frozen.params.pop("nn_backbone_layers")
    frozen.params["nn_backbone_layers_idx"] = 999

    with pytest.raises(ValueError, match="nn_backbone_layers_idx"):
        tune_nn._trial_to_params(frozen)


def test_trial_to_params_rejects_stale_scheduler_mismatch():
    frozen = MagicMock()
    frozen.params = _base_valid_overrides(scheduler_type="onecycle")
    frozen.params["cosine_t0"] = 40

    with pytest.raises(ValueError, match="irrelevant scheduler keys"):
        tune_nn._trial_to_params(frozen)


def test_study_storage_is_versioned_for_scheduler_search_space():
    assert tune_nn._study_name("RB") == "nn_scheduler_v2_rb"
    assert tune_nn._study_db_path("RB") == "tune_nn_scheduler_v2_rb.db"
    assert tune_nn._s3_key_prefix("RB") == "tune_nn/scheduler_v2/rb"


def test_mps_graph_storage_profile_is_separate_from_eager():
    version = tune_nn._resolve_search_space_version("mps", cuda_graph=True)
    assert version == "scheduler_v2_mps_graph"
    assert tune_nn._study_name("RB", version) == "nn_scheduler_v2_mps_graph_rb"
    assert tune_nn._study_db_path("RB", version) == "tune_nn_scheduler_v2_mps_graph_rb.db"
    assert tune_nn._s3_key_prefix("RB", version) == "tune_nn/scheduler_v2_mps_graph/rb"


# ---------------------------------------------------------------------------
# _worker_sampler_seed — MPS worker startup-phase draws
# ---------------------------------------------------------------------------


def test_worker_sampler_seed_unique_across_workers_and_iterations():
    seeds = {tune_nn._worker_sampler_seed(w, i) for w in range(8) for i in range(50)}
    assert len(seeds) == 8 * 50


def test_mps_worker_iterations_draw_distinct_startup_params(tmp_path, monkeypatch):
    """Two _mps_worker_entry loop iterations must not replay the same TPE
    random-startup draw: the sampler is rebuilt fresh each iteration, so a
    fixed per-worker seed redraws the identical point (the duplicate-trial
    bug observed on Batch)."""
    monkeypatch.chdir(tmp_path)  # study db path is CWD-relative
    params: list[dict] = []
    for iteration in range(2):
        study = tune_nn._create_or_load_study(
            "RB",
            storage_version="seedtest",
            sqlite_timeout=5,
            base_cfg={"nn_epochs": 10},
            sampler_seed=tune_nn._worker_sampler_seed(0, iteration),
        )
        trial, _ = _ask_overrides(study)
        study.tell(trial, 60.0)
        params.append(trial.params)
    assert params[0] != params[1]


def test_auto_parallel_backend_selects_mps_only_on_g6_l4_linux(monkeypatch):
    monkeypatch.setattr(
        tune_nn,
        "detect_platform",
        lambda: _platform_info(
            backend="cuda",
            os_name="Linux",
            gpu_name="NVIDIA L4",
            compute_capability=(8, 9),
        ),
    )
    assert tune_nn._resolve_parallel_backend("auto") == "mps"


@pytest.mark.parametrize(
    "info",
    [
        _platform_info(backend="mps", os_name="macOS", gpu_name="Apple MPS"),
        _platform_info(
            backend="cuda",
            os_name="Linux",
            is_wsl=True,
            gpu_name="NVIDIA GeForce RTX 5080",
            compute_capability=(12, 0),
        ),
        _platform_info(
            backend="cuda",
            os_name="Linux",
            gpu_name="NVIDIA GeForce RTX 5080",
            compute_capability=(12, 0),
        ),
    ],
)
def test_auto_parallel_backend_preserves_mac_and_5080_paths(monkeypatch, info):
    monkeypatch.setattr(tune_nn, "detect_platform", lambda: info)
    assert tune_nn._resolve_parallel_backend("auto") == "thread"


def test_sqlite_backup_captures_wal_state(tmp_path):
    db_path = tmp_path / "study.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("create table t (v integer)")
    conn.execute("insert into t values (7)")
    conn.commit()
    conn.close()

    backup_path = tune_nn._sqlite_backup(str(db_path))
    try:
        copy = sqlite3.connect(backup_path)
        try:
            assert copy.execute("select v from t").fetchone() == (7,)
        finally:
            copy.close()
    finally:
        os.remove(backup_path)


def test_mps_context_fails_loudly_when_binary_missing(monkeypatch):
    monkeypatch.setattr(tune_nn.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="nvidia-cuda-mps-control"):
        with tune_nn._NvidiaMPS(enabled=True):
            pass


def test_mps_context_noop_when_disabled(monkeypatch):
    called = False

    def fake_which(name):
        nonlocal called
        called = True
        return "/usr/bin/nvidia-cuda-mps-control"

    monkeypatch.setattr(tune_nn.shutil, "which", fake_which)
    with tune_nn._NvidiaMPS(enabled=False):
        pass
    assert called is False


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


def test_objective_validates_overrides_before_training(monkeypatch):
    def fail_runner(seed, config):  # pragma: no cover - assertion is that this is never called
        raise AssertionError("runner should not be called for invalid overrides")

    bad_overrides = _base_valid_overrides()
    bad_overrides["nn_head_hidden"] = 0

    monkeypatch.setattr(tune_nn, "_sample_overrides", lambda trial: bad_overrides)
    monkeypatch.setattr(tune_nn, "get_runner", lambda pos: fail_runner)

    study = optuna.create_study(direction="minimize")
    objective = tune_nn._make_objective("QB", {"train_attention_nn": True}, seed=42)
    trial = study.ask()

    with pytest.raises(ValueError, match="nn_head_hidden"):
        objective(trial)


@pytest.mark.parametrize(
    ("scheduler_type", "sampled_key", "attn_key", "base_value", "sampled_value"),
    [
        ("cosine_warm_restarts", "cosine_eta_min", "attn_cosine_eta_min", 2e-5, 3e-5),
        ("onecycle", "onecycle_max_lr", "attn_onecycle_max_lr", 4e-3, 7e-3),
    ],
)
def test_objective_maps_sampled_scheduler_lr_to_attention_override(
    monkeypatch,
    scheduler_type,
    sampled_key,
    attn_key,
    base_value,
    sampled_value,
):
    overrides = _base_valid_overrides(scheduler_type=scheduler_type)
    overrides[sampled_key] = sampled_value

    def fake_runner(seed, config):
        assert config[attn_key] == pytest.approx(sampled_value)
        assert config[attn_key] != pytest.approx(base_value)
        config["epoch_callback"](0, 0.5)
        return {"attn_history": {"val_loss": [0.5]}}

    base_cfg = {"train_attention_nn": True, "scheduler_type": scheduler_type, attn_key: base_value}
    monkeypatch.setattr(tune_nn, "_sample_overrides", lambda trial: dict(overrides))
    monkeypatch.setattr(tune_nn, "get_runner", lambda pos: fake_runner)

    study = optuna.create_study(direction="minimize")
    objective = tune_nn._make_objective("QB", base_cfg, seed=42)
    study.optimize(objective, n_trials=1)

    assert study.best_value == pytest.approx(0.5)


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


# ---------------------------------------------------------------------------
# RAM-aware n_jobs clamp (container OOM guardrail)
# ---------------------------------------------------------------------------

_BATCH_JOB_LIMIT_BYTES = 15000 * 1024**2  # the 15000-MiB ff-training-job shape


def test_cgroup_limit_reads_v2_value(tmp_path):
    v2 = tmp_path / "memory.max"
    v2.write_text(f"{_BATCH_JOB_LIMIT_BYTES}\n")
    assert (
        tune_nn._cgroup_memory_limit_bytes(v2_path=str(v2), v1_path=str(tmp_path / "absent"))
        == _BATCH_JOB_LIMIT_BYTES
    )


def test_cgroup_limit_v2_max_means_unlimited(tmp_path):
    v2 = tmp_path / "memory.max"
    v2.write_text("max\n")
    assert (
        tune_nn._cgroup_memory_limit_bytes(v2_path=str(v2), v1_path=str(tmp_path / "absent"))
        is None
    )


def test_cgroup_limit_falls_back_to_v1(tmp_path):
    v1 = tmp_path / "memory.limit_in_bytes"
    v1.write_text(f"{_BATCH_JOB_LIMIT_BYTES}\n")
    assert (
        tune_nn._cgroup_memory_limit_bytes(v2_path=str(tmp_path / "absent"), v1_path=str(v1))
        == _BATCH_JOB_LIMIT_BYTES
    )


def test_cgroup_limit_v1_no_limit_sentinel_means_unlimited(tmp_path):
    v1 = tmp_path / "memory.limit_in_bytes"
    v1.write_text("9223372036854771712\n")  # v1 "unlimited" (2**63 page-rounded)
    assert (
        tune_nn._cgroup_memory_limit_bytes(v2_path=str(tmp_path / "absent"), v1_path=str(v1))
        is None
    )


def test_cgroup_limit_none_when_files_absent(tmp_path):
    assert (
        tune_nn._cgroup_memory_limit_bytes(
            v2_path=str(tmp_path / "absent"), v1_path=str(tmp_path / "also-absent")
        )
        is None
    )


def test_ram_safe_n_jobs_no_limit_passes_through():
    assert tune_nn._ram_safe_n_jobs(32, None) == (32, None)


def test_ram_safe_n_jobs_single_worker_never_clamped():
    # n_jobs=1 must run even in a tiny container — clamping to zero would
    # deadlock the study.
    assert tune_nn._ram_safe_n_jobs(1, 1 * 1024**3) == (1, None)


def test_ram_safe_n_jobs_fits_unchanged_no_warning():
    # The validated Batch config: n_jobs=4 on the 15000-MiB job shape.
    assert tune_nn._ram_safe_n_jobs(4, _BATCH_JOB_LIMIT_BYTES) == (4, None)


@pytest.mark.parametrize("requested", [8, 32])
def test_ram_safe_n_jobs_clamps_overcommit_on_batch_shape(requested):
    # Both observed OOM cases (8 mid-run, 32 at startup) clamp to the same
    # fit: (15000 MiB - 1 GiB reserve) // 2 GiB = 6 workers.
    clamped, warning = tune_nn._ram_safe_n_jobs(requested, _BATCH_JOB_LIMIT_BYTES)
    assert clamped == 6
    assert warning is not None
    assert f"clamping n_jobs {requested} -> 6" in warning
    assert "OOM-killed" in warning


def test_ram_safe_n_jobs_floor_is_one_worker():
    # A container too small for even one estimated worker still gets one —
    # the guardrail degrades to "try anyway", never to zero workers.
    clamped, warning = tune_nn._ram_safe_n_jobs(4, 2 * 1024**3)
    assert clamped == 1
    assert warning is not None
