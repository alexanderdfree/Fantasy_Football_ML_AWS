"""Optuna-based hyperparameter tuning for the attention NN, per position.

Usage:
    python -m src.tuning.tune_nn QB                     # tune one position
    python -m src.tuning.tune_nn QB RB WR TE            # multiple (sequential)
    python -m src.tuning.tune_nn RB --n-trials 30
    python -m src.tuning.tune_nn RB --timeout 7200      # seconds, per position
    python -m src.tuning.tune_nn RB --n-jobs 3          # concurrent trials (GPU-bound)
    python -m src.tuning.tune_nn RB --print-best        # inspect saved study

MVP scope (v1)
--------------
Mirrors src/tuning/tune_lgbm.py's shape: per-position SQLite study, paste-ready
config output via `_format_config_lines`, BEST_PARAMS_JSON markers for CI log
capture. Differences vs the LightGBM tuner:

* Search space targets the **attention NN** (architecture + optimizer + scheduler knobs).
* Pruner is HyperbandPruner, fed by the `epoch_callback` hook on
  `MultiHeadTrainer` — kills clearly-bad trials at low epoch counts.
* Trial objective is `min(result["attn_history"]["val_loss"])` — val-only, no
  test contamination. The pipeline's `result["attn_nn_metrics"]` is test-set
  MAE and would leak into the search; we deliberately don't use it.
* Single train/val/test split per trial (no CV). 30 trials × CV folds × full
  NN training is too slow for laptop validation; CV is v2.

Out of scope (v1)
-----------------
* **Loss-config search** (`head_losses`, `loss_weights`, `huber_deltas`,
  `gated_targets`): per CLAUDE.md, `LOSS_WEIGHTS ≈ 2.0 / HUBER_DELTAS` is a
  coupling, not two independent axes. Searching deltas + deriving weights also
  blows up search dimensionality past what ~30 trials resolve. Hand-tune loss
  config via the `ablate_rb_gate.py` pattern.
* **`ATTN_STATIC_FEATURES` / `ATTN_HISTORY_STATS`**: structural feature
  choices, not hyperparams. CLAUDE.md's stop-rule on rolling-features-in-
  static still applies.
* **`nn_non_negative_targets`**: per-head correctness constraint.

Batch follow-up
---------------
`src/tuning/launch_tune.py` and `.github/workflows/retune-nn-batch.yml` will
fan out one Spot g6.xlarge per position (matching `train-batch.yml`). The
container dispatches to this script via a new `--mode=tune` flag added to
`src/batch/train.py` — the Batch job's `command` becomes:

    ["--position", "RB", "--mode", "tune", "--n-trials", "30"]

instead of the current

    ["--position", "RB", "--seed", "42"].

A `--checkpoint-s3` flag added here will periodically upload the SQLite study
DB to `s3://$S3_BUCKET/tune_nn/{search-space-version}/{pos}/` and trap SIGTERM
so a Spot interruption can resume the search on Batch's retry.

The Batch launcher passes `--parallel-backend auto`: native-Linux L4/g6 hosts
resolve to NVIDIA MPS; Mac/MPS and RTX 5080 local hosts keep the historical
thread backend unless an operator explicitly forces `--parallel-backend mps`.
"""

import argparse
import contextlib
import copy
import json
import multiprocessing as mp
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.config import SPLITS_DIR
from src.shared.core_pool import ENV_ADDR as _CORE_POOL_ADDR_ENV
from src.shared.core_pool import ENV_POS as _CORE_POOL_POS_ENV
from src.shared.core_pool import lease_cores as _lease_cores
from src.shared.core_pool import start_coordinator as _start_core_pool
from src.shared.platform_detect import detect_platform
from src.shared.registry import get_config, get_runner
from src.tuning.history import append_tuning_run
from src.tuning.tune_nn_storage import (
    SEARCH_SPACE_VERSION as _DEFAULT_SEARCH_SPACE_VERSION,
)
from src.tuning.tune_nn_storage import (
    resolve_search_space_version as _resolve_search_space_version,
)
from src.tuning.tune_nn_storage import (
    s3_key_prefix as _s3_key_prefix,
)
from src.tuning.tune_nn_storage import (
    study_db_path as _study_db_path,
)
from src.tuning.tune_nn_storage import (
    study_name as _study_name,
)


def _ensure_data_from_s3() -> None:
    """Bootstrap data/splits/ + data/raw/ from S3 when ``S3_BUCKET`` is set
    and the local files are missing. Mirrors ``src/tuning/tune_lgbm.py::
    _ensure_data_from_s3`` so the Batch container needs no extra plumbing —
    just set ``S3_BUCKET`` and (optionally) ``S3_DATA_PREFIX`` in the job
    environment, same convention as the training path.

    No-op locally without ``S3_BUCKET``; the local CLI form expects the
    parquets already on disk (either via the worktree symlink pattern in
    [[feedback_worktree_data_symlink]] or a real local pull).
    """
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return
    prefix = os.environ.get("S3_DATA_PREFIX", "data")

    # Local import — boto3 is only needed for the S3 path, which is gated on
    # S3_BUCKET. Lets ``--print-best`` and the local CLI run on machines
    # without boto3 or full src/batch/train deps.
    from src.batch.train import download_data, sync_raw_data

    splits_needed = not all(
        os.path.exists(os.path.join(SPLITS_DIR, f))
        for f in ("train.parquet", "val.parquet", "test.parquet")
    )
    if splits_needed:
        print(f"[tune_nn] Downloading splits from s3://{bucket}/{prefix}/ to {SPLITS_DIR}/")
        download_data(bucket, prefix, SPLITS_DIR)
    else:
        print(f"[tune_nn] Splits already present at {SPLITS_DIR}/")

    if not os.path.isdir("data/raw") or not any(
        f.endswith(".parquet") for f in os.listdir("data/raw")
    ):
        print(f"[tune_nn] Syncing data/raw/ from s3://{bucket}/data/raw/")
        sync_raw_data(bucket)
    else:
        print("[tune_nn] data/raw/ already populated")


# 15 trials per position is a laptop-friendly default; bump to 30 on Batch
# g6.xlarge. Per-trial cost dropped substantially after 2026-05-21 from the
# FP16 / sync-removal / GPU-resident-dataset wave and from skipping Ridge /
# LGBM / base NN inside trials (see _make_objective below); the original
# "5–10 min locally, 1–2 min on Batch" design figures are stale — remeasure
# rather than relying on them.
_DEFAULT_N_TRIALS = 15
_DEFAULT_PARALLEL_BACKEND = "thread"
_DEFAULT_N_JOBS = 2
_DEFAULT_SQLITE_TIMEOUT_SECONDS = 120
_DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 60
_AUTO_BACKEND = "auto"
_MPS_BACKEND = "mps"
_THREAD_BACKEND = "thread"
_PARALLEL_BACKENDS = (_THREAD_BACKEND, _MPS_BACKEND, _AUTO_BACKEND)
_TRUTHY = {"1", "true", "yes", "on"}

# HyperbandPruner: `min_resource` is the minimum epoch count a trial must
# complete before it's eligible for pruning. We pick 8 to give the trial a
# chance to escape its lr-warmup transient before being judged. `reduction
# _factor=3` is Optuna's default. `max_resource` is pinned to the cfg's
# ``nn_epochs`` per study (instead of Optuna's `"auto"`) so an early-
# stopped trial 1 can't establish a too-low rung ladder that prunes every
# later trial at the wrong epoch — `"auto"` infers from completed trials,
# which silently underestimates when ``trainer.train`` exits at ``patience``
# rather than the configured epoch ceiling.
_HYPERBAND_MIN_RESOURCE = 8
_HYPERBAND_REDUCTION_FACTOR = 3
# Backbone-layer presets keyed by an integer index. Optuna's persistent storage
# (SQLite, JSON) only round-trips scalar categorical choices (None/bool/int/
# float/str); a tuple like ``(128, 64)`` triggers a UserWarning and may not
# survive a study reload. Storing an index and resolving to the preset inside
# ``_sample_overrides`` / ``_trial_to_params`` keeps the search space stable
# across reloads.
_BACKBONE_PRESETS: list[list[int]] = [
    [64],
    [128],
    [96, 48],
    [128, 64],
    [128, 32],
]

_BASE_TUNED_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
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
)
_SCHEDULER_PARAM_KEYS: dict[str, frozenset[str]] = {
    "cosine_warm_restarts": frozenset({"cosine_t0", "cosine_t_mult", "cosine_eta_min"}),
    "onecycle": frozenset({"onecycle_max_lr", "onecycle_pct_start"}),
}
_TUNED_OVERRIDE_KEYS: frozenset[str] = _BASE_TUNED_OVERRIDE_KEYS | frozenset().union(
    *_SCHEDULER_PARAM_KEYS.values()
)


# ---------------------------------------------------------------------------
# Storage / process backend helpers
# ---------------------------------------------------------------------------


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _is_g6_l4_linux(platform_info) -> bool:
    gpu_name = (platform_info.gpu_name or "").lower()
    return (
        platform_info.backend == "cuda"
        and platform_info.os == "Linux"
        and not platform_info.is_wsl
        and (platform_info.compute_capability == (8, 9) or "l4" in gpu_name)
    )


def _resolve_parallel_backend(requested: str) -> str:
    """Resolve auto without changing local 5080/Mac defaults."""
    if requested != _AUTO_BACKEND:
        return requested
    info = detect_platform()
    if _is_g6_l4_linux(info):
        print(f"[tune_nn] parallel backend auto -> mps ({info.summary()})", flush=True)
        return _MPS_BACKEND
    print(f"[tune_nn] parallel backend auto -> thread ({info.summary()})", flush=True)
    return _THREAD_BACKEND


def _make_storage(db_path: str, sqlite_timeout: int = _DEFAULT_SQLITE_TIMEOUT_SECONDS):
    """Optuna RDB storage with a SQLite busy timeout for concurrent workers."""
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={
            "connect_args": {"timeout": int(sqlite_timeout)},
            "pool_pre_ping": True,
        },
    )


def _configure_sqlite_for_parallel(
    db_path: str, sqlite_timeout: int = _DEFAULT_SQLITE_TIMEOUT_SECONDS
) -> None:
    """Make a local SQLite study DB friendlier to multi-process workers."""
    timeout_ms = int(sqlite_timeout * 1000)
    conn = sqlite3.connect(db_path, timeout=sqlite_timeout)
    try:
        conn.execute(f"PRAGMA busy_timeout={timeout_ms}")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    finally:
        conn.close()


def _study_state_counts(study: optuna.Study) -> dict[str, int]:
    counts = {state.name.lower(): 0 for state in optuna.trial.TrialState}
    for trial in study.trials:
        counts[trial.state.name.lower()] = counts.get(trial.state.name.lower(), 0) + 1
    counts["total"] = len(study.trials)
    return counts


def _completed_trials(study: optuna.Study) -> int:
    return _study_state_counts(study).get("complete", 0)


def _current_cpu_ids() -> list[int]:
    with contextlib.suppress(AttributeError, OSError):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


class _NvidiaMPS:
    """Per-job NVIDIA CUDA MPS daemon wrapper.

    Batch tune jobs run a single container per g6.xlarge. Starting MPS inside
    that container gives the worker subprocesses one shared CUDA scheduling
    context without changing the outer six-position AWS Batch fan-out.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.pipe_dir: str | None = None
        self.log_dir: str | None = None
        self._old_pipe: str | None = None
        self._old_log: str | None = None

    def __enter__(self):
        if not self.enabled:
            return self
        if shutil.which("nvidia-cuda-mps-control") is None:
            raise RuntimeError(
                "parallel-backend=mps requires nvidia-cuda-mps-control in the Batch "
                "container. Confirm the ECS GPU AMI exposes the NVIDIA MPS utilities."
            )
        self.pipe_dir = tempfile.mkdtemp(prefix="ff-mps-pipe-", dir="/tmp")
        self.log_dir = tempfile.mkdtemp(prefix="ff-mps-log-", dir="/tmp")
        self._old_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        self._old_log = os.environ.get("CUDA_MPS_LOG_DIRECTORY")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = self.log_dir
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
        print(f"[mps] started nvidia-cuda-mps-control (pipe={self.pipe_dir})", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        with contextlib.suppress(Exception):
            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="quit\n",
                text=True,
                check=False,
                timeout=10,
            )
        if self._old_pipe is None:
            os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        else:
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = self._old_pipe
        if self._old_log is None:
            os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
        else:
            os.environ["CUDA_MPS_LOG_DIRECTORY"] = self._old_log
        for path in (self.pipe_dir, self.log_dir):
            if path:
                shutil.rmtree(path, ignore_errors=True)
        print("[mps] stopped nvidia-cuda-mps-control", flush=True)
        return False


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0


def _validate_overrides(overrides: dict) -> None:
    """Validate sampled tune_nn overrides before training or reporting them.

    ``attn_encoder_hidden_dim == 0`` is the one intentional zero sentinel: it
    selects the single-layer game encoder in ``_build_game_encoder``. Every real
    dimension, width, batch size, and optimizer scale must be positive.
    """
    errors: list[str] = []

    unknown = sorted(set(overrides) - _TUNED_OVERRIDE_KEYS)
    if unknown:
        errors.append(f"unknown keys: {unknown}")

    missing = sorted(_BASE_TUNED_OVERRIDE_KEYS - overrides.keys())
    if missing:
        errors.append(f"missing keys: {missing}")

    d_model = overrides.get("attn_d_model")
    n_heads = overrides.get("attn_n_heads")
    if not _is_positive_int(d_model):
        errors.append("attn_d_model must be a positive int")
    if not _is_positive_int(n_heads):
        errors.append("attn_n_heads must be a positive int")
    if _is_positive_int(d_model) and _is_positive_int(n_heads) and d_model % n_heads != 0:
        errors.append("attn_d_model must be divisible by attn_n_heads")

    encoder_hidden = overrides.get("attn_encoder_hidden_dim")
    if (
        not isinstance(encoder_hidden, int)
        or isinstance(encoder_hidden, bool)
        or encoder_hidden < 0
    ):
        errors.append(
            "attn_encoder_hidden_dim must be 0 (single-layer encoder sentinel) or a positive int"
        )

    backbone_layers = overrides.get("nn_backbone_layers")
    if not isinstance(backbone_layers, list) or not backbone_layers:
        errors.append("nn_backbone_layers must be a non-empty list of positive ints")
    elif any(not _is_positive_int(v) for v in backbone_layers):
        errors.append("nn_backbone_layers entries must be positive ints")

    if not _is_positive_int(overrides.get("nn_head_hidden")):
        errors.append("nn_head_hidden must be a positive int")

    for key in ("attn_dropout", "nn_dropout"):
        value = overrides.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value < 1.0:
            errors.append(f"{key} must be in [0, 1)")

    for key in ("attn_lr", "nn_lr", "nn_weight_decay"):
        if not _is_positive_number(overrides.get(key)):
            errors.append(f"{key} must be positive")

    for key in ("attn_batch_size", "nn_batch_size"):
        if not _is_positive_int(overrides.get(key)):
            errors.append(f"{key} must be a positive int")

    sched_type = overrides.get("scheduler_type")
    if sched_type not in _SCHEDULER_PARAM_KEYS:
        errors.append(
            f"scheduler_type must be one of {sorted(_SCHEDULER_PARAM_KEYS)}, got {sched_type!r}"
        )
    else:
        required = _SCHEDULER_PARAM_KEYS[sched_type]
        missing_sched = sorted(required - overrides.keys())
        if missing_sched:
            errors.append(f"{sched_type} missing scheduler keys: {missing_sched}")
        irrelevant = sorted(
            (set().union(*_SCHEDULER_PARAM_KEYS.values()) - required) & overrides.keys()
        )
        if irrelevant:
            errors.append(f"{sched_type} overrides include irrelevant scheduler keys: {irrelevant}")

        if sched_type == "cosine_warm_restarts":
            for key in ("cosine_t0", "cosine_t_mult"):
                if not _is_positive_int(overrides.get(key)):
                    errors.append(f"{key} must be a positive int")
            eta_min = overrides.get("cosine_eta_min")
            if not _is_positive_number(eta_min):
                errors.append("cosine_eta_min must be positive")
            elif _is_positive_number(overrides.get("attn_lr")) and eta_min >= overrides["attn_lr"]:
                errors.append("cosine_eta_min must be less than attn_lr")
            elif _is_positive_number(overrides.get("nn_lr")) and eta_min >= overrides["nn_lr"]:
                errors.append("cosine_eta_min must be less than nn_lr")
        elif sched_type == "onecycle":
            max_lr = overrides.get("onecycle_max_lr")
            pct_start = overrides.get("onecycle_pct_start")
            if not _is_positive_number(max_lr):
                errors.append("onecycle_max_lr must be positive")
            if (
                not isinstance(pct_start, (int, float))
                or isinstance(pct_start, bool)
                or not 0.0 < pct_start < 1.0
            ):
                errors.append("onecycle_pct_start must be in (0, 1)")

    if errors:
        raise ValueError("Invalid tune_nn overrides: " + "; ".join(errors))


def _sample_overrides(trial: optuna.Trial) -> dict:
    """Sample one trial's cfg overrides. Raises ``optuna.TrialPruned`` for
    invalid combinations (e.g. ``n_heads`` not dividing ``d_model``).
    """
    d_model = trial.suggest_categorical("attn_d_model", [16, 24, 32, 48, 64])
    n_heads = trial.suggest_categorical("attn_n_heads", [1, 2, 4])
    if _is_positive_int(d_model) and _is_positive_int(n_heads) and d_model % n_heads != 0:
        raise optuna.TrialPruned()

    backbone_idx = trial.suggest_categorical(
        "nn_backbone_layers_idx", list(range(len(_BACKBONE_PRESETS)))
    )

    attn_lr = trial.suggest_float("attn_lr", 1e-4, 5e-3, log=True)
    nn_lr = trial.suggest_float("nn_lr", 1e-4, 5e-3, log=True)
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", ["cosine_warm_restarts", "onecycle"]
    )
    scheduler_overrides: dict
    if scheduler_type == "cosine_warm_restarts":
        scheduler_overrides = {
            "cosine_t0": trial.suggest_categorical("cosine_t0", [10, 20, 30, 40, 60]),
            "cosine_t_mult": trial.suggest_categorical("cosine_t_mult", [1, 2]),
            "cosine_eta_min": trial.suggest_float("cosine_eta_min", 1e-6, 5e-5, log=True),
        }
    else:
        onecycle_max_lr = trial.suggest_float("onecycle_max_lr", 1e-4, 1e-2, log=True)
        scheduler_overrides = {
            "onecycle_max_lr": onecycle_max_lr,
            "onecycle_pct_start": trial.suggest_float("onecycle_pct_start", 0.1, 0.4),
        }

    return {
        "attn_d_model": d_model,
        "attn_n_heads": n_heads,
        "attn_encoder_hidden_dim": trial.suggest_categorical(
            "attn_encoder_hidden_dim", [0, 16, 32, 64]
        ),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
        "attn_lr": attn_lr,
        "attn_batch_size": trial.suggest_categorical("attn_batch_size", [128, 256, 512]),
        "scheduler_type": scheduler_type,
        **scheduler_overrides,
        "nn_backbone_layers": list(_BACKBONE_PRESETS[backbone_idx]),
        "nn_head_hidden": trial.suggest_categorical("nn_head_hidden", [16, 24, 32, 48, 64]),
        "nn_dropout": trial.suggest_float("nn_dropout", 0.0, 0.4),
        "nn_lr": nn_lr,
        "nn_weight_decay": trial.suggest_float("nn_weight_decay", 1e-5, 1e-3, log=True),
        "nn_batch_size": trial.suggest_categorical("nn_batch_size", [128, 256, 512]),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _make_objective(pos: str, base_cfg: dict, seed: int):
    """Return the Optuna objective for ``pos``.

    Each trial:
      1. Samples cfg overrides via ``_sample_overrides``.
      2. Builds an ``epoch_callback`` that reports per-epoch val loss to the
         trial (for HyperbandPruner) and accumulates the trajectory for the
         final objective value.
      3. Runs the position's ``run()`` with the overridden cfg. The pipeline
         calls our callback once per epoch during attention NN training —
         see ``src/shared/pipeline.py::_run_nn_training`` (gated on attention
         trainer kinds so the regular NN's phase doesn't bleed into our
         trajectory).
      4. Returns ``min(captured_val_losses)``.

    ``optuna.TrialPruned`` raised inside the callback propagates up through
    ``trainer.train()`` and out of ``run()``; Optuna's ``study.optimize``
    treats it as a pruned trial, not a failure.
    """
    runner = get_runner(pos)

    def objective(trial: optuna.Trial) -> float:
        overrides = _sample_overrides(trial)
        _validate_overrides(overrides)
        # Deep-copy so per-trial mutations (overrides + epoch_callback
        # installation + K's runner-side `attn_history_builder_fn`
        # injection) don't leak across trials. Assumes cfg values are
        # deepcopy-safe; today every position's cfg holds primitives,
        # dicts of primitives, and module-level callables — all of which
        # round-trip through deepcopy. A future cfg that captures a lambda
        # or a closure-with-state would break this; switch to a manual
        # shallow-copy + per-key strategy at that point.
        cfg = copy.deepcopy(base_cfg)
        cfg.update(overrides)
        _apply_attention_scheduler_overrides(cfg, overrides)

        # The objective only reads result["attn_history"]["val_loss"] (below),
        # so Ridge / ElasticNet / LightGBM / base NN are wasted compute per
        # trial. Disabling them drops trial wall-clock substantially and frees
        # the CPU branch, which is what makes n_jobs > 1 in study.optimize
        # viable on the 4 vCPU g6.xlarge. Attention NN is independent of
        # the other branches (it builds its own scaler; no stacking).
        cfg["train_ridge"] = False
        cfg["train_elasticnet"] = False
        cfg["train_lightgbm"] = False
        cfg["train_base_nn"] = False

        captured: list[float] = []

        def epoch_callback(epoch: int, avg_val_loss: float) -> None:
            captured.append(float(avg_val_loss))
            trial.report(float(avg_val_loss), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        cfg["epoch_callback"] = epoch_callback

        with _lease_cores("tune_nn_trial", default=None):
            result = runner(seed=seed, config=cfg)

        # Belt + suspenders: prefer the captured trajectory (always present
        # for attention trainers), fall back to the result dict's
        # attn_history (also val-only) if the callback never fired — would
        # only happen if cfg["train_attention_nn"] is False, which we don't
        # support.
        if captured:
            return float(min(captured))
        attn_history = result.get("attn_history") or {}
        val_losses = attn_history.get("val_loss") or []
        if not val_losses:
            raise RuntimeError(
                f"{pos}: no val_loss trajectory captured for trial {trial.number}. "
                f"Is train_attention_nn enabled in the position's CONFIG?"
            )
        return float(min(val_losses))

    return objective


# ---------------------------------------------------------------------------
# Config file output
# ---------------------------------------------------------------------------

# Map Optuna param names to the constants you'd hand-paste into
# src/{pos}/config.py. Names mirror what `build_pipeline_config` reads from
# `POSITION_CONFIG`.
_PARAM_TO_CONST = {
    "attn_d_model": "ATTN_D_MODEL",
    "attn_n_heads": "ATTN_N_HEADS",
    "attn_encoder_hidden_dim": "ATTN_ENCODER_HIDDEN_DIM",
    "attn_dropout": "ATTN_DROPOUT",
    "attn_lr": "ATTN_LR",
    "attn_batch_size": "ATTN_BATCH_SIZE",
    "scheduler_type": "SCHEDULER_TYPE",
    "cosine_t0": "COSINE_T0",
    "cosine_t_mult": "COSINE_T_MULT",
    # The tuner objective trains attention only, so scheduler LR scale params
    # must be pasted into the attention-specific config fields. The shared
    # scheduler fields still exist for the regular NN path.
    "cosine_eta_min": "ATTN_COSINE_ETA_MIN",
    "onecycle_max_lr": "ATTN_ONECYCLE_MAX_LR",
    "onecycle_pct_start": "ONECYCLE_PCT_START",
    "nn_backbone_layers": "NN_BACKBONE_LAYERS",
    "nn_head_hidden": "NN_HEAD_HIDDEN",
    "nn_dropout": "NN_DROPOUT",
    "nn_lr": "NN_LR",
    "nn_weight_decay": "NN_WEIGHT_DECAY",
    "nn_batch_size": "NN_BATCH_SIZE",
}


def _apply_attention_scheduler_overrides(cfg: dict, overrides: dict) -> None:
    """Route sampled scheduler LR scale params to attention-specific keys.

    Production configs now carry ``attn_cosine_eta_min`` /
    ``attn_onecycle_max_lr`` so attention training can use scaled scheduler
    values without changing the regular NN schedule. ``tune_nn`` still samples
    the historical unprefixed Optuna params for study compatibility; mirror
    those onto the attention keys before running the attention-only objective.
    """
    sched_type = cfg.get("scheduler_type")
    if sched_type == "cosine_warm_restarts" and "cosine_eta_min" in overrides:
        cfg["attn_cosine_eta_min"] = overrides["cosine_eta_min"]
    elif sched_type == "onecycle" and "onecycle_max_lr" in overrides:
        cfg["attn_onecycle_max_lr"] = overrides["onecycle_max_lr"]


def _format_value(val) -> str:
    """Render an Optuna param value as a Python literal for config.py."""
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, float):
        return f"{val:.6g}"
    if isinstance(val, (list, tuple)):
        inner = ", ".join(_format_value(v) for v in val)
        return f"[{inner}]"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _format_config_lines(pos: str, best_params: dict) -> str:
    """Render best params as ``{POS}_NN_*`` constants ready to paste into
    ``src/{pos}/config.py``.
    """
    prefix = pos.upper()
    lines = [
        f"# Tuned attention-NN params for {pos} — paste into src/{pos.lower()}/config.py:",
        "# ATTN_ENCODER_HIDDEN_DIM=0 means single-layer encoder, not zero-width hidden.",
    ]
    for param, const_suffix in _PARAM_TO_CONST.items():
        if param not in best_params:
            continue
        val = best_params[param]
        if param == "nn_backbone_layers":
            # Stored as a tuple from suggest_categorical; render as a list.
            val = list(val)
        lines.append(f"{prefix}_{const_suffix} = {_format_value(val)}")
    return "\n".join(lines)


def _trial_to_params(trial: optuna.trial.FrozenTrial) -> dict:
    """Pull a clean params dict from a completed Optuna trial.

    Resolves ``nn_backbone_layers_idx`` (stored as int in Optuna) back to the
    concrete preset list so downstream consumers (JSON output, config-line
    rendering) see the user-facing key name and shape.
    """
    p = dict(trial.params)
    if "nn_backbone_layers_idx" in p:
        idx = p.pop("nn_backbone_layers_idx")
        if (
            not isinstance(idx, int)
            or isinstance(idx, bool)
            or not 0 <= idx < len(_BACKBONE_PRESETS)
        ):
            raise ValueError(
                "Invalid tune_nn overrides: nn_backbone_layers_idx must map to "
                "a known backbone preset"
            )
        p["nn_backbone_layers"] = list(_BACKBONE_PRESETS[idx])
    _validate_overrides(p)
    return p


# ---------------------------------------------------------------------------
# S3 checkpoint (Spot resilience)
# ---------------------------------------------------------------------------


def _sqlite_backup(db_path: str, sqlite_timeout: int = _DEFAULT_SQLITE_TIMEOUT_SECONDS) -> str:
    """Return a temporary consistent SQLite backup path for S3 upload."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(db_path) + ".", suffix=".backup")
    os.close(fd)
    src = sqlite3.connect(
        f"file:{os.path.abspath(db_path)}?mode=ro", uri=True, timeout=sqlite_timeout
    )
    dst = sqlite3.connect(tmp_path)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return tmp_path


class _S3Checkpoint:
    """Round-trip the Optuna SQLite study DB to S3 so a Spot interruption
    can be resumed on Batch's retry.

    Layout: ``s3://{bucket}/tune_nn/{search-space-version}/{pos}/study.db`` (+ ``results.json``
    after the run completes). On startup we pull the DB if it exists;
    Optuna's ``load_if_exists=True`` then picks up every trial already
    completed and the next attempt only runs ``n_trials - already_done``
    more. After each trial completes (Optuna callback) we re-upload the DB
    so the worst case (immediate Spot reclaim) loses at most one in-flight
    trial. A SIGTERM handler does the same on graceful shutdown — Spot
    gives a 2-minute warning that Batch propagates to the container.
    """

    def __init__(self, bucket: str, pos: str, db_path: str, storage_version: str):
        # Local import — boto3 is only required when --checkpoint-s3 is set,
        # so the local CLI form runs without it.
        import boto3

        self.bucket = bucket
        self.pos = pos
        self.db_path = db_path
        self.s3 = boto3.client("s3")
        self.key_prefix = _s3_key_prefix(pos, storage_version)

    def _study_key(self) -> str:
        return f"{self.key_prefix}/study.db"

    def _results_key(self) -> str:
        return f"{self.key_prefix}/results.json"

    def download_study_db(self) -> None:
        """Pull the prior study.db from S3 if present so the next study.optimize()
        resumes from the previous attempt's last-completed trial."""
        from botocore.exceptions import ClientError

        key = self._study_key()
        try:
            self.s3.download_file(self.bucket, key, self.db_path)
            print(f"[checkpoint] resumed from s3://{self.bucket}/{key}")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                print(f"[checkpoint] no prior study at s3://{self.bucket}/{key}; starting fresh")
                return
            raise

    def upload_study_db(self) -> None:
        if not os.path.exists(self.db_path):
            return
        key = self._study_key()
        snapshot_path = _sqlite_backup(self.db_path)
        try:
            self.s3.upload_file(snapshot_path, self.bucket, key)
            print(f"[checkpoint] uploaded s3://{self.bucket}/{key}")
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(snapshot_path)

    def upload_results(self, results_path: str) -> None:
        if not os.path.exists(results_path):
            return
        key = self._results_key()
        self.s3.upload_file(results_path, self.bucket, key)
        print(f"[checkpoint] uploaded s3://{self.bucket}/{key}")


def _install_sigterm_handler(checkpoint: "_S3Checkpoint") -> None:
    """Trap SIGTERM (Spot 2-minute warning is delivered as SIGTERM by Batch)
    and upload the study DB before exiting so the next attempt resumes from
    the last-completed trial.

    Exits 0 because Batch's retry policy (see src/batch/launch.py's
    RETRY_STRATEGY) keys on the AWS-level ``statusReason`` (``Host EC2*``
    for Spot reclaim), not the container exit code. An exit-non-zero would
    bypass that policy and pin the failure as a real app error.

    **There is no race with the per-trial callback's S3 upload.** Python
    delivers signals only at bytecode boundaries — never inside C-level
    calls like ``boto3.client.upload_file``. So the handler cannot fire
    *during* an in-flight upload; it waits for the C call to return. The
    worst case is one duplicate upload of the same S3 key right after a
    trial completes (callback uploads, then handler uploads again before
    the loop advances to the next trial). S3 ``put_object`` is idempotent
    by key, so the second write is a no-op — last-writer-wins on identical
    bytes. The prior reviewer flagged this as a potential race; documenting
    it in code so the next reviewer doesn't re-litigate.
    """

    def _handler(signum, frame):
        print("[checkpoint] SIGTERM received — uploading study DB and exiting")
        try:
            checkpoint.upload_study_db()
        except Exception as e:
            # Never mask the shutdown with a checkpoint failure — log loudly
            # but exit anyway. Spot is about to kill us regardless.
            print(f"[checkpoint] WARNING: final upload failed: {e!r}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handler)


def _create_or_load_study(
    pos: str,
    *,
    storage_version: str,
    sqlite_timeout: int,
    base_cfg: dict | None = None,
    sampler_seed: int = 42,
) -> optuna.Study:
    return optuna.create_study(
        study_name=_study_name(pos, storage_version),
        storage=_make_storage(_study_db_path(pos, storage_version), sqlite_timeout),
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=sampler_seed),
        pruner=HyperbandPruner(
            min_resource=_HYPERBAND_MIN_RESOURCE,
            reduction_factor=_HYPERBAND_REDUCTION_FACTOR,
            max_resource=int((base_cfg or get_config(pos))["nn_epochs"]),
        ),
    )


def _mps_worker_entry(
    pos: str,
    worker_idx: int,
    target_completed_trials: int,
    seed: int,
    storage_version: str,
    sqlite_timeout: int,
    deadline_epoch: float | None,
    env_overrides: dict[str, str],
) -> None:
    os.environ.update(env_overrides)
    os.environ[_CORE_POOL_POS_ENV] = f"{pos}-tune-{worker_idx}"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")

    db_path = _study_db_path(pos, storage_version)
    _configure_sqlite_for_parallel(db_path, sqlite_timeout)
    base_cfg = get_config(pos)
    objective = _make_objective(pos, base_cfg, seed)

    while True:
        study = _create_or_load_study(
            pos,
            storage_version=storage_version,
            sqlite_timeout=sqlite_timeout,
            base_cfg=base_cfg,
            sampler_seed=42 + worker_idx,
        )
        if _completed_trials(study) >= target_completed_trials:
            return
        timeout = None
        if deadline_epoch is not None:
            timeout = max(0.0, deadline_epoch - time.time())
            if timeout <= 0:
                return
        study.optimize(objective, n_trials=1, timeout=timeout, show_progress_bar=False)


def _run_mps_optimize(
    pos: str,
    *,
    n_jobs: int,
    n_trials: int,
    seed: int,
    timeout: int | None,
    storage_version: str,
    sqlite_timeout: int,
    checkpoint: "_S3Checkpoint | None",
    checkpoint_interval: int,
) -> None:
    n_jobs = max(1, int(n_jobs))
    deadline = time.time() + timeout if timeout is not None else None
    db_path = _study_db_path(pos, storage_version)
    _configure_sqlite_for_parallel(db_path, sqlite_timeout)

    cpu_ids = _current_cpu_ids()
    core_pool_dir = tempfile.mkdtemp(prefix="ff-tune-core-pool-", dir="/tmp")
    core_pool_addr, set_active_count, stop_core_pool = _start_core_pool(cpu_ids, core_pool_dir)
    set_active_count(n_jobs)

    ctx = mp.get_context("spawn")
    env_overrides = {
        _CORE_POOL_ADDR_ENV: core_pool_addr,
        "FF_DEVICE": os.environ.get("FF_DEVICE", "cuda"),
        "CUDA_MPS_PIPE_DIRECTORY": os.environ.get("CUDA_MPS_PIPE_DIRECTORY", ""),
        "CUDA_MPS_LOG_DIRECTORY": os.environ.get("CUDA_MPS_LOG_DIRECTORY", ""),
    }
    if "FF_CUDA_GRAPH" in os.environ:
        env_overrides["FF_CUDA_GRAPH"] = os.environ["FF_CUDA_GRAPH"]
    if "FF_AMP_DTYPE" in os.environ:
        env_overrides["FF_AMP_DTYPE"] = os.environ["FF_AMP_DTYPE"]
    if "FF_COMPILE" in os.environ:
        env_overrides["FF_COMPILE"] = os.environ["FF_COMPILE"]

    workers = [
        ctx.Process(
            target=_mps_worker_entry,
            args=(
                pos,
                idx,
                n_trials,
                seed,
                storage_version,
                sqlite_timeout,
                deadline,
                env_overrides,
            ),
            name=f"tune-nn-{pos.lower()}-{idx}",
        )
        for idx in range(n_jobs)
    ]

    print(
        f"[mps] launching {n_jobs} Optuna worker processes for {pos} "
        f"(target complete trials={n_trials})",
        flush=True,
    )
    last_checkpoint = time.monotonic()
    try:
        for worker in workers:
            worker.start()

        while True:
            alive = [worker for worker in workers if worker.is_alive()]
            failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
            if failed:
                for worker in alive:
                    worker.terminate()
                raise RuntimeError(
                    f"{pos}: MPS tune worker failed: "
                    + ", ".join(f"{w.name} exit={w.exitcode}" for w in failed)
                )

            now = time.monotonic()
            if (
                checkpoint is not None
                and checkpoint_interval > 0
                and now - last_checkpoint >= checkpoint_interval
            ):
                checkpoint.upload_study_db()
                last_checkpoint = now

            if not alive:
                break
            time.sleep(2.0)
    finally:
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        stop_core_pool()
        shutil.rmtree(core_pool_dir, ignore_errors=True)
        if checkpoint is not None:
            checkpoint.upload_study_db()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tune attention-NN hyperparameters via Optuna, per position."
    )
    parser.add_argument(
        "positions",
        nargs="+",
        help="Positions to tune (QB, RB, WR, TE, K, DST).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=_DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials per position (default: {_DEFAULT_N_TRIALS}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-position wall-clock cap in seconds (default: no cap).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for pipeline runs inside each trial (Optuna sampler uses its own seed=42).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_DEFAULT_N_JOBS,
        help=(
            "Concurrent Optuna trials (thread-based; safe with the SQLite storage). "
            "NN trials are GPU-bound and share the single GPU, so this is bounded by "
            "GPU memory, not CPU cores — default 2 fits a 16 GB card (T4 / RTX 5080); "
            "raising it gives diminishing returns since trials time-slice one GPU."
        ),
    )
    parser.add_argument(
        "--parallel-backend",
        choices=list(_PARALLEL_BACKENDS),
        default=_DEFAULT_PARALLEL_BACKEND,
        help=(
            "Trial concurrency backend. 'thread' uses Optuna's in-process n_jobs "
            "path for local compatibility. 'mps' starts true subprocess workers. "
            "'auto' resolves to mps only on native-Linux L4/g6 hosts; Mac and "
            "5080 hosts keep thread mode unless mps is explicitly requested."
        ),
    )
    parser.add_argument(
        "--sqlite-timeout",
        type=int,
        default=_DEFAULT_SQLITE_TIMEOUT_SECONDS,
        help=(
            "SQLite busy timeout in seconds for the Optuna study DB. MPS mode uses "
            "multi-process writers, so keep this high enough to ride out short locks."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=_DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
        help=(
            "Seconds between parent-owned S3 checkpoint uploads in MPS mode. "
            "Thread mode still checkpoints after each trial via Optuna callback."
        ),
    )
    parser.add_argument(
        "--print-best",
        action="store_true",
        help="Load existing studies and print best params without running new trials.",
    )
    parser.add_argument(
        "--checkpoint-s3",
        action="store_true",
        help=(
            "Batch/Spot mode: round-trip the SQLite study DB to "
            f"s3://$S3_BUCKET/tune_nn/{_DEFAULT_SEARCH_SPACE_VERSION}/{{pos}}/study.db "
            "so a Spot interruption can be resumed on Batch's retry. On startup "
            "we pull the DB if it exists; after each trial completes we "
            "re-upload it; a SIGTERM handler does a final upload before exit. "
            "Requires S3_BUCKET in the env (matches the convention used by "
            "src/tuning/tune_lgbm.py and src/batch/train.py)."
        ),
    )
    args = parser.parse_args()

    positions = [p.upper() for p in args.positions]
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be >= 1")
    requested_backend = args.parallel_backend
    parallel_backend = _resolve_parallel_backend(requested_backend)
    storage_version = _resolve_search_space_version(
        parallel_backend, cuda_graph=_env_truthy("FF_CUDA_GRAPH")
    )

    if not args.print_best:
        _ensure_data_from_s3()

    all_results: dict[str, dict] = {}

    for pos in positions:
        study_name = _study_name(pos, storage_version)
        db_path = _study_db_path(pos, storage_version)

        if args.print_best:
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=_make_storage(db_path, args.sqlite_timeout),
                )
                best = _trial_to_params(study.best_trial)
                print(
                    f"\n{pos} best trial #{study.best_trial.number} "
                    f"(val_loss = {study.best_value:.4f}):"
                )
                print(_format_config_lines(pos, best))
            except Exception as e:
                print(f"No saved study for {pos}: {e}")
            continue

        base_cfg = get_config(pos)
        if not base_cfg.get("train_attention_nn", False):
            print(
                f"\n[{pos}] skipping: train_attention_nn is False in CONFIG. "
                f"Enable it before tuning."
            )
            continue

        checkpoint: _S3Checkpoint | None = None
        if args.checkpoint_s3:
            bucket = os.environ.get("S3_BUCKET")
            if not bucket:
                raise SystemExit(
                    "--checkpoint-s3 requires the S3_BUCKET environment variable "
                    "(matches src/batch/train.py + src/tuning/tune_lgbm.py)."
                )
            checkpoint = _S3Checkpoint(bucket, pos, db_path, storage_version)
            checkpoint.download_study_db()
            _install_sigterm_handler(checkpoint)

        t0 = time.time()
        study = _create_or_load_study(
            pos,
            storage_version=storage_version,
            sqlite_timeout=args.sqlite_timeout,
            base_cfg=base_cfg,
        )
        if parallel_backend == _MPS_BACKEND:
            _configure_sqlite_for_parallel(db_path, args.sqlite_timeout)

        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        remaining = max(0, args.n_trials - completed)

        objective = _make_objective(pos, base_cfg, args.seed)
        callbacks = []
        if checkpoint is not None:
            # Optuna invokes callbacks after every trial regardless of state
            # (complete / pruned / failed). Upload after each so the worst-case
            # Spot reclaim loses at most the in-flight trial. Bind `checkpoint`
            # via default-arg so the closure isn't fragile to the outer
            # `for pos` reassignment (caught by ruff B023).
            callbacks.append(lambda study, trial, ck=checkpoint: ck.upload_study_db())

        print(f"\n{'=' * 70}")
        print(
            f"  Tuning {pos} attention NN — {completed}/{args.n_trials} done, "
            f"running {remaining} more"
        )
        print(
            f"  backend={parallel_backend} requested_backend={requested_backend} n_jobs={args.n_jobs} "
            f"storage={storage_version} cuda_graph={_env_truthy('FF_CUDA_GRAPH')}"
        )
        print(f"{'=' * 70}")

        if remaining > 0:
            if parallel_backend == _MPS_BACKEND:
                with _NvidiaMPS(enabled=True):
                    _run_mps_optimize(
                        pos,
                        n_jobs=args.n_jobs,
                        n_trials=args.n_trials,
                        seed=args.seed,
                        timeout=args.timeout,
                        storage_version=storage_version,
                        sqlite_timeout=args.sqlite_timeout,
                        checkpoint=checkpoint,
                        checkpoint_interval=args.checkpoint_interval,
                    )
                study = _create_or_load_study(
                    pos,
                    storage_version=storage_version,
                    sqlite_timeout=args.sqlite_timeout,
                    base_cfg=base_cfg,
                )
            else:
                study.optimize(
                    objective,
                    n_trials=remaining,
                    timeout=args.timeout,
                    show_progress_bar=True,
                    # Concurrent-trial count via --n-jobs (default 2). Trials are
                    # attention-NN-only (Ridge / base NN / LGBM skipped via the cfg
                    # overrides in _make_objective), so the CPU branch is idle and a
                    # 16 GB card (T4 / RTX 5080) easily holds two small attention
                    # models. NN trials are GPU-bound and time-slice the single GPU,
                    # so raising --n-jobs is bounded by GPU memory, not CPU cores;
                    # above ~2-3 risks CPU-side contention from FE / data loading.
                    n_jobs=args.n_jobs,
                    callbacks=callbacks,
                )
        else:
            print(f"[{pos}] all {args.n_trials} trials already completed; skipping optimize()")

        elapsed = time.time() - t0
        best = _trial_to_params(study.best_trial)
        state_counts = _study_state_counts(study)
        print(f"\n{pos} tuning complete in {elapsed:.0f}s")
        print(f"  Best trial #{study.best_trial.number}: val_loss = {study.best_value:.4f}")
        print(f"  Trial states: {state_counts}")
        print(f"\n{_format_config_lines(pos, best)}")

        all_results[pos] = {
            "best_trial": study.best_trial.number,
            "best_val_loss": study.best_value,
            "best_params": best,
            "n_trials": len(study.trials),
            "trial_state_counts": state_counts,
            "storage_version": storage_version,
            "parallel_backend": parallel_backend,
            "requested_parallel_backend": requested_backend,
            "n_jobs": args.n_jobs,
            "cuda_graph": _env_truthy("FF_CUDA_GRAPH"),
            "elapsed_seconds": round(elapsed, 1),
        }

        if checkpoint is not None:
            # Per-position results JSON consumed by src/tuning/aggregate_results.py
            # in the Batch fan-out workflow (each Spot job is one position, so the
            # cross-position merge happens in the aggregate step). Also push the
            # study DB one last time so the final trial's state survives even if
            # the per-trial callback didn't fire on the last trial.
            per_pos_path = f"tune_nn_{pos.lower()}_results.json"
            with open(per_pos_path, "w") as f:
                json.dump({pos: all_results[pos]}, f, indent=2, default=str)
            checkpoint.upload_results(per_pos_path)
            checkpoint.upload_study_db()

    if all_results:
        results_path = "tune_nn_results.json"
        tmp = f"{results_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        os.replace(tmp, results_path)
        print(f"\nResults saved to {results_path}")

        print("\n==== BEST_PARAMS_JSON_START ====")
        print(json.dumps(all_results, indent=2, default=str))
        print("==== BEST_PARAMS_JSON_END ====")

        hist_path = append_tuning_run(
            "tune_nn",
            all_results,
            n_trials=args.n_trials,
            positions=list(all_results),
        )
        print(f"History entry: {hist_path}")


if __name__ == "__main__":
    main()
