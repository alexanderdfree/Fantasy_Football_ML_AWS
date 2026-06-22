"""Optuna-based hyperparameter tuning for the attention NN, per position.

Usage:
    python -m src.tuning.tune_nn QB                     # tune one position
    python -m src.tuning.tune_nn QB RB WR TE            # multiple (sequential)
    python -m src.tuning.tune_nn RB --n-trials 30
    python -m src.tuning.tune_nn RB --timeout 7200      # seconds, per position
    python -m src.tuning.tune_nn RB --n-jobs auto       # concurrent trials (RAM/CPU-bound)
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
* **`ATTN_STATIC_FEATURES`**: structural feature choice, not a hyperparam.
  CLAUDE.md's stop-rule on rolling-features-in-static still applies.
  (**`ATTN_HISTORY_STATS`** is fixed in the default *full* scope but IS
  searched as a token-bundle subset under ``--scope history`` — see that
  arg and ``src/tuning/attn_history_space.py``; the windowed-column
  stop-rule is enforced by ``assert_raw_per_game``.)
* **`nn_non_negative_targets`**: per-head correctness constraint.

Scope (``--scope``)
-------------------
``full`` (default) searches attention sizing + static backbone + scheduler
(the historical scheduler_v2 space). ``history`` (v2 isolation) searches ONLY
the attention game-history branch — ``attn_max_seq_len`` (sequence length) +
an ``attn_history_stats`` token-bundle subset — and FREEZES the entire
production recipe (attention sizing, lr, batch, scheduler, AND the static
backbone) at the position's POSITION_CONFIG. Isolating those two axes is
deliberate: the history effects are small (~2-3%), and v1 — which co-tuned lr
+ sizing alongside them — let lr dominate the objective and swamp them
(GH #1239). It lands in the separate ``history_v2`` study namespace and
supports QB/RB/WR/TE only. The Batch route carries it via ``FF_TUNE_SCOPE``
(the fixed ENTRYPOINT can't take ``--scope``), set by ``launch_tune --scope``.

Batch follow-up
---------------
`src/tuning/launch_tune.py` and `.github/workflows/retune-nn-batch.yml` will
fan out one Spot GPU host per position (preferring g6.xlarge, falling back to
g5.xlarge; matching `train-batch.yml`). The
container dispatches to this script via a new `--mode=tune` flag added to
`src/batch/train.py` — the Batch job's `command` becomes:

    ["--position", "RB", "--mode", "tune", "--n-trials", "30"]

instead of the current

    ["--position", "RB", "--seed", "42"].

A `--checkpoint-s3` flag added here will periodically upload the SQLite study
DB to `s3://$S3_BUCKET/tune_nn/{search-space-version}/{pos}/` and trap SIGTERM
so a Spot interruption can resume the search on Batch's retry.

The Batch launcher passes `--parallel-backend auto`: native-Linux Batch L4/g6
or A10G/g5 hosts resolve to NVIDIA MPS; Mac/MPS and RTX 5080 local hosts keep
the historical thread backend unless an operator explicitly forces
`--parallel-backend mps`.
"""

import argparse
import contextlib
import copy
import itertools
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

# Already paid for: detect_platform imports torch at module level, so pulling
# in the capture resolver adds no new import weight to the tune CLI.
from src.shared.utils import cuda_graph_enabled as _cuda_graph_enabled
from src.shared.utils import cuda_graph_full_enabled as _cuda_graph_full_enabled

# Attention game-history-branch search space (``--scope history``). Dependency-
# free (re only); safe to import at module top.
from src.tuning import attn_history_space as _attn_hist
from src.tuning.history import append_tuning_run
from src.tuning.tune_nn_storage import (
    SCOPE_ROOTS as _SCOPE_ROOTS,
)
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

_DEFAULT_SCOPE = "full"
_HISTORY_SCOPE = "history"


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
# GPU hosts. Per-trial cost dropped substantially after 2026-05-21 from the
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
# Per-process trial-data memo, injected into each trial's cfg AFTER the
# per-trial deepcopy (never put it in base_cfg — entries hold large numpy
# arrays/frames that must not be deepcopied). The pipeline consults it via
# cfg["trial_data_memo"] to skip the per-trial split re-read, frame hashing,
# and attention history-array rebuild (PR D1). Thread mode shares it across
# concurrent trials: first concurrent trials may duplicate the compute and
# both store equivalent values — a benign race, so no lock.
_TRIAL_DATA_MEMO: dict = {}
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

# ``--scope history`` search space (v2 isolation): ONLY the two game-history-
# branch axes — sequence length (``attn_max_seq_len``) + the resolved per-game
# token list (``attn_history_stats``). The ENTIRE production recipe (attention
# sizing, lr, batch, scheduler AND the static backbone) is FROZEN at the
# position's POSITION_CONFIG, so the small (~2-3%) history effects aren't
# swamped by the large lr/sizing nuisances that confounded v1 (GH #1239). The
# token bundle booleans are Optuna trial params only; the override carries the
# resolved ``attn_history_stats`` list, so validation never sees the booleans.
_HISTORY_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "attn_max_seq_len",
        "attn_history_stats",
    }
)
# History scope samples no scheduler/sizing knobs, so allowed == required.
_HISTORY_ALLOWED_KEYS: frozenset[str] = _HISTORY_REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Storage / process backend helpers
# ---------------------------------------------------------------------------


def _is_batch_mps_gpu_linux(platform_info) -> bool:
    gpu_name = (platform_info.gpu_name or "").lower()
    return (
        platform_info.backend == "cuda"
        and platform_info.os == "Linux"
        and not platform_info.is_wsl
        and (
            platform_info.compute_capability in {(8, 6), (8, 9)}
            or "a10g" in gpu_name
            or "l4" in gpu_name
        )
    )


def _resolve_parallel_backend(requested: str) -> str:
    """Resolve auto without changing local 5080/Mac defaults."""
    if requested != _AUTO_BACKEND:
        return requested
    info = detect_platform()
    if _is_batch_mps_gpu_linux(info):
        print(f"[tune_nn] parallel backend auto -> mps ({info.summary()})", flush=True)
        return _MPS_BACKEND
    print(f"[tune_nn] parallel backend auto -> thread ({info.summary()})", flush=True)
    return _THREAD_BACKEND


def _force_eager_for_concurrent_thread_trials(parallel_backend: str, n_jobs: int) -> bool:
    """Force the eager NN path when concurrent thread-mode trials would capture graphs.

    ``torch.cuda.graph`` captures in the default *global* mode, which raises
    ``cudaErrorIllegalState`` ("the operation cannot be performed in the present
    state") the moment any OTHER thread in the process has in-flight GPU work —
    with the thread backend and ``n_jobs > 1``, one trial's
    ``MultiHeadTrainer._maybe_graph_model`` capture races every other trial's
    kernel launches (measured on Batch g6/L4 2026-06-11, job 614ae7d7,
    ``--parallel-backend thread --n-jobs 4``). A capture lock can't fix it:
    global-mode capture also conflicts with other threads' *ordinary* kernel
    launches mid-capture, not just with concurrent captures, so eager is the
    only safe thread-concurrent configuration. The process-per-trial mps
    backend is immune (capture state is per-process) and ``n_jobs == 1`` thread
    mode has no concurrent threads — both stay graphed.

    Deliberately overrides an explicit ``FF_CUDA_GRAPH=1`` (that is the exact
    config the measured job crashed under). Must run BEFORE
    ``_resolve_storage_version`` so the study lands in the eager namespace
    (``scheduler_v2``) the trials will actually train under.
    """
    if parallel_backend != _THREAD_BACKEND or n_jobs <= 1 or not _cuda_graph_enabled():
        return False
    os.environ["FF_CUDA_GRAPH"] = "0"
    # The full-step + optimizer-tail captures (cuda_graph_full_enabled /
    # cuda_graph_opt_enabled) cascade off cuda_graph_enabled, so zeroing the base
    # env already kills both; force their envs too so the regime is explicit in
    # the environment a worker or subprocess might inherit.
    os.environ["FF_CUDA_GRAPH_FULL"] = "0"
    os.environ["FF_CUDA_GRAPH_OPT"] = "0"
    print(
        f"[tune_nn] thread backend with n_jobs={n_jobs} > 1: forcing FF_CUDA_GRAPH=0 "
        "(eager trials) — torch.cuda.graph's global-mode capture races other trial "
        "threads' in-flight GPU work (cudaErrorIllegalState); use the mps backend or "
        "n_jobs=1 to tune graphed.",
        flush=True,
    )
    return True


def _resolve_storage_version(
    parallel_backend: str, scope: str = _DEFAULT_SCOPE
) -> tuple[str, bool, bool]:
    """Storage namespace for this run, plus the capture decision it keys on.

    ``scope`` selects the search-space root (``scheduler_v2`` for full,
    ``history_v2`` for ``--scope history``) so the two never share a study DB.

    Keyed on ``cuda_graph_enabled()`` — the same autodetect + force-off-override
    resolver the trainer consults at capture time — NOT on FF_CUDA_GRAPH
    env-truthiness. Since the 2026-06-05 autodetect cutover the env is a
    force-OFF override only, so a raw truthy read mislabels two real cases: an
    sm_80+ box (5080, g6/L4, g5/A10G) with the env unset trains GRAPHED, and a
    sub-sm_80 box (T4) with ``FF_CUDA_GRAPH=1`` trains EAGER (the sm gate
    refuses capture). Either mislabel resumes a study DB from the wrong
    training trajectory (ADR-0017: graphed and eager are not comparable).
    """
    cuda_graph = _cuda_graph_enabled()
    full_graph = _cuda_graph_full_enabled()
    return (
        _resolve_search_space_version(
            parallel_backend,
            cuda_graph=cuda_graph,
            full_graph=full_graph,
            root=_SCOPE_ROOTS.get(scope, _DEFAULT_SEARCH_SPACE_VERSION),
        ),
        cuda_graph,
        full_graph,
    )


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


# Per-worker steady-state RSS on the Batch GPU shape, measured 2026-06-10 on a
# g6.xlarge (4 vCPU / 15000 MiB job): n_jobs=4 fits, n_jobs=8 is OOM-killed
# (exit=-9) mid-run, n_jobs=32 is OOM-killed during worker startup. Each
# spawn-context worker carries a ~1.2-1.5 GiB torch+CUDA runtime floor plus its
# own pandas splits / feature frames — nothing is shared across workers. The
# binding resource is container RAM, NOT GPU VRAM (peak_mem_gb≈0.1/trial on the
# 24 GiB L4).
_MPS_WORKER_RSS_BYTES = 2 * 1024**3
# Parent process + nvidia-cuda-mps-control daemon + slack.
_MPS_PARENT_RESERVE_BYTES = 1 * 1024**3
_CGROUP_V2_MEMORY_MAX = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_MEMORY_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
# cgroup v1 encodes "no limit" as a huge page-rounded sentinel near 2**63.
_CGROUP_NO_LIMIT_THRESHOLD = 1 << 60


def _cgroup_memory_limit_bytes(
    v2_path: str = _CGROUP_V2_MEMORY_MAX,
    v1_path: str = _CGROUP_V1_MEMORY_LIMIT,
) -> int | None:
    """The container's cgroup memory limit in bytes, or ``None`` when
    unlimited or not in a memory-limited cgroup (local macOS / bare Linux)."""
    for path in (v2_path, v1_path):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":  # cgroup v2 unlimited
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _CGROUP_NO_LIMIT_THRESHOLD:
            return None
        return value
    return None


def _ram_safe_n_jobs(n_jobs: int, limit_bytes: int | None) -> tuple[int, str | None]:
    """Clamp the MPS worker count to what the container memory limit holds.

    Returns ``(clamped_n_jobs, warning)``; ``warning`` is ``None`` when no
    clamp applied. Worker RSS is an estimate (``_MPS_WORKER_RSS_BYTES``), so
    this is a guardrail against egregious overcommit — n_jobs=32 on a
    15000 MiB job dies in a SIGKILL storm before any trial completes — not a
    guarantee that the clamped count fits.
    """
    if limit_bytes is None or n_jobs <= 1:
        return n_jobs, None
    fits = max(1, int((limit_bytes - _MPS_PARENT_RESERVE_BYTES) // _MPS_WORKER_RSS_BYTES))
    if n_jobs <= fits:
        return n_jobs, None
    warning = (
        f"[mps] clamping n_jobs {n_jobs} -> {fits}: container memory limit "
        f"{limit_bytes / 1024**3:.1f} GiB holds ~{fits} workers at "
        f"~{_MPS_WORKER_RSS_BYTES / 1024**3:.1f} GiB each (+ parent/daemon "
        "reserve); exceeding it gets workers OOM-killed (exit=-9)"
    )
    return fits, warning


def _resolve_n_jobs(raw: "str | int", parallel_backend: str) -> int:
    """Resolve ``--n-jobs``: a positive int, or ``auto`` to size to the host.

    ``auto`` = CPU count — the training loop is host/launch-bound, so ~1
    useful concurrent trial per vCPU — additionally RAM-clamped for the
    process-per-trial mps backend via :func:`_ram_safe_n_jobs` (each spawned
    worker carries its own ~2 GiB torch+CUDA runtime; the thread backend
    shares one runtime, so only CPUs bind). On the Batch g6.xlarge shape
    (4 vCPU / 15000 MiB) auto resolves to 4.
    """
    text = str(raw).strip().lower()
    if text == "auto":
        cpus = max(1, len(_current_cpu_ids()))
        if parallel_backend == _MPS_BACKEND:
            fit, _ = _ram_safe_n_jobs(cpus, _cgroup_memory_limit_bytes())
            return fit
        return cpus
    try:
        value = int(text)
    except ValueError:
        raise SystemExit(f"--n-jobs must be a positive integer or 'auto', got {raw!r}") from None
    if value < 1:
        raise SystemExit("--n-jobs must be >= 1")
    return value


def _default_n_jobs_text() -> str:
    """argparse default for ``--n-jobs``: the ``FF_TUNE_N_JOBS`` env wins.

    The Batch route: the training image's ENTRYPOINT parses argv with
    ``src/batch/train.py``, whose ``--n-jobs`` is ``type=int`` — the "auto"
    sentinel can't ride the command (exit 2 — Batch job 23b157e3), so
    ``src/tuning/launch_tune.py`` passes it via the job environment instead
    (same fixed-ENTRYPOINT channel as FF_TUNE_STACKED_SEEDS / FF_TUNE_AB_SPEC).
    An explicit ``--n-jobs`` flag still wins over the env value; either way
    the value is validated by :func:`_resolve_n_jobs`.
    """
    return os.environ.get("FF_TUNE_N_JOBS", "").strip() or str(_DEFAULT_N_JOBS)


class _NvidiaMPS:
    """Per-job NVIDIA CUDA MPS daemon wrapper.

    Batch tune jobs run a single container per 4-vCPU GPU host. Starting MPS
    inside that container gives the worker subprocesses one shared CUDA scheduling
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


def _validate_overrides(overrides: dict, scope: str = _DEFAULT_SCOPE) -> None:
    """Validate sampled tune_nn overrides before training or reporting them.

    ``attn_encoder_hidden_dim == 0`` is the one intentional zero sentinel: it
    selects the single-layer game encoder in ``_build_game_encoder``. Every real
    dimension, width, batch size, and optimizer scale must be positive.

    ``scope == "history"`` (v2 isolation) validates only the two game-history
    axes — ``attn_max_seq_len`` + ``attn_history_stats`` — because the entire
    production recipe (sizing, lr, batch, scheduler, static backbone) is frozen
    at POSITION_CONFIG and never sampled, so no recipe keys are required or
    allowed in ``overrides``.
    """
    history = scope == _HISTORY_SCOPE
    required_keys = _HISTORY_REQUIRED_KEYS if history else _BASE_TUNED_OVERRIDE_KEYS
    allowed_keys = _HISTORY_ALLOWED_KEYS if history else _TUNED_OVERRIDE_KEYS

    errors: list[str] = []

    unknown = sorted(set(overrides) - allowed_keys)
    if unknown:
        errors.append(f"unknown keys: {unknown}")

    missing = sorted(required_keys - overrides.keys())
    if missing:
        errors.append(f"missing keys: {missing}")

    # History scope (v2) FREEZES the whole production recipe — attention sizing,
    # lr, batch, scheduler AND the static backbone — at POSITION_CONFIG and
    # searches only the two history axes (validated below), so none of those
    # recipe keys appear in `overrides`. Full scope validates the recipe it
    # sampled.
    if not history:
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
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= value < 1.0
            ):
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
                errors.append(
                    f"{sched_type} overrides include irrelevant scheduler keys: {irrelevant}"
                )

            if sched_type == "cosine_warm_restarts":
                for key in ("cosine_t0", "cosine_t_mult"):
                    if not _is_positive_int(overrides.get(key)):
                        errors.append(f"{key} must be a positive int")
                eta_min = overrides.get("cosine_eta_min")
                if not _is_positive_number(eta_min):
                    errors.append("cosine_eta_min must be positive")
                elif (
                    _is_positive_number(overrides.get("attn_lr"))
                    and eta_min >= overrides["attn_lr"]
                ):
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

    if history:
        if not _is_positive_int(overrides.get("attn_max_seq_len")):
            errors.append("attn_max_seq_len must be a positive int")
        stats = overrides.get("attn_history_stats")
        if (
            not isinstance(stats, list)
            or not stats
            or any(not isinstance(s, str) or not s for s in stats)
        ):
            errors.append("attn_history_stats must be a non-empty list of column-name strings")
        else:
            # Stop-rule guard: history tokens must be raw per-game signals, never
            # windowed/expanding/rolling derivations.
            try:
                _attn_hist.assert_raw_per_game(stats)
            except ValueError as exc:
                errors.append(str(exc))

    if errors:
        raise ValueError("Invalid tune_nn overrides: " + "; ".join(errors))


def _sample_scheduler(trial: optuna.Trial) -> tuple[str, dict]:
    """Sample the (attention) scheduler type + its shape params. Used by full
    scope only (v2 history freezes the scheduler); the param names/ranges match
    the historical scheduler_v2 space."""
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", ["cosine_warm_restarts", "onecycle"]
    )
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
    return scheduler_type, scheduler_overrides


def _sample_overrides(
    trial: optuna.Trial, scope: str = _DEFAULT_SCOPE, position: str | None = None
) -> dict:
    """Sample one trial's cfg overrides. Raises ``optuna.TrialPruned`` for
    invalid combinations (e.g. ``n_heads`` not dividing ``d_model``).

    ``scope == "history"`` (v2 isolation) samples ONLY the two game-history-
    branch axes — ``attn_max_seq_len`` (sequence length) and a per-game token
    subset (one boolean per optional bundle, resolved to ``attn_history_stats``)
    — and freezes the ENTIRE production recipe (attention sizing, lr, batch,
    scheduler, and the static backbone) at the position's POSITION_CONFIG. No
    sizing/scheduler/nn_* params are sampled, so those large nuisances can't
    swamp the small history effects (the v1 lr confound, GH #1239).
    """
    if scope == _HISTORY_SCOPE:
        if position is None:
            raise ValueError("history scope requires a position (for token bundles)")
        # The ONLY two sampled axes: sequence length + per-game token subset.
        # Everything else stays at the production POSITION_CONFIG (frozen).
        seq_len = trial.suggest_categorical("attn_max_seq_len", _attn_hist.SEQ_LEN_CHOICES)
        enabled = [
            bundle
            for bundle in _attn_hist.optional_bundles(position)
            if trial.suggest_categorical(f"histbundle_{bundle}", [False, True])
        ]
        stats = _attn_hist.resolve_history_stats(position, enabled)
        # Resolved list + chosen bundles as user_attrs so _trial_to_params can
        # round-trip the config without reconstructing from the booleans.
        trial.set_user_attr("attn_history_stats", stats)
        trial.set_user_attr("attn_history_bundles", enabled)
        return {
            "attn_max_seq_len": seq_len,
            "attn_history_stats": stats,
        }

    d_model = trial.suggest_categorical("attn_d_model", [16, 24, 32, 48, 64])
    n_heads = trial.suggest_categorical("attn_n_heads", [1, 2, 4])
    if _is_positive_int(d_model) and _is_positive_int(n_heads) and d_model % n_heads != 0:
        raise optuna.TrialPruned()

    backbone_idx = trial.suggest_categorical(
        "nn_backbone_layers_idx", list(range(len(_BACKBONE_PRESETS)))
    )

    attn_lr = trial.suggest_float("attn_lr", 1e-4, 5e-3, log=True)
    nn_lr = trial.suggest_float("nn_lr", 1e-4, 5e-3, log=True)
    scheduler_type, scheduler_overrides = _sample_scheduler(trial)

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


def _make_objective(
    pos: str,
    base_cfg: dict,
    seed: int,
    *,
    stacked_n: int = 0,
    stacked_epochs: int = 30,
    scope: str = _DEFAULT_SCOPE,
):
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

    ``stacked_n >= 2`` switches each trial to the vmap seed-ensemble path
    (owner-sanctioned opt-in, 2026-06-11): capture ``stacked_n`` seeds of the
    trial config, train them as ONE stacked ensemble for ``stacked_epochs``
    fixed epochs (the ensemble regime — the caller applies
    ``apply_ensemble_env`` process-wide), and report the across-member MEAN
    val loss per epoch. The objective becomes seed-averaged (the
    "single-seed NN val loss is noise" fix) at ~1.5–2× single-seed trial
    cost; results live in ``_ens{N}x{E}``-suffixed study namespaces because
    the objective semantics differ from the eager early-stop path.
    """
    runner = get_runner(pos)

    if stacked_n >= 2:
        return _make_stacked_objective(pos, base_cfg, seed, stacked_n, stacked_epochs, scope=scope)

    def objective(trial: optuna.Trial) -> float:
        overrides = _sample_overrides(trial, scope, pos)
        _validate_overrides(overrides, scope)
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
        # viable on the 4 vCPU Batch GPU hosts. Attention NN is independent of
        # the other branches (it builds its own scaler; no stacking).
        cfg["train_ridge"] = False
        cfg["train_elasticnet"] = False
        cfg["train_lightgbm"] = False
        cfg["train_base_nn"] = False
        # Post-deepcopy on purpose (see _TRIAL_DATA_MEMO).
        cfg["trial_data_memo"] = _TRIAL_DATA_MEMO

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


def _make_stacked_objective(
    pos: str,
    base_cfg: dict,
    seed: int,
    stacked_n: int,
    stacked_epochs: int,
    scope: str = _DEFAULT_SCOPE,
):
    """The vmap seed-ensemble trial body (see ``_make_objective``'s docstring).

    Per trial: capture ``stacked_n`` seed constructions of the sampled config
    through the REAL pipeline (non-attention branches disabled, the worker's
    trial-data memo shared), train them as one stacked ensemble, and report
    the across-member MEAN combined val loss per epoch via
    ``train_stacked(epoch_callback=...)``. ``optuna.TrialPruned`` raised in
    the callback propagates out of ``train_stacked``. Objective value =
    ``min`` over the seed-averaged trajectory.
    """

    def objective(trial: optuna.Trial) -> float:
        import torch

        from src.tuning.ab_ensemble_seeds import capture_seeds, train_stacked

        overrides = _sample_overrides(trial, scope, pos)
        _validate_overrides(overrides, scope)
        cfg = copy.deepcopy(base_cfg)
        cfg.update(overrides)
        _apply_attention_scheduler_overrides(cfg, overrides)

        captured: list[float] = []

        def epoch_callback(epoch: int, mean_val_loss: float) -> None:
            captured.append(float(mean_val_loss))
            trial.report(float(mean_val_loss), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        seeds = [seed + k for k in range(stacked_n)]
        with _lease_cores("tune_nn_trial", default=None):
            captures, _ = capture_seeds(pos, seeds, base_cfg=cfg, memo=_TRIAL_DATA_MEMO)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_stacked(captures, cfg, device, stacked_epochs, epoch_callback=epoch_callback)
        if not captured:
            raise RuntimeError(
                f"{pos}: stacked trial {trial.number} captured no val trajectory "
                f"(stacked_epochs={stacked_epochs})"
            )
        return float(min(captured))

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
    # Game-history-branch axes (--scope history only): emitted when present so a
    # tuned winner pastes straight into POSITION_CONFIG. Absent from full-scope
    # best_params, so _format_config_lines skips them there.
    "attn_max_seq_len": "ATTN_MAX_SEQ_LEN",
    "attn_history_stats": "ATTN_HISTORY_STATS",
    # The tuner objective trains ATTENTION only, so every sampled scheduler param
    # (type + shape + LR scale) must be pasted into the attention-specific config
    # fields (ATTN_*), never the shared SCHEDULER_*/COSINE_*/ONECYCLE_* fields the
    # regular NN path also reads — else pasting tuned attention values silently
    # re-schedules the regular NN (#792, completing PR #743's partial namespacing).
    "scheduler_type": "ATTN_SCHEDULER_TYPE",
    "cosine_t0": "ATTN_COSINE_T0",
    "cosine_t_mult": "ATTN_COSINE_T_MULT",
    "cosine_eta_min": "ATTN_COSINE_ETA_MIN",
    "onecycle_max_lr": "ATTN_ONECYCLE_MAX_LR",
    "onecycle_pct_start": "ATTN_ONECYCLE_PCT_START",
    "nn_backbone_layers": "NN_BACKBONE_LAYERS",
    "nn_head_hidden": "NN_HEAD_HIDDEN",
    "nn_dropout": "NN_DROPOUT",
    "nn_lr": "NN_LR",
    "nn_weight_decay": "NN_WEIGHT_DECAY",
    "nn_batch_size": "NN_BATCH_SIZE",
}


def _apply_attention_scheduler_overrides(cfg: dict, overrides: dict) -> None:
    """Route ALL sampled scheduler params to attention-specific keys.

    The tuner objective trains attention only, so every sampled scheduler param
    (type + shape + LR scale) must land on the ``attn_*`` cfg keys the attention
    trainer reads (via ``_scheduler_value(..., "attn_")``), never the shared keys
    the regular NN path also reads. ``tune_nn`` samples the historical unprefixed
    Optuna param names for study compatibility; mirror them onto the attention
    keys here so the trial evaluates the same namespaced config that
    ``_PARAM_TO_CONST`` later emits (#792).
    """
    sched_type = overrides.get("scheduler_type")
    if sched_type is not None:
        cfg["attn_scheduler_type"] = sched_type
    if sched_type == "cosine_warm_restarts":
        for k in ("cosine_t0", "cosine_t_mult", "cosine_eta_min"):
            if k in overrides:
                cfg[f"attn_{k}"] = overrides[k]
    elif sched_type == "onecycle":
        for k in ("onecycle_max_lr", "onecycle_pct_start"):
            if k in overrides:
                cfg[f"attn_{k}"] = overrides[k]


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
    """Render best params as ``PositionConfig`` kwargs ready to paste into the
    ``POSITION_CONFIG = PositionConfig(...)`` call in ``src/{pos}/config.py``.

    The old UPPER_CASE module-level constants were retired by the PositionConfig
    migration (#826), so emit the snake_case kwargs the config object reads —
    the kwarg name is the constant suffix lowercased, which keeps scheduler
    params attention-namespaced (``attn_scheduler_type``, not ``scheduler_type``).
    """
    lines = [
        f"# Tuned attention-NN params for {pos} — paste into the",
        f"# POSITION_CONFIG = PositionConfig(...) call in src/{pos.lower()}/config.py:",
        "# (attn_encoder_hidden_dim=0 means single-layer encoder, not zero-width hidden.)",
    ]
    for param, const_suffix in _PARAM_TO_CONST.items():
        if param not in best_params:
            continue
        val = best_params[param]
        if param == "nn_backbone_layers":
            # Stored as a tuple from suggest_categorical; render as a list.
            val = list(val)
        lines.append(f"    {const_suffix.lower()}={_format_value(val)},")
    return "\n".join(lines)


def _trial_to_params(
    trial: optuna.trial.FrozenTrial, scope: str = _DEFAULT_SCOPE, position: str | None = None
) -> dict:
    """Pull a clean params dict from a completed Optuna trial.

    Full scope: resolves ``nn_backbone_layers_idx`` (stored as int in Optuna)
    back to the concrete preset list so downstream consumers (JSON output,
    config-line rendering) see the user-facing key name and shape.

    History scope: the per-game token bundle booleans (``histbundle_*``) are
    search params, not cfg keys — drop them and substitute the resolved
    ``attn_history_stats`` (stored as a ``user_attr`` at sample time;
    reconstructed from the booleans + ``position`` as a fallback).
    """
    p = dict(trial.params)
    if scope == _HISTORY_SCOPE:
        bundle_flags = {
            k[len("histbundle_") :]: v for k, v in p.items() if k.startswith("histbundle_")
        }
        for k in [k for k in p if k.startswith("histbundle_")]:
            del p[k]
        stats = trial.user_attrs.get("attn_history_stats")
        if stats is None:
            if position is None:
                raise ValueError(
                    "history _trial_to_params needs a position to resolve token bundles"
                )
            enabled = [b for b, on in bundle_flags.items() if on]
            stats = _attn_hist.resolve_history_stats(position, enabled)
        p["attn_history_stats"] = list(stats)
        _validate_overrides(p, scope)
        return p
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
    max_resource: int | None = None,
) -> optuna.Study:
    """``max_resource`` overrides the pruner's epoch ceiling — the stacked
    objective trains a FIXED epoch count (ensemble regime), so its rung
    ladder must top out there, not at the eager path's ``nn_epochs``."""
    return optuna.create_study(
        study_name=_study_name(pos, storage_version),
        storage=_make_storage(_study_db_path(pos, storage_version), sqlite_timeout),
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=sampler_seed),
        pruner=HyperbandPruner(
            min_resource=_HYPERBAND_MIN_RESOURCE,
            reduction_factor=_HYPERBAND_REDUCTION_FACTOR,
            max_resource=(
                int(max_resource)
                if max_resource is not None
                else int((base_cfg or get_config(pos))["nn_epochs"])
            ),
        ),
    )


def _worker_sampler_seed(worker_idx: int, iteration: int) -> int:
    """Sampler seed for one MPS-worker loop iteration.

    Rebuilding the study + sampler every iteration is the standard
    distributed-Optuna pattern (TPE must see fresh history), but a FIXED
    per-worker seed makes each fresh TPESampler replay the identical random
    draw during the startup phase (n_startup_trials=10) — observed as
    bit-identical duplicate trials on Batch (2026-06-10 RB run). Vary the
    seed by iteration; 1009 is prime and far above any plausible n_jobs, so
    worker seed streams never collide. Cross-run search-path reproducibility
    is already lost to concurrent trial-completion timing.
    """
    return 42 + worker_idx + 1009 * iteration


def _mps_worker_entry(
    pos: str,
    worker_idx: int,
    target_completed_trials: int,
    seed: int,
    storage_version: str,
    sqlite_timeout: int,
    deadline_epoch: float | None,
    env_overrides: dict[str, str],
    stacked_n: int = 0,
    stacked_epochs: int = 30,
    scope: str = _DEFAULT_SCOPE,
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
    objective = _make_objective(
        pos, base_cfg, seed, stacked_n=stacked_n, stacked_epochs=stacked_epochs, scope=scope
    )

    for iteration in itertools.count():
        study = _create_or_load_study(
            pos,
            storage_version=storage_version,
            sqlite_timeout=sqlite_timeout,
            base_cfg=base_cfg,
            sampler_seed=_worker_sampler_seed(worker_idx, iteration),
            max_resource=stacked_epochs if stacked_n >= 2 else None,
        )
        if _completed_trials(study) >= target_completed_trials:
            return
        timeout = None
        if deadline_epoch is not None:
            timeout = max(0.0, deadline_epoch - time.time())
            if timeout <= 0:
                return
        study.optimize(objective, n_trials=1, timeout=timeout, show_progress_bar=False)
        # Release the finished trial's CUDA-graph private pools + allocator
        # slack between trials — graph captures otherwise ratchet
        # reserved memory across a worker's sequential trials (gate-v
        # observation, todo/gpu_launch_bound_levers.md).
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    stacked_n: int = 0,
    stacked_epochs: int = 30,
    scope: str = _DEFAULT_SCOPE,
) -> int:
    """Run the spawn-based MPS optimize loop.

    Returns the effective worker count actually launched (RAM clamp may
    reduce the requested ``n_jobs``) so callers record what ran, not what
    was asked for.
    """
    n_jobs = max(1, int(n_jobs))
    n_jobs, ram_warning = _ram_safe_n_jobs(n_jobs, _cgroup_memory_limit_bytes())
    if ram_warning:
        print(ram_warning, flush=True)
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
    if "FF_CUDA_GRAPH_FULL" in os.environ:
        env_overrides["FF_CUDA_GRAPH_FULL"] = os.environ["FF_CUDA_GRAPH_FULL"]
    if "FF_CUDA_GRAPH_OPT" in os.environ:
        env_overrides["FF_CUDA_GRAPH_OPT"] = os.environ["FF_CUDA_GRAPH_OPT"]
    if "FF_AMP_DTYPE" in os.environ:
        env_overrides["FF_AMP_DTYPE"] = os.environ["FF_AMP_DTYPE"]
    if "FF_COMPILE" in os.environ:
        env_overrides["FF_COMPILE"] = os.environ["FF_COMPILE"]
    # Stacked mode: carry the ensemble regime explicitly (spawn inherits the
    # parent env anyway; explicit mirrors the FF_CUDA_GRAPH pattern above).
    for _key in ("FF_NN_NORM", "FF_NN_FIXED_EPOCHS"):
        if _key in os.environ:
            env_overrides[_key] = os.environ[_key]

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
                stacked_n,
                stacked_epochs,
                scope,
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
    return n_jobs


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
        "--scope",
        choices=[_DEFAULT_SCOPE, _HISTORY_SCOPE],
        # Env default (FF_TUNE_SCOPE) is the Batch route: src/batch/train.py
        # --mode=tune forwards a FIXED argv, so --scope can't ride the command;
        # launch_tune sets FF_TUNE_SCOPE in the job environment instead (same
        # channel as FF_TUNE_STACKED_SEEDS / FF_TUNE_N_JOBS).
        default=os.environ.get("FF_TUNE_SCOPE", _DEFAULT_SCOPE),
        help=(
            "Search-space scope. 'full' (default) tunes attention sizing + static "
            "backbone + scheduler (the historical scheduler_v2 space). 'history' "
            "(v2 isolation) tunes ONLY the attention GAME-HISTORY branch — "
            "attn_max_seq_len (sequence length) + a per-game token subset "
            "(attn_history_stats bundles) — and FREEZES the entire production "
            "recipe (attention sizing, lr, batch, scheduler, and the static "
            "backbone) at the position's config, so the small history effects "
            "aren't swamped by the lr/sizing nuisances that confounded v1. "
            "History-scope studies live in the separate history_v2 namespace and "
            "support QB/RB/WR/TE only (flat history). Pairs with stacked seeds for "
            "cheap seed-robust evaluation. Env default: FF_TUNE_SCOPE (the Batch route)."
        ),
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
        type=str,
        default=_default_n_jobs_text(),
        help=(
            "Concurrent Optuna trials, or 'auto' to size to the host (thread: "
            "CPU count; mps: CPU count, RAM-clamped). GPU VRAM is NOT the "
            "constraint (~0.1 GiB/trial); the binding resources are container/"
            "host RAM (~2 GiB per MPS worker process — measured 2026-06-10 on "
            "the 4-vCPU/15000-MiB Batch job: 4 fits, 8 OOMs mid-run, 32 OOMs "
            "at startup) and CPU cores (~1 core per trial; the training loop "
            "is launch-bound). MPS mode clamps to the container's cgroup "
            "memory limit; thread mode shares one process (GIL) and tops out "
            f"around 2-3. Env default: FF_TUNE_N_JOBS, else {_DEFAULT_N_JOBS} — "
            "the Batch route, since train.py's fixed argv parses --n-jobs as "
            "int and can't carry 'auto'."
        ),
    )
    parser.add_argument(
        "--parallel-backend",
        choices=list(_PARALLEL_BACKENDS),
        default=_DEFAULT_PARALLEL_BACKEND,
        help=(
            "Trial concurrency backend. 'thread' uses Optuna's in-process n_jobs "
            "path for local compatibility. 'mps' starts true subprocess workers. "
            "'auto' resolves to mps only on native-Linux Batch L4/g6 or "
            "A10G/g5 hosts; Mac and 5080 hosts keep thread mode unless mps is "
            "explicitly requested."
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
        "--stacked-seeds",
        type=int,
        default=os.environ.get("FF_TUNE_STACKED_SEEDS"),
        help=(
            "N >= 2 evaluates each trial as a vmap-stacked N-seed ensemble "
            "(seed-averaged objective, ensemble regime, fixed --stacked-epochs; "
            "studies land in _ens{N}x{E} namespaces, coexisting with eager-regime "
            "studies). DEFAULT is now GPU-gated: 24 on CUDA (the measured per-seed "
            "optimum), 0 (eager single-seed) off-CUDA where the FP32 stack is "
            "slower. Pass 0 to force eager. Env: FF_TUNE_STACKED_SEEDS (the Batch "
            "route, since train.py forwards a fixed argv). QB/RB/WR/TE only."
        ),
    )
    parser.add_argument(
        "--stacked-epochs",
        type=int,
        default=int(os.environ.get("FF_TUNE_STACKED_EPOCHS", "30") or 30),
        help="Fixed epochs per stacked trial (env default: FF_TUNE_STACKED_EPOCHS; default 30).",
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
    scope = args.scope
    # argparse `choices` validates an explicit --scope but NOT an env-sourced
    # default (FF_TUNE_SCOPE), so guard the env path explicitly.
    if scope not in (_DEFAULT_SCOPE, _HISTORY_SCOPE):
        raise SystemExit(
            f"scope must be one of {{{_DEFAULT_SCOPE}, {_HISTORY_SCOPE}}}, got {scope!r}"
        )
    if scope == _HISTORY_SCOPE:
        unsupported = [p for p in positions if not _attn_hist.is_supported(p)]
        if unsupported:
            raise SystemExit(
                f"--scope history supports {_attn_hist.supported_positions()} "
                f"(flat-history game-history branch); got {unsupported}"
            )

    # Batch env-flag dispatch: the training image's ENTRYPOINT is fixed to
    # src.batch.train, whose --mode=tune forwards a fixed argv here — there is
    # no arg-level channel for other src/tuning modules, and editing
    # src/batch/train.py fires a 6-position retrain. The ensemble A/B harness
    # therefore rides the tune route via FF_TUNE_ENSEMBLE_AB=1 in the job
    # environment (containerOverrides.environment passes through untouched).
    if os.environ.get("FF_TUNE_ENSEMBLE_AB", "").strip() == "1":
        from src.tuning.ab_ensemble_seeds import run_batch_entry

        run_batch_entry(positions[0])
        return

    # Same env-flag route for the eager-vs-stacked timing A/B (FP16+full-step
    # graph vs FP32+vmap, per-seed, norm held constant). See
    # src.tuning.ab_ensemble_seeds.run_compare_batch_entry for the env knobs.
    if os.environ.get("FF_TUNE_ENSEMBLE_COMPARE", "").strip() == "1":
        from src.tuning.ab_ensemble_seeds import run_compare_batch_entry

        run_compare_batch_entry(positions[0])
        return

    # Same env-flag route for the shared A/B harness: FF_TUNE_AB_SPEC=<dotted
    # spec module> (set by src/tuning/launch_ab.py) runs this position's
    # variant x seed cells with per-cell S3 checkpointing instead of Optuna.
    # See src/tuning/ab_batch.py for the env contract.
    if os.environ.get("FF_TUNE_AB_SPEC", "").strip():
        from src.tuning.ab_batch import run_batch_entry as ab_run_batch_entry

        ab_run_batch_entry(positions[0])
        return

    # Same env-flag route for the eager ablation_runner family: FF_TUNE_ABLATE_MOD=
    # <dotted ablation module> (set by src/tuning/launch_ablate.py) runs this
    # position's variant x seed cells eagerly (one full pipeline per cell) with
    # per-cell S3 checkpointing. See src/tuning/ablate_batch.py for the env contract.
    if os.environ.get("FF_TUNE_ABLATE_MOD", "").strip():
        from src.tuning.ablate_batch import run_batch_entry as ablate_run_batch_entry

        ablate_run_batch_entry(positions[0])
        return

    # GPU-gated default: 24 on CUDA, eager off-CUDA (resolver handles an explicit
    # 0/N override and rejects 1). K/DST aren't flat-history, so they always fall
    # back to eager below regardless of the resolved width.
    from src.tuning.ab_ensemble_seeds import resolve_default_stacked_seeds

    stacked_n = resolve_default_stacked_seeds(args.stacked_seeds)
    stacked_epochs = int(args.stacked_epochs)
    if stacked_n and not args.print_best:
        from src.tuning.ab_ensemble_seeds import ENSEMBLE_POSITIONS, apply_ensemble_env

        bad = [p for p in positions if p not in ENSEMBLE_POSITIONS]
        if bad:
            # K/DST aren't flat-history (nested / own-splits) so they can't vmap.
            # An EXPLICIT --stacked-seeds for them is a hard error; the default-on
            # path falls back to eager so a plain `tune_nn K` still works.
            explicit = args.stacked_seeds is not None
            if explicit:
                raise SystemExit(
                    f"--stacked-seeds supports {ENSEMBLE_POSITIONS} (flat-history); got {bad}"
                )
            print(
                f"[tune] {bad}: not flat-history — stacking unavailable, running eager.",
                flush=True,
            )
            stacked_n = 0
    if stacked_n and not args.print_best:
        from src.tuning.ab_ensemble_seeds import apply_ensemble_env

        # Regime BEFORE backend/namespace resolution: ensemble mode forces the
        # graphs off (the hand-rolled vmapped loop is eager) and FP32/LN, so
        # the storage version resolves graph-less and then gets the explicit
        # _ens{N}x{E} suffix below.
        apply_ensemble_env(stacked_epochs)

    requested_backend = args.parallel_backend
    parallel_backend = _resolve_parallel_backend(requested_backend)
    n_jobs = _resolve_n_jobs(args.n_jobs, parallel_backend)
    # Order matters: the eager-force mutates FF_CUDA_GRAPH[_FULL], and the storage
    # resolver below keys the study namespace off cuda_graph[_full]_enabled().
    _force_eager_for_concurrent_thread_trials(parallel_backend, n_jobs)
    storage_version, cuda_graph, cuda_graph_full = _resolve_storage_version(parallel_backend, scope)
    if stacked_n:
        # Seed-averaged fixed-epochs objective ≠ the eager early-stop objective;
        # never mix their trials in one study.
        storage_version = f"{storage_version}_ens{stacked_n}x{stacked_epochs}"

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
                best = _trial_to_params(study.best_trial, scope, pos)
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
            max_resource=stacked_epochs if stacked_n else None,
        )
        if parallel_backend == _MPS_BACKEND:
            _configure_sqlite_for_parallel(db_path, args.sqlite_timeout)

        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        remaining = max(0, args.n_trials - completed)

        objective = _make_objective(
            pos,
            base_cfg,
            args.seed,
            stacked_n=stacked_n,
            stacked_epochs=stacked_epochs,
            scope=scope,
        )
        from src.tuning.resource_probe import ResourceProbe

        probe = ResourceProbe().start()
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
            f"  scope={scope} backend={parallel_backend} "
            f"requested_backend={requested_backend} n_jobs={n_jobs} "
            f"storage={storage_version} cuda_graph={cuda_graph} "
            f"cuda_graph_full={cuda_graph_full}"
        )
        print(f"{'=' * 70}")

        effective_n_jobs = n_jobs
        if remaining > 0:
            if parallel_backend == _MPS_BACKEND:
                with _NvidiaMPS(enabled=True):
                    effective_n_jobs = _run_mps_optimize(
                        pos,
                        n_jobs=n_jobs,
                        n_trials=args.n_trials,
                        seed=args.seed,
                        timeout=args.timeout,
                        storage_version=storage_version,
                        sqlite_timeout=args.sqlite_timeout,
                        checkpoint=checkpoint,
                        checkpoint_interval=args.checkpoint_interval,
                        stacked_n=stacked_n,
                        stacked_epochs=stacked_epochs,
                        scope=scope,
                    )
                study = _create_or_load_study(
                    pos,
                    storage_version=storage_version,
                    sqlite_timeout=args.sqlite_timeout,
                    base_cfg=base_cfg,
                    max_resource=stacked_epochs if stacked_n else None,
                )
            else:
                study.optimize(
                    objective,
                    n_trials=remaining,
                    timeout=args.timeout,
                    show_progress_bar=True,
                    # Concurrent-trial count via --n-jobs (default 2). Trials are
                    # attention-NN-only (Ridge / base NN / LGBM skipped via the cfg
                    # overrides in _make_objective). GPU VRAM is NOT the constraint
                    # (~0.1 GiB/trial); the training loop is CPU-launch-bound and
                    # threads contend on the GIL, so thread mode tops out ~2-3.
                    # For more concurrency use the mps backend (worker processes),
                    # where container RAM (~2 GiB/worker) is what binds.
                    n_jobs=n_jobs,
                    callbacks=callbacks,
                )
        else:
            print(f"[{pos}] all {args.n_trials} trials already completed; skipping optimize()")

        resources = probe.stop()
        elapsed = time.time() - t0
        best = _trial_to_params(study.best_trial, scope, pos)
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
            "scope": scope,
            "storage_version": storage_version,
            "parallel_backend": parallel_backend,
            "requested_parallel_backend": requested_backend,
            # Effective worker count: the MPS path may RAM-clamp the request
            # (see _ram_safe_n_jobs), and tuning-history provenance must
            # record what ran, not what was asked for.
            "n_jobs": effective_n_jobs,
            # The actual capture decisions (cuda_graph[_full]_enabled()), not
            # the raw env — tuning-history provenance must record what ran.
            "cuda_graph": cuda_graph,
            "cuda_graph_full": cuda_graph_full,
            "elapsed_seconds": round(elapsed, 1),
            # Stacked-objective provenance + the peak-compute investigation
            # block (cgroup peak covers the MPS worker processes too).
            "stacked_seeds": stacked_n,
            "stacked_epochs": stacked_epochs if stacked_n else None,
            "resources": resources,
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
