"""Optuna-based hyperparameter tuning for the attention NN, per position.

Usage:
    python -m src.tuning.tune_nn QB                     # tune one position
    python -m src.tuning.tune_nn QB RB WR TE            # multiple (sequential)
    python -m src.tuning.tune_nn RB --n-trials 30
    python -m src.tuning.tune_nn RB --timeout 7200      # seconds, per position
    python -m src.tuning.tune_nn RB --print-best        # inspect saved study

MVP scope (v1)
--------------
Mirrors src/tuning/tune_lgbm.py's shape: per-position SQLite study, paste-ready
config output via `_format_config_lines`, BEST_PARAMS_JSON markers for CI log
capture. Differences vs the LightGBM tuner:

* Search space targets the **attention NN** (architecture + optimizer knobs).
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
* **`scheduler_type` search**: switching between `cosine_warm_restarts` and
  `onecycle` requires the matching scheduler-specific cfg keys to be present
  (`onecycle_max_lr`, `cosine_t0`, etc.). Out of scope until the cfg builder
  guarantees both sets — for v1, we keep whatever scheduler the position
  config already uses.
* **`ATTN_STATIC_FEATURES` / `ATTN_HISTORY_STATS`**: structural feature
  choices, not hyperparams. CLAUDE.md's stop-rule on rolling-features-in-
  static still applies.
* **`nn_non_negative_targets`**: per-head correctness constraint.

Batch follow-up
---------------
`src/batch/launch_tune.py` and `.github/workflows/retune-nn-batch.yml` will
fan out one Spot g4dn.xlarge per position (matching `train-batch.yml`). The
container dispatches to this script via a new `--mode=tune` flag added to
`src/batch/train.py` — the Batch job's `command` becomes:

    ["--position", "RB", "--mode", "tune", "--n-trials", "30"]

instead of the current

    ["--position", "RB", "--seed", "42"].

A `--checkpoint-s3` flag added here will periodically upload the SQLite study
DB to `s3://$S3_BUCKET/tune_nn/{pos}/` and trap SIGTERM so a Spot
interruption can resume the search on Batch's retry.
"""

import argparse
import copy
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.config import SPLITS_DIR
from src.shared.registry import get_config, get_runner


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


# Default trial count chosen with NN wall-clock in mind: a single attention
# trial takes 5–10 min locally, so 15 trials ≈ 1.5–2.5 hr per position on a
# laptop. Bump to 30 on a Batch g4dn.xlarge where each trial is 1–2 min.
_DEFAULT_N_TRIALS = 15

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


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def _sample_overrides(trial: optuna.Trial) -> dict:
    """Sample one trial's cfg overrides. Raises ``optuna.TrialPruned`` for
    invalid combinations (e.g. ``n_heads`` not dividing ``d_model``).
    """
    d_model = trial.suggest_categorical("attn_d_model", [16, 24, 32, 48, 64])
    n_heads = trial.suggest_categorical("attn_n_heads", [1, 2, 4])
    if d_model % n_heads != 0:
        raise optuna.TrialPruned()

    backbone_idx = trial.suggest_categorical(
        "nn_backbone_layers_idx", list(range(len(_BACKBONE_PRESETS)))
    )

    return {
        "attn_d_model": d_model,
        "attn_n_heads": n_heads,
        "attn_encoder_hidden_dim": trial.suggest_categorical(
            "attn_encoder_hidden_dim", [0, 16, 32, 64]
        ),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
        "attn_lr": trial.suggest_float("attn_lr", 1e-4, 5e-3, log=True),
        "attn_batch_size": trial.suggest_categorical("attn_batch_size", [128, 256, 512]),
        "nn_backbone_layers": list(_BACKBONE_PRESETS[backbone_idx]),
        "nn_head_hidden": trial.suggest_categorical("nn_head_hidden", [16, 24, 32, 48, 64]),
        "nn_dropout": trial.suggest_float("nn_dropout", 0.0, 0.4),
        "nn_lr": trial.suggest_float("nn_lr", 1e-4, 5e-3, log=True),
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
        cfg = copy.deepcopy(base_cfg)
        cfg.update(overrides)

        captured: list[float] = []

        def epoch_callback(epoch: int, avg_val_loss: float) -> None:
            captured.append(float(avg_val_loss))
            trial.report(float(avg_val_loss), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        cfg["epoch_callback"] = epoch_callback

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
    "nn_backbone_layers": "NN_BACKBONE_LAYERS",
    "nn_head_hidden": "NN_HEAD_HIDDEN",
    "nn_dropout": "NN_DROPOUT",
    "nn_lr": "NN_LR",
    "nn_weight_decay": "NN_WEIGHT_DECAY",
    "nn_batch_size": "NN_BATCH_SIZE",
}


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
    lines = [f"# Tuned attention-NN params for {pos} — paste into src/{pos.lower()}/config.py:"]
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
        p["nn_backbone_layers"] = list(_BACKBONE_PRESETS[idx])
    return p


# ---------------------------------------------------------------------------
# S3 checkpoint (Spot resilience)
# ---------------------------------------------------------------------------


class _S3Checkpoint:
    """Round-trip the Optuna SQLite study DB to S3 so a Spot interruption
    can be resumed on Batch's retry.

    Layout: ``s3://{bucket}/tune_nn/{pos}/study.db`` (+ ``results.json``
    after the run completes). On startup we pull the DB if it exists;
    Optuna's ``load_if_exists=True`` then picks up every trial already
    completed and the next attempt only runs ``n_trials - already_done``
    more. After each trial completes (Optuna callback) we re-upload the DB
    so the worst case (immediate Spot reclaim) loses at most one in-flight
    trial. A SIGTERM handler does the same on graceful shutdown — Spot
    gives a 2-minute warning that Batch propagates to the container.
    """

    def __init__(self, bucket: str, pos: str, db_path: str):
        # Local import — boto3 is only required when --checkpoint-s3 is set,
        # so the local CLI form runs without it.
        import boto3

        self.bucket = bucket
        self.pos = pos
        self.db_path = db_path
        self.s3 = boto3.client("s3")
        self.key_prefix = f"tune_nn/{pos.lower()}"

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
        self.s3.upload_file(self.db_path, self.bucket, key)
        print(f"[checkpoint] uploaded s3://{self.bucket}/{key}")

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
        "--print-best",
        action="store_true",
        help="Load existing studies and print best params without running new trials.",
    )
    parser.add_argument(
        "--checkpoint-s3",
        action="store_true",
        help=(
            "Batch/Spot mode: round-trip the SQLite study DB to "
            "s3://$S3_BUCKET/tune_nn/{pos}/study.db so a Spot interruption "
            "can be resumed on Batch's retry. On startup we pull the DB if it "
            "exists; after each trial completes we re-upload it; a SIGTERM "
            "handler does a final upload before exit. Requires S3_BUCKET in "
            "the env (matches the convention used by src/tuning/tune_lgbm.py "
            "and src/batch/train.py)."
        ),
    )
    args = parser.parse_args()

    positions = [p.upper() for p in args.positions]

    if not args.print_best:
        _ensure_data_from_s3()

    all_results: dict[str, dict] = {}

    for pos in positions:
        study_name = f"nn_{pos.lower()}"
        db_path = f"tune_nn_{pos.lower()}.db"

        if args.print_best:
            try:
                study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
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
            checkpoint = _S3Checkpoint(bucket, pos, db_path)
            checkpoint.download_study_db()
            _install_sigterm_handler(checkpoint)

        t0 = time.time()
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(
                min_resource=_HYPERBAND_MIN_RESOURCE,
                reduction_factor=_HYPERBAND_REDUCTION_FACTOR,
                # Pin to the cfg's epoch ceiling so an early-stopped trial 1
                # can't establish a too-low rung ladder for the rest of the
                # search (see _HYPERBAND_MIN_RESOURCE comment).
                max_resource=int(base_cfg["nn_epochs"]),
            ),
        )

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
        print(f"{'=' * 70}")

        if remaining > 0:
            study.optimize(
                objective,
                n_trials=remaining,
                timeout=args.timeout,
                show_progress_bar=True,
                n_jobs=1,  # NN trials are GPU/CPU-bound; parallel trials would oversubscribe.
                callbacks=callbacks,
            )
        else:
            print(f"[{pos}] all {args.n_trials} trials already completed; skipping optimize()")

        elapsed = time.time() - t0
        best = _trial_to_params(study.best_trial)
        print(f"\n{pos} tuning complete in {elapsed:.0f}s")
        print(f"  Best trial #{study.best_trial.number}: val_loss = {study.best_value:.4f}")
        print(f"\n{_format_config_lines(pos, best)}")

        all_results[pos] = {
            "best_trial": study.best_trial.number,
            "best_val_loss": study.best_value,
            "best_params": best,
            "n_trials": len(study.trials),
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


if __name__ == "__main__":
    main()
