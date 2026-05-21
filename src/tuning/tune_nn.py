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
* **K and DST**: their `run()` signatures take only `seed=` (they build their
  own data internally). Adding `config=` to those entry points is a 2-line
  follow-up; until then the tuner rejects K/DST with a clear error.
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
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.shared.registry import get_config, get_runner

# Positions whose `run()` signature currently lacks a `config=` kwarg. Until
# that's added, the tuner can't pass override cfgs to them and so we refuse
# to start a study. K and DST also build their data internally, which makes
# the bypass-`run()` workaround invasive — better to fix it once in a follow-
# up than to special-case here.
_UNSUPPORTED_POSITIONS = {"K", "DST"}

# Default trial count chosen with NN wall-clock in mind: a single attention
# trial takes 5–10 min locally, so 15 trials ≈ 1.5–2.5 hr per position on a
# laptop. Bump to 30 on a Batch g4dn.xlarge where each trial is 1–2 min.
_DEFAULT_N_TRIALS = 15

# HyperbandPruner: `min_resource` is the minimum epoch count a trial must
# complete before it's eligible for pruning. We pick 8 to give the trial a
# chance to escape its lr-warmup transient before being judged. `reduction
# _factor=3` is Optuna's default; `max_resource="auto"` infers from the
# longest trial seen so far.
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tune attention-NN hyperparameters via Optuna, per position."
    )
    parser.add_argument(
        "positions",
        nargs="+",
        help="Positions to tune (QB, RB, WR, TE). K and DST are not yet supported.",
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
    args = parser.parse_args()

    positions = [p.upper() for p in args.positions]
    bad = [p for p in positions if p in _UNSUPPORTED_POSITIONS]
    if bad:
        raise SystemExit(
            f"tune_nn: positions {bad} are not supported in v1 (their run() takes "
            f"only seed=, no config=). Tune QB/RB/WR/TE; K/DST support lands in a "
            f"follow-up that adds config= to their run() signatures."
        )

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
                max_resource="auto",
            ),
        )

        objective = _make_objective(pos, base_cfg, args.seed)

        print(f"\n{'=' * 70}")
        print(f"  Tuning {pos} attention NN — up to {args.n_trials} trials")
        print(f"{'=' * 70}")

        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
            n_jobs=1,  # NN trials are GPU/CPU-bound; parallel trials would oversubscribe.
        )

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
