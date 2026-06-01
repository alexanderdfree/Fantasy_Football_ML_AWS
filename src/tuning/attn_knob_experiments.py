"""Issue #720 attention-knob experiments.

Two expensive, operator-triggered experiment designs live here:

* ``fanova``: repeated Optuna studies over eight attention-only knobs, scored on
  attention-NN test fantasy-point MAE, then summarized with Optuna fANOVA
  importances.
* ``doe``: a 12-run Plackett-Burman main-effects screen over the same eight
  knobs, repeated across seeds.

These are research diagnostics, not production tuning. They deliberately score
test MAE because issue #720 asks for post-hoc knob attribution; do not use these
numbers as a direct config-selection objective without a held-out confirmation
run.

Examples:
    python -m src.tuning.attn_knob_experiments fanova --position RB
    python -m src.tuning.attn_knob_experiments doe --position RB --seeds 42,43
    python -m src.tuning.attn_knob_experiments doe --dry-run
"""

from __future__ import annotations

import argparse
import copy
import importlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any

import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from src.tuning.history import append_tuning_run

DEFAULT_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)


@dataclass(frozen=True)
class AttentionKnob:
    """One attention-only hyperparameter and its two-level DoE bounds."""

    name: str
    low: Any
    high: Any


ATTN_KNOBS: tuple[AttentionKnob, ...] = (
    AttentionKnob("attn_d_model", 16, 64),
    AttentionKnob("attn_n_heads", 1, 4),
    AttentionKnob("attn_encoder_hidden_dim", 0, 64),
    AttentionKnob("attn_dropout", 0.0, 0.3),
    AttentionKnob("attn_lr", 1e-4, 1e-2),
    AttentionKnob("attn_batch_size", 128, 1024),
    AttentionKnob("attn_weight_decay", 1e-5, 1e-3),
    AttentionKnob("attn_patience", 15, 30),
)
KNOB_NAMES = tuple(k.name for k in ATTN_KNOBS)


def _sample_attention_overrides(trial: optuna.Trial) -> dict[str, Any]:
    """Sample the eight attention-only knobs for the fANOVA run."""

    d_model = trial.suggest_categorical("attn_d_model", [16, 24, 32, 48, 64])
    n_heads = trial.suggest_categorical("attn_n_heads", [1, 2, 4])
    if d_model % n_heads != 0:
        raise optuna.TrialPruned()
    return {
        "attn_d_model": d_model,
        "attn_n_heads": n_heads,
        "attn_encoder_hidden_dim": trial.suggest_categorical(
            "attn_encoder_hidden_dim", [0, 16, 32, 64]
        ),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
        "attn_lr": trial.suggest_float("attn_lr", 1e-4, 1e-2, log=True),
        "attn_batch_size": trial.suggest_categorical("attn_batch_size", [128, 256, 512, 1024]),
        "attn_weight_decay": trial.suggest_float("attn_weight_decay", 1e-5, 1e-3, log=True),
        # Patience is monotone-favored by a min(val/test loss) objective; keep
        # the screen tight and treat any high-importance result skeptically.
        "attn_patience": trial.suggest_int("attn_patience", 15, 30),
    }


def _make_cfg(base_cfg: dict, overrides: dict[str, Any], *, ridge_sentinel: bool) -> dict:
    """Build a per-trial config that trains only what this experiment reads."""

    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides)
    cfg["train_elasticnet"] = False
    cfg["train_lightgbm"] = False
    cfg["train_base_nn"] = False
    cfg["train_ridge"] = bool(ridge_sentinel)
    return cfg


def _attn_test_mae(result: dict) -> float:
    metrics = result.get("attn_nn_metrics")
    if not metrics or "total" not in metrics:
        raise RuntimeError(f"missing attention metrics in result keys: {sorted(result)}")
    return float(metrics["total"]["mae"])


def _ridge_mae(result: dict) -> float | None:
    metrics = result.get("ridge_metrics")
    if not metrics:
        return None
    return float(metrics["total"]["mae"])


def _parse_seeds(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _load_position(position: str):
    mod = importlib.import_module(f"src.{position.lower()}.run_pipeline")
    return mod.CONFIG, mod.run


def _run_one(position: str, seed: int, overrides: dict[str, Any], *, ridge_sentinel: bool) -> dict:
    base_cfg, run_fn = _load_position(position)
    cfg = _make_cfg(base_cfg, overrides, ridge_sentinel=ridge_sentinel)
    result = run_fn(seed=seed, config=cfg)
    return {
        "position": position.upper(),
        "seed": seed,
        "overrides": dict(overrides),
        "attn_test_mae": _attn_test_mae(result),
        "ridge_mae": _ridge_mae(result),
    }


def plackett_burman_design(n_factors: int) -> list[list[int]]:
    """Return the 12-run PB design for up to 11 factors, trimmed to n_factors."""

    if not 1 <= n_factors <= 11:
        raise ValueError("the built-in 12-run Plackett-Burman design supports 1..11 factors")
    base = [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]
    rows: list[list[int]] = []
    for shift in range(11):
        rows.append(base[-shift:] + base[:-shift] if shift else list(base))
    rows.append([-1] * 11)
    return [row[:n_factors] for row in rows]


def doe_overrides(signs: list[int]) -> dict[str, Any]:
    """Map one PB row of -1/+1 signs to concrete cfg overrides."""

    if len(signs) != len(ATTN_KNOBS):
        raise ValueError(f"expected {len(ATTN_KNOBS)} signs, got {len(signs)}")
    return {
        knob.name: knob.high if sign > 0 else knob.low
        for knob, sign in zip(ATTN_KNOBS, signs, strict=True)
    }


def estimate_doe_effects(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate high-minus-low main effects from completed DoE rows."""

    effects: dict[str, list[float]] = {name: [] for name in KNOB_NAMES}
    seeds = sorted({row["seed"] for row in rows})
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        for name in KNOB_NAMES:
            hi = [r["attn_test_mae"] for r in seed_rows if r["signs"][name] > 0]
            lo = [r["attn_test_mae"] for r in seed_rows if r["signs"][name] < 0]
            if hi and lo:
                effects[name].append(statistics.mean(hi) - statistics.mean(lo))

    summary = {}
    for name, vals in effects.items():
        if not vals:
            continue
        summary[name] = {
            "mean_effect": statistics.mean(vals),
            "std_effect": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        }
    return summary


def ridge_sentinel_ok(rows: list[dict], *, atol: float = 1e-9) -> bool:
    """Ridge is attention-free, so it must be identical within each seed."""

    seeds = sorted({row["seed"] for row in rows})
    for seed in seeds:
        vals = [row["ridge_mae"] for row in rows if row["seed"] == seed]
        vals = [v for v in vals if v is not None]
        if vals and max(vals) - min(vals) > atol:
            return False
    return True


def run_doe(position: str, seeds: list[int], *, ridge_sentinel: bool) -> dict:
    design = plackett_burman_design(len(ATTN_KNOBS))
    rows = []
    for seed in seeds:
        for run_idx, signs in enumerate(design, start=1):
            overrides = doe_overrides(signs)
            print(f"[doe] {position.upper()} seed={seed} run={run_idx}/{len(design)} {overrides}")
            row = _run_one(position, seed, overrides, ridge_sentinel=ridge_sentinel)
            row["run_idx"] = run_idx
            row["signs"] = dict(zip(KNOB_NAMES, signs, strict=True))
            rows.append(row)
    return {
        "design": "plackett_burman_12",
        "knobs": [knob.__dict__ for knob in ATTN_KNOBS],
        "rows": rows,
        "effects": estimate_doe_effects(rows),
        "ridge_sentinel_ok": ridge_sentinel_ok(rows) if ridge_sentinel else None,
    }


def run_fanova(
    position: str,
    seeds: list[int],
    *,
    n_trials: int,
    sampler_seed: int,
    ridge_sentinel: bool,
) -> dict:
    seed_results = {}
    for idx, seed in enumerate(seeds):
        study_seed = sampler_seed + idx
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=study_seed),
            study_name=f"{position}-{seed}",
        )

        def objective(trial: optuna.Trial, *, seed: int = seed) -> float:
            overrides = _sample_attention_overrides(trial)
            print(f"[fanova] {position.upper()} seed={seed} trial={trial.number} {overrides}")
            row = _run_one(position, seed, overrides, ridge_sentinel=ridge_sentinel)
            trial.set_user_attr("ridge_mae", row["ridge_mae"])
            return row["attn_test_mae"]

        study.optimize(objective, n_trials=n_trials)
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        importances = {}
        if len(completed) >= 2:
            importances = get_param_importances(
                study, evaluator=FanovaImportanceEvaluator(seed=study_seed)
            )
        seed_results[str(seed)] = {
            "best_trial": study.best_trial.number if completed else None,
            "best_attn_test_mae": study.best_value if completed else None,
            "best_params": dict(study.best_trial.params) if completed else {},
            "importances": importances,
            "completed_trials": len(completed),
            "trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": dict(t.params),
                    "ridge_mae": t.user_attrs.get("ridge_mae"),
                    "state": t.state.name,
                }
                for t in study.trials
            ],
        }
    return {
        "knobs": [knob.__dict__ for knob in ATTN_KNOBS],
        "n_trials": n_trials,
        "sampler": "TPESampler",
        "sampler_seed": sampler_seed,
        "seeds": seed_results,
    }


def print_dry_run(mode: str, position: str, seeds: list[int], n_trials: int) -> None:
    print(f"mode={mode} position={position.upper()} seeds={seeds}")
    print(f"knobs={KNOB_NAMES}")
    if mode == "fanova":
        print(f"planned pipeline runs: {len(seeds) * n_trials}")
    else:
        design = plackett_burman_design(len(ATTN_KNOBS))
        print(f"design=plackett_burman_12 planned pipeline runs: {len(seeds) * len(design)}")
        for idx, signs in enumerate(design, start=1):
            print(f"  run {idx:02d}: {dict(zip(KNOB_NAMES, signs, strict=True))}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mode", choices=("fanova", "doe"))
    parser.add_argument("--position", default="RB", help="Position to run (default: RB)")
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 42..49)",
    )
    parser.add_argument("--n-trials", type=int, default=30, help="fANOVA trials per seed")
    parser.add_argument("--sampler-seed", type=int, default=42, help="Optuna sampler seed")
    parser.add_argument(
        "--ridge-sentinel",
        action="store_true",
        help="Also train Ridge so attention-only runs get a data-identity sentinel",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the design without training")
    parser.add_argument("--no-history", action="store_true", help="Do not write benchmark_history")
    args = parser.parse_args(argv)

    position = args.position.upper()
    seeds = _parse_seeds(args.seeds)
    if args.dry_run:
        print_dry_run(args.mode, position, seeds, args.n_trials)
        return

    if args.mode == "fanova":
        results = run_fanova(
            position,
            seeds,
            n_trials=args.n_trials,
            sampler_seed=args.sampler_seed,
            ridge_sentinel=args.ridge_sentinel,
        )
    else:
        results = run_doe(position, seeds, ridge_sentinel=args.ridge_sentinel)

    print(results)
    if not args.no_history:
        append_tuning_run(
            f"attn_knob_{args.mode}",
            {position: results},
            n_trials=args.n_trials
            if args.mode == "fanova"
            else len(plackett_burman_design(len(ATTN_KNOBS))),
            positions=[position],
            note="Issue #720 attention-knob experiment runner; test-MAE research diagnostic.",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
