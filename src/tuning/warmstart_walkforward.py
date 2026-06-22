"""Warm-start NN walk-forward prototype (experiment, not a production path).

Measures whether carrying the **attention NN's** weights forward across
walk-forward (rolling-origin) folds beats the production from-scratch fit. This
is the "let the model learn cumulatively as it walks through seasons" question:
the expanding-window walk-forward already accumulates *data* each origin; this
script tests whether also accumulating *model state* helps.

It drives the real production attention path (``_train_attention_holdout``) via
the default-off ``init_state_dict`` hook (numerically inert when ``None`` —
production always passes ``None``), so there is zero behavioural drift between
what this measures and what ships.

Two arms per ``(position, seed)``:

  * **COLD**  — every origin builds a fresh NN (== current production behaviour).
  * **WARM**  — origin 0 trains fresh; origin ``i>0`` loads origin ``i-1``'s
    trained weights *before* training (weights-only warm start).

Scope: QB / RB / WR / TE (the flat attention path that accepts injected frames).
K / DST are excluded — they use the nested trainer and build their own splits.

Caveats — read before trusting any number:

  * **Scaler refits per origin** inside ``_train_attention_holdout``. The warm
    weights were trained under origin ``i``'s ``StandardScaler``; origin ``i+1``
    refits on its (larger) train, so the warm arm sees a mild input-distribution
    shift. Freezing the scaler across origins is a follow-up (needs a scaler
    injection hook); v1 deliberately keeps the production per-fold refit.
  * **Optimizer is fresh each fold** (Adam moments reset to zero). Carrying
    optimizer state is a follow-up (would need an ``init_optimizer_state`` hook).
  * **Warm-arm metrics are not reproducible from ``(data, seed)`` alone** — a
    fold depends on the prior fold's weights, by design. The COLD arm stays
    seed-reproducible and reproduces the standard rolling-origin numbers.

Usage::

    python -m src.tuning.warmstart_walkforward --positions QB RB --seeds 42 123
    python -m src.tuning.warmstart_walkforward --positions RB --seeds 42 \
        --test-seasons 2022 2023 2024 2025 --epochs-warm 30
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from datetime import UTC, datetime

import pandas as pd
import torch

from src.config import ROLLING_ORIGIN_TEST_SEASONS, SPLITS_DIR
from src.data.split import rolling_origin_folds
from src.shared.pipeline import (
    _prepare_position_data,
    _read_split,
    _train_attention_holdout,
)
from src.shared.registry import get_config
from src.shared.utils import seed_everything

SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")


def _configure_runtime_env():
    """Set the harness's runtime env knobs — called from ``main()`` ONLY, never
    at import (importing this module must not mutate global env, or it pollutes
    the pytest process: an import-time ``FF_CUDA_GRAPH=0`` once broke the CUDA
    graph + feature-cache unit tests). All three FF_* knobs are read lazily at
    train/feature-prep time, so setting them here (before any training) is in
    time.

    * FF_CUDA_GRAPH=0 — keep each fold deterministic so the only intended
      cross-fold difference is the carried weights (graph capture is a speed
      knob, irrelevant to this comparison).
    * FF_COMPILE=0 — torch.compile off; keeps warm weights on the raw model the
      harness reads back via state_dict.
    * FF_FEATURE_CACHE_DISABLE=1 — the cache keys on data, not code; disable it
      so features always match THIS checkout's feature code.
    * Thread caps — best-effort (this box also games); the harness is GPU-bound
      (attention NN only), so CPU threads are a minor factor.
    """
    os.environ.setdefault("FF_CUDA_GRAPH", "0")
    os.environ.setdefault("FF_COMPILE", "0")
    os.environ.setdefault("FF_FEATURE_CACHE_DISABLE", "1")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "8")


def _load_full_featured_frame() -> pd.DataFrame:
    """Concat the three on-disk split parquets into one all-season, all-position
    featured frame. Every engineered column is split-independent (see
    ``rolling_origin_folds``'s docstring), so re-slicing by season is leakage
    free. Mirrors ``benchmark.py::_load_full_featured_frame``."""
    return pd.concat(
        [
            _read_split(f"{SPLITS_DIR}/train.parquet"),
            _read_split(f"{SPLITS_DIR}/val.parquet"),
            _read_split(f"{SPLITS_DIR}/test.parquet"),
        ],
        ignore_index=True,
    )


def _state_to_cpu(model) -> dict:
    """Detached CPU clone of a model's state_dict — safe to carry to the next
    fold without holding GPU memory or aliasing live parameters. ``load_state_dict``
    copies values cross-device, so the next (cuda) model loads these fine."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _run_arm(position, cfg, folds, seed, *, warm, epochs_warm):
    """Run one arm (COLD or WARM) of the walk-forward for a (position, seed).

    Returns a list of per-fold dicts with the total-fantasy-point metrics on
    each origin's held-out test season.
    """
    targets = cfg["targets"]
    out = []
    prev_state = None
    for fold_idx, fold_train_df, fold_val_df, fold_test_df in folds:
        test_season = int(fold_test_df["season"].iloc[0])
        arm = "WARM" if warm else "COLD"
        carry = prev_state if (warm and prev_state is not None) else None
        print(
            f"\n===== {position} seed={seed} {arm} | origin {fold_idx} "
            f"(test {test_season}) | warm_started={carry is not None} ====="
        )

        prep = _prepare_position_data(position, cfg, fold_train_df, fold_val_df, fold_test_df)
        (
            X_train,
            X_val,
            X_test,
            y_train_dict,
            y_val_dict,
            y_test_dict,
            pos_train,
            pos_val,
            pos_test,
            feature_cols,
        ) = prep

        # Same seed each origin so the COLD arm matches standard rolling-origin;
        # for the WARM arm the loaded weights override the post-seed random init,
        # so the seed only governs dropout / shuffling here.
        seed_everything(seed)

        # Optionally shorten the warm folds (a warm model may need fewer epochs).
        # Never mutate the shared cfg — shallow-copy with the override.
        run_cfg = cfg
        if carry is not None and epochs_warm is not None:
            run_cfg = {**cfg, "nn_epochs": epochs_warm}

        model, _scaler, _preds, metrics, _hist, _attn_cols = _train_attention_holdout(
            position,
            run_cfg,
            targets,
            seed,
            X_train,
            X_val,
            X_test,
            y_train_dict,
            y_val_dict,
            y_test_dict,
            pos_train,
            pos_val,
            pos_test,
            feature_cols,
            opp_source_frames=(fold_train_df, fold_val_df, fold_test_df),
            init_state_dict=carry,
        )

        total = metrics["total"]
        out.append(
            {
                "origin": fold_idx,
                "test_season": test_season,
                "warm_started": carry is not None,
                "n_test": int(len(X_test)),
                "mae": float(total["mae"]),
                "rmse": float(total["rmse"]),
                "r2": float(total["r2"]),
            }
        )

        if warm:
            prev_state = _state_to_cpu(model)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def _agg(values):
    """mean / std (population-style, std=0 for n<2) over a list of floats."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "n": len(vals)}


def _summarize(runs):
    """Build the COLD-vs-WARM comparison, per position × test_season (averaged
    over seeds) and as the walk-forward mean±std across origins per arm."""
    summary = {}
    positions = sorted({r["position"] for r in runs})
    for position in positions:
        pos_runs = [r for r in runs if r["position"] == position]
        seasons = sorted({f["test_season"] for r in pos_runs for f in r["folds"]})

        per_season = {}
        for season in seasons:
            entry = {}
            for arm in ("cold", "warm"):
                maes, rmses, r2s = [], [], []
                for r in pos_runs:
                    if r["arm"] != arm:
                        continue
                    for f in r["folds"]:
                        if f["test_season"] == season:
                            maes.append(f["mae"])
                            rmses.append(f["rmse"])
                            r2s.append(f["r2"])
                entry[arm] = {"mae": _agg(maes), "rmse": _agg(rmses), "r2": _agg(r2s)}
            cold_mae = entry["cold"]["mae"]["mean"]
            warm_mae = entry["warm"]["mae"]["mean"]
            entry["delta_mae_warm_minus_cold"] = (
                round(warm_mae - cold_mae, 4)
                if cold_mae is not None and warm_mae is not None
                else None
            )
            per_season[str(season)] = entry

        # Walk-forward aggregate (mean±std across origins), per arm per seed,
        # then averaged over seeds — the "instead of just the mean/STD" view.
        walkforward = {}
        for arm in ("cold", "warm"):
            seed_means = []
            for r in pos_runs:
                if r["arm"] != arm:
                    continue
                fold_maes = [f["mae"] for f in r["folds"]]
                if fold_maes:
                    seed_means.append(statistics.fmean(fold_maes))
            walkforward[arm] = _agg(seed_means)

        summary[position] = {
            "per_season": per_season,
            "walkforward_mae_over_origins": walkforward,
        }
    return summary


def _print_table(summary):
    print(f"\n{'=' * 78}")
    print("  WARM-START vs COLD (from-scratch) — total fantasy-point MAE")
    print(f"{'=' * 78}")
    for position, s in summary.items():
        print(f"\n{position}")
        print(f"  {'test season':<14}{'COLD mae':>12}{'WARM mae':>12}{'Δ (warm-cold)':>16}")
        print(f"  {'-' * 52}")
        for season, e in s["per_season"].items():
            cold = e["cold"]["mae"]["mean"]
            warm = e["warm"]["mae"]["mean"]
            delta = e["delta_mae_warm_minus_cold"]
            cold_s = f"{cold:.3f}" if cold is not None else "—"
            warm_s = f"{warm:.3f}" if warm is not None else "—"
            delta_s = f"{delta:+.3f}" if delta is not None else "—"
            flag = ""
            if delta is not None:
                flag = "  (warm better)" if delta < 0 else ("  (cold better)" if delta > 0 else "")
            print(f"  {season:<14}{cold_s:>12}{warm_s:>12}{delta_s:>16}{flag}")
        wf = s["walkforward_mae_over_origins"]
        for arm in ("cold", "warm"):
            a = wf[arm]
            if a["mean"] is not None:
                print(
                    f"  walk-forward {arm.upper():<5} mae (mean±std over origins): "
                    f"{a['mean']:.3f} ± {a['std']:.3f}  (n_seeds={a['n']})"
                )


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["QB", "RB"],
        help="Positions to run (QB/RB/WR/TE only).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123],
        help="Random seeds (mean±std across them).",
    )
    parser.add_argument(
        "--test-seasons",
        nargs="+",
        type=int,
        default=list(ROLLING_ORIGIN_TEST_SEASONS),
        help="Walk-forward test (origin) seasons, ascending. Each origin trains "
        "on [min..T-2], val T-1, test T.",
    )
    parser.add_argument(
        "--epochs-warm",
        type=int,
        default=None,
        help="Override nn_epochs for WARM folds (i>0). Default: same as COLD "
        "(pure weight-carry comparison).",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["cold", "warm"],
        default=["cold", "warm"],
        help="Which arms to run (default both).",
    )
    parser.add_argument("--out-dir", default="warmstart_runs")
    args = parser.parse_args(argv)

    bad = [p for p in args.positions if p.upper() not in SUPPORTED_POSITIONS]
    if bad:
        parser.error(
            f"Unsupported position(s) {bad}; warm-start prototype is QB/RB/WR/TE only "
            "(K/DST use the nested trainer / build their own splits)."
        )
    positions = [p.upper() for p in args.positions]
    test_seasons = sorted(args.test_seasons)

    print(
        f"Warm-start walk-forward: positions={positions} seeds={args.seeds} "
        f"test_seasons={test_seasons} arms={args.arms} epochs_warm={args.epochs_warm}"
    )
    # Set runtime env only now — after validation (so the error path never
    # mutates a caller's/pytest's env) and before any training/feature work.
    _configure_runtime_env()
    full_df = _load_full_featured_frame()

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    sha = _git_sha()
    out_path = f"{args.out_dir}/{stamp}_{sha}_warmstart.json"
    meta = {
        "timestamp_utc": stamp,
        "git_sha": sha,
        "positions": positions,
        "seeds": args.seeds,
        "test_seasons": test_seasons,
        "epochs_warm": args.epochs_warm,
        "arms": args.arms,
        "env": {
            k: os.environ.get(k)
            for k in (
                "FF_CUDA_GRAPH",
                "FF_COMPILE",
                "FF_FEATURE_CACHE_DISABLE",
                "FF_NN_FIXED_EPOCHS",
                "FF_DEVICE",
                "FF_AMP_DTYPE",
            )
        },
        "caveats": (
            "weights-only warm start; scaler + optimizer refit per fold; "
            "warm-arm metrics not (data,seed)-reproducible by design"
        ),
    }

    def _flush(runs):
        # Rewrite the full artifact after every completed arm so a gaming
        # interrupt mid-run still leaves a valid, partial JSON (this box is
        # also a gaming machine — long runs get pre-empted).
        with open(out_path, "w") as fh:
            json.dump({"meta": meta, "runs": runs, "summary": _summarize(runs)}, fh, indent=2)

    runs = []
    for position in positions:
        cfg = get_config(position)
        for seed in args.seeds:
            # Fresh folds per arm — cheap, deterministic, and keeps the slices
            # local so COLD and WARM never share a consumed iterator.
            for arm in args.arms:
                try:
                    fold_metrics = _run_arm(
                        position,
                        cfg,
                        rolling_origin_folds(full_df, test_seasons=test_seasons),
                        seed,
                        warm=(arm == "warm"),
                        epochs_warm=args.epochs_warm,
                    )
                except Exception as exc:  # noqa: BLE001 — surface, keep other cells
                    print(f"!! {position} seed={seed} {arm} FAILED: {exc!r}")
                    continue
                runs.append(
                    {
                        "position": position,
                        "seed": seed,
                        "arm": arm,
                        "folds": fold_metrics,
                    }
                )
                _flush(runs)

    summary = _summarize(runs)
    _print_table(summary)
    _flush(runs)
    print(f"\nWrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
