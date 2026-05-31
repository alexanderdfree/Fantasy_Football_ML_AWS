"""Ablation: does the train-only ``MIN_GAMES_PER_SEASON`` filter help or hurt?

The production pipeline filters TRAIN rows to players with ``>= MIN_GAMES_PER_SEASON``
games in a season ([src/shared/pipeline.py] ``_prepare_position_data_uncached``,
the ``games_per_season >= MIN_GAMES_PER_SEASON`` line) but leaves VAL/TEST
unfiltered. So the model trains only on established players yet is evaluated on
everyone — a train->test covariate shift the split design itself manufactures,
landing on the low-volume / cold-start subgroup the model never saw.

This script sweeps the threshold and reports per-subgroup fantasy-point MAE so we
can see whether relaxing the filter helps the low-volume test rows
(``would_filter``, < 6 games in the test season) and at what cost to the
established rows (``kept``, >= 6 games).

Investigation-only:
  * It monkeypatches the module constant rather than editing production code.
  * It lives in ``src/tuning/`` (no retrain trigger; scope_positions.py).
  * It writes results to ``benchmark_history/ablations/``.

Cache gotcha (the one way this differs from ``ablate_rb_gate.py``): the threshold
is NOT part of the feature-cache key (``src/shared/feature_cache.py`` hashes only
position + frame content + cfg), so a live cache would serve the first variant's
filtered features to every subsequent threshold. This module forces
``FF_FEATURE_CACHE_DISABLE=1`` at import to prevent that.

Two-phase protocol:
    # Phase 1 — deterministic Ridge+LGBM signal, full threshold curve, 1 seed
    python -m src.tuning.ablate_min_games --positions rb wr

    # Phase 2 — confirm the best (NN/attention) model on the decisive pair
    python -m src.tuning.ablate_min_games --positions rb wr --thresholds 1 6 --seeds 8

Notes:
  * ``--seeds N`` runs seeds ``[42, 43, ..., 42+N-1]`` — the SAME list across every
    threshold, so threshold comparisons are paired. Deterministic models (Ridge,
    LightGBM) show std~=0; the seed spread is the NN/attention signal.
  * Default positions are rb/wr (highest low-volume fraction). QB/K/DST are
    expected near-null — the printed train-row count per threshold confirms the
    filter barely bites there (starters/teams play ~full seasons). For K the
    population filter may live in ``src/k/data.py`` rather than the patched
    pipeline constant; trust the train-row delta, not the assumption.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from importlib import import_module
from unittest.mock import patch

# MUST precede any import that pulls in ``src.shared.feature_cache``. The min-games
# threshold is not in the cache key, so a live cache would serve the first
# variant's filtered features to every subsequent threshold. See module docstring.
os.environ.setdefault("FF_FEATURE_CACHE_DISABLE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import SPLITS_DIR  # noqa: E402
from src.shared.benchmark_utils import (  # noqa: E402
    append_to_history,
    get_git_hash,
    utc_now_iso,
)

ABLATION_NAME = "min_games_filter"
HISTORY_DIR = "benchmark_history"
DEFAULT_THRESHOLDS = [1, 3, 6, 8]
DEFAULT_POSITIONS = ["rb", "wr"]
BASE_SEED = 42
# Buckets printed in the headline tables (the rest feed the JSON + finer curve).
KEY_BUCKETS = ["ALL", "would_filter(<6)", "kept(>=6)"]


def _bucket_masks(g: pd.Series) -> dict[str, pd.Series]:
    """Test-side cohorts keyed by the player's game count IN THE TEST SEASON,
    computed with the SAME groupby the train filter uses. ``would_filter`` is the
    subgroup the production threshold (6) strips from TRAIN — the regime the model
    is under-exposed to but still evaluated on."""
    return {
        "ALL": pd.Series(True, index=g.index),
        "would_filter(<6)": g < 6,
        "kept(>=6)": g >= 6,
        "g=1": g == 1,
        "g=2-3": g.between(2, 3),
        "g=4-5": g.between(4, 5),
        "g=6-9": g.between(6, 9),
        "g>=10": g >= 10,
    }


def _load_position(pos: str):
    mod = import_module(f"src.{pos}.run_pipeline")
    return mod.run, mod.CONFIG


def _pred_total_cols(test_df: pd.DataFrame) -> list[str]:
    """Per-model fantasy-point total columns attached by run_pipeline
    (``pred_ridge_total``, ``pred_nn_total``, ``pred_lgbm_total``,
    ``pred_attn_nn_total``, ...). Discovered dynamically so the table adapts to
    whichever models trained. ``pred_baseline`` lacks the ``_total`` suffix and is
    excluded."""
    return sorted(c for c in test_df.columns if c.startswith("pred_") and c.endswith("_total"))


def _short(pred_col: str) -> str:
    return pred_col[len("pred_") : -len("_total")]


def _subgroup_mae(test_df: pd.DataFrame, targets, agg) -> dict[str, dict]:
    """Per-(bucket, model) fantasy-point MAE on the test set.

    The actual FP total reuses ``cfg["aggregate_fn"]`` applied to the actual stat
    columns — the same ``_total`` aggregator run_pipeline uses for predictions — so
    actual and predicted totals are scored like-for-like (production scoring, not a
    reimplementation). MAE of a vector is computed directly; that is identical to
    ``compute_metrics(...)["mae"]`` but avoids r2-on-tiny-bucket noise.
    """
    g = test_df.groupby(["player_id", "season"])["week"].transform("count")
    if agg is not None:
        actual_total = np.asarray(
            agg({t: test_df[t].to_numpy(dtype=float) for t in targets}), dtype=float
        )
    else:  # no aggregator registered — fall back to sum(heads) (skill-position shape)
        actual_total = np.sum([test_df[t].to_numpy(dtype=float) for t in targets], axis=0)

    pred_cols = _pred_total_cols(test_df)
    out: dict[str, dict] = {}
    for bname, mask in _bucket_masks(g).items():
        m = mask.to_numpy()
        n = int(m.sum())
        rec: dict[str, float] = {"n": n}
        if n:
            at = actual_total[m]
            for pc in pred_cols:
                rec[pc] = float(np.mean(np.abs(test_df[pc].to_numpy(dtype=float)[m] - at)))
        out[bname] = rec
    return out


def _train_rows_by_threshold(cfg, thresholds) -> tuple[dict[int, int], bool]:
    """Surviving TRAIN rows at each threshold — replicates the production filter
    (filter-to-position, then ``groupby(player_id, season).week.count() >= thr``) so
    the table shows how much each threshold strips.

    Returns ``(counts, reliable)``. ``reliable`` is False for K/DST: their
    ``filter_fn`` is identity on the shared (skill-only) ``train.parquet`` because
    K/DST load their real training data separately — so the count would reflect the
    *skill* split, not the position's true train population. For those, trust the
    per-bucket *test* deltas (which come from the real pipeline run), not this count.
    QB/RB/WR/TE counts are the true post-filter train sizes.
    """
    train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
    if "season_type" in train_df.columns:
        train_df = train_df[train_df["season_type"] == "REG"]
    pos_train = cfg["filter_fn"](train_df)
    reliable = len(pos_train) < len(train_df)  # filter_fn actually subset to a position
    g = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    return {int(thr): int((g >= thr).sum()) for thr in thresholds}, reliable


def run_one(pos: str, run_fn, cfg, threshold: int, seed: int, run_config=None) -> dict:
    """One production pipeline run with the train filter monkeypatched.

    ``run_config`` overrides the module CONFIG passed to ``run()``. The Phase-1
    fast path passes a copy with the NN/attention heads disabled — the deterministic
    Ridge + LightGBM stay at full production config (this is a skip, not a weakened
    proxy), so their per-bucket MAE is production-faithful.
    """
    mode = "Ridge+LGBM only" if run_config is not None else "all models"
    print(f"\n{'=' * 72}\n{pos.upper()}  threshold={threshold}  seed={seed}  [{mode}]\n{'=' * 72}")
    # The filter reads ``src.shared.pipeline.MIN_GAMES_PER_SEASON`` (a module global,
    # bound via ``from src.config import ...``); patching that path rebinds exactly
    # what ``_prepare_position_data_uncached`` reads. The filter sits downstream of
    # frame injection, so passing frames to run() cannot achieve this.
    with patch("src.shared.pipeline.MIN_GAMES_PER_SEASON", threshold):
        result = run_fn(seed=seed, config=run_config)
    test_df = result["test_df"]
    buckets = _subgroup_mae(test_df, cfg["targets"], cfg.get("aggregate_fn"))
    pred_cols = _pred_total_cols(test_df)
    if run_config is not None:
        # Skip-nn fast path: the base NN ran at minimal epochs (degraded) and
        # attention was skipped — report only the deterministic Ridge/LightGBM/ENet.
        pred_cols = [c for c in pred_cols if "nn" not in _short(c)]
        buckets = {
            b: {k: v for k, v in rec.items() if k == "n" or k in pred_cols}
            for b, rec in buckets.items()
        }
    return {"buckets": buckets, "pred_cols": pred_cols}


def _aggregate_seeds(seed_runs: list[dict]) -> tuple[dict, list[str]]:
    """Collapse per-seed bucket dicts to mean/std per (bucket, model)."""
    pred_cols = seed_runs[0]["pred_cols"]
    agg: dict[str, dict] = {}
    for bname in seed_runs[0]["buckets"]:
        rec: dict = {"n": seed_runs[0]["buckets"][bname]["n"]}
        for pc in pred_cols:
            vals = [sr["buckets"][bname][pc] for sr in seed_runs if pc in sr["buckets"][bname]]
            if vals:
                rec[pc] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_seeds": len(vals),
                }
        agg[bname] = rec
    return agg, pred_cols


def _fmt(cell: dict | None, multi_seed: bool) -> str:
    if cell is None:
        return f"{'-':>11}"
    if multi_seed:
        return f"{cell['mean']:>7.3f}±{cell['std']:.3f}"
    return f"{cell['mean']:>11.3f}"


def print_position_summary(
    pos: str,
    train_rows: dict,
    per_threshold: dict,
    baseline_threshold: int,
    train_rows_reliable: bool = True,
) -> None:
    thresholds = sorted(per_threshold)
    agg_by_thr = {thr: _aggregate_seeds(per_threshold[thr])[0] for thr in thresholds}
    pred_cols = _aggregate_seeds(per_threshold[thresholds[0]])[1]
    multi_seed = len(per_threshold[thresholds[0]]) > 1
    models = [_short(pc) for pc in pred_cols]

    print(f"\n{'#' * 96}")
    print(
        f"# {pos.upper()} — min-games filter ablation"
        f"{'  (mean±std over seeds)' if multi_seed else ''}"
    )
    print(f"{'#' * 96}")
    caveat = (
        ""
        if train_rows_reliable
        else "  (UNRELIABLE — filter_fn identity on shared skill-split; trust test deltas)"
    )
    print(
        "train rows by threshold: "
        + "  ".join(f"{thr}={train_rows[thr]}" for thr in thresholds)
        + caveat
    )

    for bucket in KEY_BUCKETS:
        n = agg_by_thr[thresholds[0]][bucket]["n"]
        print(f"\n  bucket = {bucket}   [test n={n}]")
        header = f"  {'thresh':<8}" + "".join(f"{m:>13}" for m in models)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for thr in thresholds:
            row = agg_by_thr[thr][bucket]
            cells = "".join(_fmt(row.get(pc), multi_seed) + "  " for pc in pred_cols)
            tag = "  <- baseline" if thr == baseline_threshold else ""
            print(f"  {thr:<8}{cells}{tag}")

    # Verdict: Δ = MAE(baseline) - MAE(min threshold). Positive => relaxing helps.
    lo = thresholds[0]
    if baseline_threshold in agg_by_thr and lo != baseline_threshold:
        print(
            f"\n  VERDICT ({pos.upper()}): relaxing filter {baseline_threshold} -> {lo} "
            f"(Δ = MAE@{baseline_threshold} − MAE@{lo}; + = relaxing HELPS)"
        )
        for bucket in ["would_filter(<6)", "kept(>=6)"]:
            base, low = agg_by_thr[baseline_threshold][bucket], agg_by_thr[lo][bucket]
            parts = []
            for pc in pred_cols:
                if pc in base and pc in low:
                    d = base[pc]["mean"] - low[pc]["mean"]
                    noise = (
                        (base[pc]["std"] ** 2 + low[pc]["std"] ** 2) ** 0.5 if multi_seed else 0.0
                    )
                    parts.append(
                        f"{_short(pc)} Δ={d:+.3f}" + (f"±{noise:.3f}" if multi_seed else "")
                    )
            body = "  ".join(parts) if parts else "n=0 (no rows — filter inert for this bucket)"
            print(f"    {bucket:<18} {body}")
        print(
            "    (filter HURTS low-volume if would_filter Δ>0 beyond noise AND kept Δ not meaningfully <0)"
        )


def _write_ablation(args, all_results: dict) -> None:
    now = utc_now_iso()
    git_hash = get_git_hash()
    entry = {
        "run_id": f"{now}_{git_hash}_{ABLATION_NAME}",
        "timestamp": now,
        "git_hash": git_hash,
        "kind": "ablation",
        "name": ABLATION_NAME,
        "thresholds": args.thresholds,
        "baseline_threshold": args.baseline_threshold,
        "seeds": [BASE_SEED + i for i in range(args.seeds)],
        "skip_nn": args.skip_nn,
        "positions": args.positions,
        "results": all_results,
    }
    append_to_history(os.path.join(HISTORY_DIR, "ablations"), entry)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--positions",
        nargs="+",
        default=DEFAULT_POSITIONS,
        help="positions to ablate (default: rb wr)",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=DEFAULT_THRESHOLDS,
        help="MIN_GAMES_PER_SEASON values to sweep (default: 1 3 6 8)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="number of seeds (42..42+N-1), same list across thresholds",
    )
    p.add_argument(
        "--baseline-threshold",
        type=int,
        default=6,
        help="threshold to diff against in the verdict (default: 6, production)",
    )
    p.add_argument(
        "--skip-nn",
        action="store_true",
        help="Phase-1 fast path: train only the production Ridge+LightGBM "
        "(NN/attention skipped, not weakened). Deterministic and ~seed-invariant.",
    )
    p.add_argument(
        "--no-history", action="store_true", help="skip writing benchmark_history/ablations/"
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = [BASE_SEED + i for i in range(args.seeds)]
    all_results: dict[str, dict] = {}
    for pos in args.positions:
        run_fn, cfg = _load_position(pos)
        run_config = None
        if args.skip_nn:
            # Train only the production Ridge + LightGBM at FULL config. run_pipeline
            # assigns ``pred_nn_total`` unconditionally, so the base NN can't be cleanly
            # disabled — it runs at minimal epochs and its degraded preds are dropped
            # from the table in run_one; attention (the expensive path) is skipped.
            run_config = copy.deepcopy(cfg)
            run_config["nn_epochs"] = 2
            run_config["train_attention_nn"] = False
            run_config["train_elasticnet"] = False
        train_rows, train_rows_reliable = _train_rows_by_threshold(cfg, args.thresholds)
        per_threshold = {
            thr: [run_one(pos, run_fn, cfg, thr, s, run_config) for s in seeds]
            for thr in args.thresholds
        }
        all_results[pos] = {
            "train_rows": train_rows,
            "train_rows_reliable": train_rows_reliable,
            "aggregated": {thr: _aggregate_seeds(per_threshold[thr])[0] for thr in args.thresholds},
        }
        print_position_summary(
            pos, train_rows, per_threshold, args.baseline_threshold, train_rows_reliable
        )
    if not args.no_history:
        _write_ablation(args, all_results)


if __name__ == "__main__":
    main()
