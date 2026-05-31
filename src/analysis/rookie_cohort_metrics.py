"""Are rookies a harder cohort for the *selected* model? A tracked subgroup metric.

Rookies are a cold-start case: every history-dependent feature
(``prior_season_*``, rolling/ewma/trend, the attention sequence) is NaN -> 0 in
their first NFL weeks (``src/shared/feature_build.py``), so the model cannot tell
a #1-overall pick from a UDFA — it only *reacts* to in-season accumulation. The
draft-capital / combine fix for this was implemented and reverted (PR #519,
``TODO.md`` ``[TESTED, REJECTED]``): benchmark-flat, with the benefit concentrated
in LightGBM (the best model only for RB) and invisible in overall MAE because
rookies are ~14% of rows. The stated precondition for ever revisiting it was *a
tracked rookie-subgroup metric*. This is that metric.

It runs the real pipeline per position and reports literal per-model
(Ridge / NN / Attention NN / LightGBM) MAE + signed bias on rookies vs veterans,
reusing ``src.shared.error_analysis.compute_stratum_metrics`` — then names the
*best* model per position and its rookie gap, because a subgroup gain only matters
if it reaches the model that is actually selected (the core #519 lesson).

Rookie cohort (source-independent, no new data dependency): a test row is a
rookie iff the player's earliest season anywhere in the splits equals its season.
Data spans 2012-2025, so any player first appearing in the test season is an
unambiguous rookie — exactly the all-``prior_season_*``-NaN rows noted in
``src/analysis/feature_audit_findings.md``. This sidesteps the post-imputation
ambiguity (after ``fillna(0)`` a rookie's zeros are indistinguishable from a
veteran's genuine zeros on ``test_df`` alone). Rookies split into the first 3
games played (``rookie_early`` — where #519 saw the largest effect) and the rest.

Two modes:
* **default:** load splits once, run the pipeline for each ``--positions`` (the
  four skill positions by default — rookie cold-start is a skill-position concern
  and #519's numbers were QB/RB/WR/TE), and print the per-model cohort table.
* **``--no-model``:** instant — just cohort sizes and realized-FP means from the
  splits, no torch. A quick check that rookies are ~14% of rows.

``run()`` is imported lazily inside ``main()`` so importing this module never
fires training/torch. The pure helpers are unit-tested in
``tests/analysis/test_rookie_cohort_metrics.py``.

This is a read-only operator diagnostic: it lives under ``src/analysis/`` and
imports shared helpers read-only, so it touches no training-path file and fires
no retrain.

Usage:
    python -m src.analysis.rookie_cohort_metrics --no-model
    python -m src.analysis.rookie_cohort_metrics --positions RB
    python -m src.analysis.rookie_cohort_metrics
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import POSITIONS, SPLITS_DIR, TEST_SEASONS  # noqa: E402
from src.shared.error_analysis import compute_stratum_metrics  # noqa: E402

# Per-model total-FP prediction columns written by src/shared/pipeline.py. Only
# the ones present on a given position's test_df are used (enet/lgbm/attn are
# per-position toggles), so listing all is safe — absent columns are skipped.
MODELS = {
    "Ridge": "pred_ridge_total",
    "ElasticNet": "pred_enet_total",
    "NN": "pred_nn_total",
    "Attention NN": "pred_attn_nn_total",
    "LightGBM": "pred_lgbm_total",
}
ACTUAL = "fantasy_points"  # pipeline truth column for skill positions (pred_*_total vs this)

# rookie_bucket stratum labels.
ROOKIE_EARLY = "rookie_early"
ROOKIE_REST = "rookie_rest"
VETERAN = "veteran"
UNKNOWN = "unknown"
ROOKIE_BUCKET = "rookie_bucket"

EARLY_GAMES = 3  # first N games played of the rookie season -> rookie_early
DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE"]  # skill positions; K/DST out of scope (see docstring)

_GRP = ["player_id", "season"]


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def player_min_season(frames: list[pd.DataFrame | None]) -> pd.Series:
    """Map ``player_id`` -> earliest season seen across the given frames.

    A player whose earliest season equals their test season is a rookie in that
    season. Frames missing the needed columns (or ``None``) are skipped.
    """
    parts = [
        f[["player_id", "season"]]
        for f in frames
        if f is not None and "player_id" in f.columns and "season" in f.columns
    ]
    if not parts:
        return pd.Series(dtype="int64", name="season")
    return pd.concat(parts, ignore_index=True).groupby("player_id")["season"].min()


def label_rookie_rows(
    test_df: pd.DataFrame,
    min_season: pd.Series,
    early_games: int = EARLY_GAMES,
) -> pd.Series:
    """Label each pipeline ``test_df`` row ``rookie_early`` / ``rookie_rest`` /
    ``veteran``.

    A row is a rookie iff its season equals the player's earliest season in
    ``min_season``. Among rookie rows, the first ``early_games`` games played (by
    week order within the rookie season) are ``rookie_early``, the rest
    ``rookie_rest``. Degrades to ``unknown`` for every row if a required column is
    missing or ``min_season`` is empty (mirrors the defensive style of
    ``error_analysis.add_stratification_columns``).
    """
    needed = ["player_id", "season", "week"]
    if any(c not in test_df.columns for c in needed) or min_season is None or len(min_season) == 0:
        return pd.Series([UNKNOWN] * len(test_df), index=test_df.index, dtype=object)

    debut = test_df["player_id"].map(min_season)
    is_rookie = (test_df["season"].eq(debut) & debut.notna()).to_numpy()

    # Game-played index within each (player, season), by week order. Computed via
    # a positional id so it is robust to any test_df index (non-unique / sorted).
    tmp = test_df[_GRP + ["week"]].copy()
    tmp["_pos"] = np.arange(len(tmp))
    tmp = tmp.sort_values(_GRP + ["week"])
    tmp["_game_idx"] = tmp.groupby(_GRP).cumcount()
    game_idx = tmp.sort_values("_pos")["_game_idx"].to_numpy()

    out = np.where(
        ~is_rookie,
        VETERAN,
        np.where(game_idx < early_games, ROOKIE_EARLY, ROOKIE_REST),
    )
    return pd.Series(out, index=test_df.index, dtype=object)


def available_models(df: pd.DataFrame, models: dict[str, str] | None = None) -> dict[str, str]:
    """The subset of ``MODELS`` whose prediction column is present on ``df``."""
    models = models or MODELS
    return {name: col for name, col in models.items() if col in df.columns}


def bias_corrected_mae(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str
) -> pd.Series:
    """Per-group ``mean(|e − mean(e)|)`` where ``e = pred − actual`` — the MAE that
    survives after each cohort's mean bias is removed (what a perfect additive
    calibration would leave). So ``raw_mae − bias_corrected_mae`` is the systematic,
    calibration-recoverable portion of the error; what remains is irreducible
    spread. Indexed by ``group_col`` value. Companion to the ``MSE = bias² + Var``
    decomposition: MAE alone cannot split bias from spread, this does.
    """
    tmp = df[[group_col, y_true_col, y_pred_col]].dropna().copy()
    tmp["_e"] = tmp[y_pred_col] - tmp[y_true_col]
    centered = tmp["_e"] - tmp.groupby(group_col, observed=True)["_e"].transform("mean")
    tmp["_abs_centered"] = centered.abs()
    return tmp.groupby(group_col, observed=True)["_abs_centered"].mean()


def cohort_model_table(df: pd.DataFrame, models: dict[str, str] | None = None) -> pd.DataFrame:
    """Per-model total-FP ``n`` / ``mae`` / ``mae_corr`` / ``rmse`` / ``bias`` on each
    ``rookie_bucket``, via ``error_analysis.compute_stratum_metrics`` + ``bias_corrected_mae``.
    ``mae_corr`` is the MAE after removing the cohort's mean bias, so ``mae − mae_corr``
    is the calibration-recoverable error. Expects a ``rookie_bucket`` column and the
    ``pred_*_total`` columns on ``df``.
    """
    models = available_models(df, models)
    out = []
    for name, col in models.items():
        metrics = compute_stratum_metrics(df, ACTUAL, col, ROOKIE_BUCKET)
        corr = bias_corrected_mae(df, ACTUAL, col, ROOKIE_BUCKET)
        for _, r in metrics.iterrows():
            out.append(
                {
                    "model": name,
                    "bucket": str(r[ROOKIE_BUCKET]),
                    "n": int(r["n"]),
                    "mae": r["mae"],
                    "mae_corr": float(corr.get(r[ROOKIE_BUCKET], float("nan"))),
                    "rmse": r["rmse"],
                    "bias": r["bias"],
                }
            )
    return pd.DataFrame(out)


def best_model(df: pd.DataFrame, models: dict[str, str] | None = None) -> tuple[str | None, float]:
    """``(name, overall_mae)`` of the lowest-overall-MAE model present on ``df`` —
    the model actually selected for this position. ``(None, nan)`` if none present.
    """
    models = available_models(df, models)
    best_name, best_mae = None, float("inf")
    for name, col in models.items():
        mae = (df[col] - df[ACTUAL]).abs().mean()
        if mae < best_mae:
            best_name, best_mae = name, mae
    return best_name, (best_mae if best_name is not None else float("nan"))


def rookie_gap(df: pd.DataFrame, model_col: str) -> tuple[float, float, float]:
    """``(rookie_mae, veteran_mae, rookie_early_mae)`` for one model column,
    using the ``rookie_bucket`` labels on ``df``. NaN where a cohort is empty.
    """

    def _mae(mask: pd.Series) -> float:
        if not mask.any():
            return float("nan")
        return (df.loc[mask, model_col] - df.loc[mask, ACTUAL]).abs().mean()

    bucket = df[ROOKIE_BUCKET]
    return (
        _mae(bucket.isin([ROOKIE_EARLY, ROOKIE_REST])),
        _mae(bucket.eq(VETERAN)),
        _mae(bucket.eq(ROOKIE_EARLY)),
    )


# --------------------------------------------------------------------------- #
# Presentation (IO; not unit-tested)
# --------------------------------------------------------------------------- #
def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _print_cohort_sizes(
    test_df: pd.DataFrame, min_season: pd.Series, positions: list[str], early_games: int
) -> None:
    _hr(f"ROOKIE COHORT SIZES (test season {TEST_SEASONS}) — data-only, no model")
    test = test_df.copy()
    if "position" in test.columns:
        test = test[test["position"].isin(positions)].copy()
    test[ROOKIE_BUCKET] = label_rookie_rows(test, min_season, early_games)

    rmask = test[ROOKIE_BUCKET].isin([ROOKIE_EARLY, ROOKIE_REST])
    n, rk = len(test), int(rmask.sum())
    pct = rk / n if n else float("nan")
    print(f"  all {'/'.join(positions)}: {n} rows, rookies {rk} ({pct:.0%})  [expect ~14%]")

    print(f"\n  {'pos':<5}{'rows':>7}{'rookies':>9}{'%':>6}{'rookie FP':>12}{'vet FP':>9}")
    for pos in positions:
        sub = test[test["position"] == pos] if "position" in test.columns else test
        if not len(sub):
            continue
        sub_rmask = sub[ROOKIE_BUCKET].isin([ROOKIE_EARLY, ROOKIE_REST])
        srk = int(sub_rmask.sum())
        r_fp = sub.loc[sub_rmask, ACTUAL].mean() if srk and ACTUAL in sub.columns else float("nan")
        vmask = sub[ROOKIE_BUCKET].eq(VETERAN)
        v_fp = (
            sub.loc[vmask, ACTUAL].mean() if vmask.any() and ACTUAL in sub.columns else float("nan")
        )
        print(f"  {pos:<5}{len(sub):>7}{srk:>9}{(srk / len(sub)):>6.0%}{r_fp:>12.2f}{v_fp:>9.2f}")
    print("\n  (realized rookie FP should sit below veteran FP — rookies score less on average.)")


def _print_model_error(
    pos: str, test_df: pd.DataFrame, min_season: pd.Series, early_games: int
) -> None:
    _hr(f"{pos}: PER-MODEL ERROR on rookie vs veteran (test set)")
    df = test_df.copy()
    df[ROOKIE_BUCKET] = label_rookie_rows(df, min_season, early_games)
    counts = df[ROOKIE_BUCKET].value_counts().to_dict()
    print(f"  cohort sizes: {counts}")
    if counts.get(ROOKIE_EARLY, 0) + counts.get(ROOKIE_REST, 0) == 0:
        print("  No rookie rows in the test set — nothing to score.")
        return

    models = available_models(df)
    tbl = cohort_model_table(df, models)
    print(f"\n  {'model':<14}{'bucket':<13}{'N':>5}{'MAE':>8}{'MAEbc':>8}{'RMSE':>8}{'bias':>8}")
    for _, r in tbl.sort_values(["model", "bucket"]).iterrows():
        print(
            f"  {r['model']:<14}{r['bucket']:<13}{int(r['n']):>5}"
            f"{r['mae']:>8.3f}{r['mae_corr']:>8.3f}{r['rmse']:>8.3f}{r['bias']:>+8.3f}"
        )

    best_name, best_mae = best_model(df, models)
    if best_name is not None:
        r_mae, v_mae, e_mae = rookie_gap(df, models[best_name])
        e_corr = float(
            bias_corrected_mae(df, ACTUAL, models[best_name], ROOKIE_BUCKET).get(
                ROOKIE_EARLY, float("nan")
            )
        )
        print(
            f"\n  => best model = {best_name} (overall MAE {best_mae:.3f}); "
            f"rookie {r_mae:.3f} vs veteran {v_mae:.3f} (Δ {r_mae - v_mae:+.3f}); "
            f"rookie_early MAE {e_mae:.3f} → {e_corr:.3f} bias-corrected "
            f"({e_mae - e_corr:+.3f} recoverable by calibration)."
        )
    print(
        "  (MAEbc = MAE after removing each cohort's mean bias; MAE − MAEbc = the systematic "
        "error a rookie calibration could remove. RMSE² = bias² + Var, so RMSE surfaces bias "
        "that MAE blends into spread — both are invisible in the overall-MAE benchmark.)"
    )


def _load_splits() -> list[pd.DataFrame]:
    """Load the train/val/test split parquets, REG-only — mirrors
    ``pipeline._read_split`` without importing the torch-laden pipeline module, so
    ``--no-model`` stays fast. The same frames are passed to every position's
    ``run()`` so cohort labels and predictions share identical data.
    """
    frames = []
    for name in ("train", "val", "test"):
        df = pd.read_parquet(f"{SPLITS_DIR}/{name}.parquet")
        if "season_type" in df.columns:
            df = df[df["season_type"] == "REG"].copy()
        frames.append(df)
    return frames


def _load_run(pos: str):
    """Lazily import a position's ``run()`` (deferred so a bare module import never
    pulls the pipeline / torch)."""
    return importlib.import_module(f"src.{pos.lower()}.run_pipeline").run


def main() -> None:
    parser = argparse.ArgumentParser(description="Rookie cold-start subgroup metric")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=DEFAULT_POSITIONS,
        help="positions to score (default: skill positions QB RB WR TE)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="data-only: cohort sizes + realized FP means from the splits, no pipeline/torch",
    )
    parser.add_argument("--seed", type=int, default=42, help="pipeline seed (default 42)")
    parser.add_argument(
        "--early-games",
        type=int,
        default=EARLY_GAMES,
        help=f"first N games played -> rookie_early (default {EARLY_GAMES})",
    )
    args = parser.parse_args()

    positions = [p.upper() for p in args.positions]
    unknown = [p for p in positions if p not in POSITIONS]
    if unknown:
        parser.error(f"unknown position(s) {unknown}; choose from {POSITIONS}")

    print("Loading splits ...", flush=True)
    train_df, val_df, test_df = _load_splits()
    min_season = player_min_season([train_df, val_df, test_df])

    if args.no_model:
        _print_cohort_sizes(test_df, min_season, positions, args.early_games)
        return

    for pos in positions:
        run = _load_run(pos)  # deferred: pulls the full pipeline/torch
        print(f"\nRunning {pos} pipeline for literal per-model error ...", flush=True)
        result = run(train_df, val_df, test_df, args.seed)
        _print_model_error(pos, result["test_df"], min_season, args.early_games)


if __name__ == "__main__":
    main()
