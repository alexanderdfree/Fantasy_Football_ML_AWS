"""Ridge-only Principal-Component-Regression (PCR) ablation for any position.

WHY THIS EXISTS
---------------
The 2026-05 feature-collinearity audit (PR #594, ``analysis_feature_audit.py``)
showed QB/TE feed a ~1e15-condition matrix straight into Ridge (no
``ridge_pca_components`` set), while WR/RB/DST run PCA-before-Ridge. Open
question: does adding PCA-before-Ridge to QB/TE actually lower test MAE, or does
tuned Ridge's L2 already handle the conditioning so PCA only adds bias?

PCA feeds ONLY the Ridge path (verified: NN/attention/LightGBM consume the raw
scaled+clipped features — see ``src/shared/pipeline.py`` ``_cpu_branch`` vs
``_gpu_branch``). Ridge is deterministic. So a Ridge-only A/B at several
``pca_n`` is an EXACT validation — no NN seed noise, no GPU needed.

WHY IT GOES THROUGH ``run_pipeline`` (do NOT hand-roll the feature matrix)
-------------------------------------------------------------------------
A standalone sweep that loads the split + calls ``add_specific_features``
directly (like ``src/wr/benchmark_ridge_variants.py``) MISSES the
weather/Vegas/contextual columns (``is_dome``, ``implied_opp_total``,
``wind_adjusted``, ``is_divisional``, ``temp_adjusted``). Those are merged in by
``merge_schedule_features`` INSIDE ``build_position_features`` at pipeline time —
they are NOT in the base parquet splits. A hand-rolled sweep crashes with
``KeyError: ['is_dome', ...] not in index`` (or, worse, silently audits a
matrix missing 5 production features). This harness calls the real
``run_pipeline`` with a Ridge-only config override, so the feature matrix is
byte-identical to what production Ridge sees. (This is the training/inference
drift CLAUDE.md warns about — don't reintroduce it.)

PERFORMANCE — READ BEFORE RUNNING (this once hung for an hour)
-------------------------------------------------------------
``_tune_ridge_alphas_cv`` fans every alpha over ``joblib.Parallel(n_jobs=-1,
prefer="threads")``, and each Ridge solve calls multi-threaded BLAS. Run
WITHOUT thread caps, that oversubscribes (joblib threads x BLAS threads) and
thrashes. ALWAYS run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 python -m src.tuning.ablate_ridge_pca QB TE

With caps, one Ridge-only ``run_pipeline`` call is ~30-90s (alpha tuning is the
cost). The full default sweep = 2 positions x N pca_n x 2 seasons (val+test)
calls; budget accordingly, or narrow ``--pca-grid``.

DATA — needs CURRENT splits (the local symlink is usually stale)
---------------------------------------------------------------
Reads ``data/splits/{train,val,test}.parquet`` (relative ``SPLITS_DIR``). In a
worktree ``data/splits`` is typically a symlink to the parent's STALE shared
splits (missing recently-added features -> ``KeyError`` in feature building).
Rebuild fresh splits into a LOCAL dir first (do NOT write through the symlink —
it corrupts the parent's shared data):

    test -L data/splits && rm data/splits        # remove ONLY the symlink
    mkdir -p data/splits
    OPENBLAS_NUM_THREADS=1 python - <<'PY'
    from src.data.loader import load_raw_data
    from src.data.preprocessing import preprocess
    from src.features.engineer import build_features
    from src.data.split import temporal_split
    temporal_split(build_features(preprocess(load_raw_data())))   # uses cached data/raw
    PY

Restore the symlink afterwards if you want the shared splits back.

VAL + TEST (robustness against single-season noise)
---------------------------------------------------
``run_pipeline`` evaluates Ridge on whatever ``test_df`` you pass, and Ridge
fits on TRAIN only (val is for NN early-stopping, which is off here). So this
harness gets BOTH holdout seasons faithfully: once with the real 2025 test, and
once passing the 2024 val frame AS the test frame. A pca_n only "wins" if it
beats the no-PCA baseline on BOTH seasons (don't headline a single-season flip).

DECIDING / SHIPPING
-------------------
Report is val_MAE + test_MAE at each pca_n, delta vs None. Ship a config change
(add ``ridge_pca_components=<n>`` to ``src/{pos}/config.py``) ONLY if a single
pca_n improves BOTH seasons by more than benchmark noise AND Ridge is the served
(best) model for that position in the latest ``benchmark_history`` — otherwise it
burns a full GPU retrain (config edits scope a full retrain; there is no
Ridge-only production retrain path) for a gain the served prediction never sees.
See the TODO.md note "[OPEN] PCA-before-Ridge for QB/TE" for the full handoff.

Usage::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -m src.tuning.ablate_ridge_pca QB TE
    # fast plumbing check (shrunk alpha search, not real numbers):
    OPENBLAS_NUM_THREADS=1 python -m src.tuning.ablate_ridge_pca QB --smoke
    # custom grid:
    OPENBLAS_NUM_THREADS=1 python -m src.tuning.ablate_ridge_pca TE --pca-grid 55 40 30
"""

from __future__ import annotations

import argparse
import importlib
import sys

import pandas as pd

from src.config import SPLITS_DIR
from src.shared.pipeline import run_pipeline
from src.shared.position_pipeline import build_pipeline_config

# Default pca_n values to sweep (None = no-PCA baseline, always first). Chosen
# around the audit's hypothetical-PCA recommendation (~53-56 components @ 99%
# variance for QB/TE) plus a spread of more aggressive truncations.
DEFAULT_PCA_GRID = [None, 80, 60, 55, 50, 40, 30, 20]


def _ridge_only_cfg(pos: str) -> dict:
    """Build the production cfg dict for ``pos`` with every non-Ridge model off.

    PCA feeds only Ridge, so disabling the NN/attention/LightGBM/ElasticNet
    leaves Ridge's inputs untouched and makes the A/B exact (and GPU-free).
    """
    posmod = importlib.import_module(f"src.{pos.lower()}.config")
    cfg = build_pipeline_config(pos, posmod.POSITION_CONFIG)
    cfg["train_base_nn"] = False
    cfg["train_attention_nn"] = False
    cfg["train_lightgbm"] = False
    cfg["train_elasticnet"] = False
    return cfg


def _shrink_for_smoke(cfg: dict) -> None:
    """In-place: collapse the alpha search so a plumbing check runs in seconds.

    Smoke numbers are NOT valid metrics — they only prove the harness imports,
    builds the real (weather-merged) feature matrix, fits Ridge, and emits a
    ``ridge_metrics['total']['mae']`` without crashing. The single fixed alpha
    also handicaps the no-PCA baseline (no-PCA Ridge needs a larger alpha to
    fight collinearity), so a smoke "improvement" is an artifact — never ship on
    it.
    """
    cfg["ridge_cv_folds"] = 2
    cfg["ridge_refine_points"] = 0
    grids = cfg.get("ridge_alpha_grids", {})
    cfg["ridge_alpha_grids"] = {
        t: [grid[len(grid) // 2]] if grid else [1.0] for t, grid in grids.items()
    }


def _ridge_total_mae(pos: str, cfg: dict, train, val, test_frame) -> float:
    """Run a Ridge-only pipeline and return total-target test MAE on ``test_frame``."""
    result = run_pipeline(pos, cfg, train, val, test_frame, seed=42)
    rm = result.get("ridge_metrics")
    if not rm or "total" not in rm:
        raise RuntimeError(
            f"{pos}: run_pipeline returned no ridge_metrics['total'] "
            f"(keys={list(rm) if rm else None}). Did train_ridge get disabled?"
        )
    return float(rm["total"]["mae"])


def sweep_position(pos: str, pca_grid: list[int | None], smoke: bool) -> list[dict]:
    train = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
    val = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
    test = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    rows: list[dict] = []
    base_val = base_test = None
    tag = "  [SMOKE]" if smoke else ""
    print(f"\n{'=' * 78}\n=== Ridge-PCR ablation: {pos}{tag} ===\n{'=' * 78}")
    print(f"{'pca_n':>6} {'val_MAE':>10} {'val_dN':>9} {'test_MAE':>10} {'test_dN':>9}")
    print("-" * 48)
    for pca_n in pca_grid:
        cfg = _ridge_only_cfg(pos)
        if smoke:
            _shrink_for_smoke(cfg)
        cfg["ridge_pca_components"] = pca_n
        # Eval on 2024 (val-as-test) and 2025 (real test); Ridge fits train only.
        # NOTE: this re-tunes alphas + refits for each frame (2x the work per
        # pca_n). That's deliberate — folding it into one fit would mean
        # predicting outside run_pipeline, which is exactly the weather/Vegas
        # merge bypass this harness exists to avoid (see module docstring). The
        # faithful feature path is worth the extra Ridge fit.
        val_mae = _ridge_total_mae(pos, cfg, train, val, val)
        test_mae = _ridge_total_mae(pos, cfg, train, val, test)
        if base_val is None:
            base_val, base_test = val_mae, test_mae
        label = "None" if pca_n is None else str(pca_n)
        vd = "" if pca_n is None else f"{val_mae - base_val:+.4f}"
        td = "" if pca_n is None else f"{test_mae - base_test:+.4f}"
        print(f"{label:>6} {val_mae:>10.4f} {vd:>9} {test_mae:>10.4f} {td:>9}")
        rows.append({"pos": pos, "pca_n": pca_n, "val_mae": val_mae, "test_mae": test_mae})

    # Consistent improver = beats None on BOTH seasons.
    consistent = [
        r
        for r in rows
        if r["pca_n"] is not None and r["val_mae"] < base_val and r["test_mae"] < base_test
    ]
    if consistent:
        best = min(consistent, key=lambda r: r["test_mae"])
        print(
            f"  -> consistent improvers: {[r['pca_n'] for r in consistent]}; "
            f"best by test MAE = pca_n={best['pca_n']} "
            f"(val {best['val_mae'] - base_val:+.4f}, test {best['test_mae'] - base_test:+.4f})"
        )
    else:
        print("  -> NO pca_n beats None on BOTH val and test (PCA not worth shipping here).")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("positions", nargs="+", help="Positions to ablate (e.g. QB TE).")
    ap.add_argument(
        "--pca-grid",
        nargs="+",
        type=int,
        default=None,
        help="pca_n values to sweep (baseline None is always prepended). "
        f"Default: {DEFAULT_PCA_GRID[1:]}",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Shrink alpha search for a fast plumbing check (numbers NOT valid).",
    )
    args = ap.parse_args()

    grid = [None, *args.pca_grid] if args.pca_grid else DEFAULT_PCA_GRID
    for pos in args.positions:
        sweep_position(pos.upper(), grid, args.smoke)
    print("\nABLATION_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
