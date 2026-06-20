"""Multicollinearity audit for the RB attention-NN feature set.

Loads ``data/splits/train.parquet`` (general features pre-computed by SETUP.md's
first-time data pull), runs ``add_specific_features`` (from ``src.rb.features``)
to materialise the RB-specific engineered columns, and computes:

  1. Pairwise Pearson + Spearman correlations across the full ``INCLUDE_FEATURES``
     set, with a focused subset for ``ATTN_STATIC_FEATURES`` (what the attention
     NN's static branch actually consumes).
  2. Variance Inflation Factor (VIF) for every ``ATTN_STATIC_FEATURES`` column,
     computed as ``1 / (1 - R^2_i)`` where ``R^2_i`` comes from regressing
     feature ``i`` against the rest via sklearn ``LinearRegression``. (Avoids
     adding ``statsmodels`` as a new dep just for one helper.)
  3. Condition number of the standardised static-feature design matrix
     (pre and post the same 80-component PCA Ridge uses) — the QB pre-PCA
     condition number was 1.8e8; this checks whether RB has the same problem.
  4. Pre-registered "by-construction" collinearity checks: every pair the plan
     called out (``opp_def_rank_vs_pos`` ↔ ``opp_fantasy_pts_allowed_to_pos``,
     ``target_share_L3`` ↔ ``target_share_L5``, etc.) — confirms or refutes
     each hypothesis explicitly.

Outputs:

  - ``analysis_output/rb_feature_audit.json``  (machine-readable findings)
  - ``analysis_output/rb_feature_audit_static_corr.png``  (clustered heatmap)

Usage::

    python -m src.analysis.analysis_rb_feature_audit [--splits-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; this script writes PNGs, no GUI needed

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis._feature_stats import (
    _clean_features,
    _high_corr_pairs,
    _print_top,
    _print_vif,
    _save_static_heatmap,
    _spearman_matrix,
    _vif,
)
from src.rb.config import ATTN_STATIC_CATEGORIES, POSITION_CONFIG  # noqa: E402
from src.rb.data import filter_to_position  # noqa: E402
from src.rb.features import add_specific_features, get_feature_columns  # noqa: E402
from src.shared.weather_features import merge_schedule_features  # noqa: E402

ATTN_HISTORY_STATS = POSITION_CONFIG.attn_history_stats
INCLUDE_FEATURES = POSITION_CONFIG.include_features
RIDGE_PCA_COMPONENTS = POSITION_CONFIG.ridge_pca_components
SPECIFIC_FEATURES = POSITION_CONFIG.specific_features
TARGETS = POSITION_CONFIG.targets

OUT_DIR = PROJECT_ROOT / "analysis_output"
OUT_JSON = OUT_DIR / "rb_feature_audit.json"
OUT_HEATMAP = OUT_DIR / "rb_feature_audit_static_corr.png"

# Hypotheses the original audit (PR #190) flagged as "likely redundant by
# construction" — we test each explicitly so the report can confirm or refute,
# rather than burying the result inside the full correlation matrix.
#
# Some of these pairs reference columns that PR #190 then dropped from
# ``INCLUDE_FEATURES`` / ``SPECIFIC_FEATURES`` based on this very audit
# (notably ``target_share_L5``, ``carry_share_L5``, ``opp_def_rank_vs_pos``,
# ``opp_fantasy_pts_allowed_to_pos``, ``weighted_opportunities_L3``). The
# audit's hypothesis-test helper reports "missing column" rather than
# crashing for any pair where one side no longer exists, so re-running this
# script on the post-drop schema is safe and serves as a regression check
# that those columns stay dropped. Pairs where both sides survived (e.g.
# ``team_rb_carry_hhi_L3`` ↔ ``team_rb_target_hhi_L3``) continue to test
# every run.
PRE_REGISTERED_PAIRS = [
    (
        "opp_def_rank_vs_pos",
        "opp_fantasy_pts_allowed_to_pos",
        "rank() of the other (dropped PR #190)",
    ),
    (
        "target_share_L3",
        "target_share_L5",
        "L5 violates the >0.97-corr drop rule (dropped PR #190)",
    ),
    ("carry_share_L3", "carry_share_L5", "L5 violates the >0.97-corr drop rule (dropped PR #190)"),
    ("team_rb_carry_hhi_L3", "team_rb_target_hhi_L3", "both team-level concentration (kept)"),
    (
        "opp_def_pts_allowed_L5",
        "opp_fantasy_pts_allowed_to_pos",
        "different aggregations of opp def quality (sum dropped PR #190)",
    ),
    ("yards_per_carry_L3", "rushing_epa_per_attempt_L3", "two rushing efficiency metrics (kept)"),
    (
        "team_rb_carry_share_L3",
        "team_rb_target_share_L3",
        "two RB usage shares on same team (kept)",
    ),
]


def _attn_static_features() -> list[str]:
    return [c for cat in ATTN_STATIC_CATEGORIES for c in INCLUDE_FEATURES[cat]]


def _present_numeric(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Filter to columns that are present and numeric and have non-zero variance."""
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        # zero-variance cols make corr/VIF undefined and inflate the heatmap
        if df[c].std(ddof=0) == 0 or df[c].nunique(dropna=True) <= 1:
            continue
        out.append(c)
    return out


def _condition_number(df: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    """Return (pre-PCA, post-PCA) condition numbers of standardised ``df[cols]``."""
    sub = df[cols].dropna()
    if len(sub) < 50 or len(cols) < 2:
        return float("nan"), float("nan")
    X = StandardScaler().fit_transform(sub.to_numpy(dtype=float))
    cond_pre = float(np.linalg.cond(X))
    n_components = min(RIDGE_PCA_COMPONENTS, X.shape[1], X.shape[0])
    Xp = PCA(n_components=n_components).fit_transform(X)
    cond_post = float(np.linalg.cond(Xp))
    return cond_pre, cond_post


def _pre_registered_table(df: pd.DataFrame, pairs: list[tuple[str, str, str]]) -> list[dict]:
    out = []
    for a, b, why in pairs:
        if a not in df.columns or b not in df.columns:
            out.append(
                {
                    "a": a,
                    "b": b,
                    "why": why,
                    "pearson": None,
                    "spearman": None,
                    "note": "missing column",
                }
            )
            continue
        sub = df[[a, b]].dropna()
        if len(sub) < 50:
            out.append(
                {
                    "a": a,
                    "b": b,
                    "why": why,
                    "pearson": None,
                    "spearman": None,
                    "note": f"only {len(sub)} non-NaN rows",
                }
            )
            continue
        p = float(sub[a].corr(sub[b]))
        s = float(sub[a].corr(sub[b], method="spearman"))
        out.append({"a": a, "b": b, "why": why, "pearson": p, "spearman": s, "n": int(len(sub))})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Directory containing train.parquet (default: data/splits)",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="Report all pairs with |Pearson| above this (default: 0.85)",
    )
    args = parser.parse_args()

    train_path = args.splits_dir / "train.parquet"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found — run SETUP.md's first-time data pull.")
        return 1

    print(f"Loading {train_path} …")
    train_df = pd.read_parquet(train_path)
    print(f"  full split rows: {len(train_df):,}")

    rb_train = filter_to_position(train_df).copy()
    print(f"  RB rows: {len(rb_train):,}")

    # Mirror build_position_features: weather/Vegas merge happens before
    # add_specific_features in the production pipeline. Without it the
    # weather_vegas category columns would all be missing.
    merge_schedule_features(rb_train, label="train")

    empty = rb_train.iloc[:0].copy()
    rb_train, _, _ = add_specific_features(rb_train, empty.copy(), empty.copy())

    feature_cols = get_feature_columns()
    static_cols = _attn_static_features()
    history_cols = list(ATTN_HISTORY_STATS)

    # Diagnostic: which feature columns are missing from the materialised frame.
    missing = [c for c in feature_cols if c not in rb_train.columns]
    if missing:
        preview = ", ".join(missing[:5]) + (
            f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        )
        print(f"  ⚠ {len(missing)} feature(s) missing from frame: {preview}")

    # Impute NaN/±inf -> 0 to match the production catch-all
    # (build_position_features) — the same full, imputed population the models
    # train on, not the veteran-heavy complete-case subset. See PR #594 / #1218.
    all_audit_cols = (
        list(feature_cols)
        + list(static_cols)
        + list(history_cols)
        + list(TARGETS)
        + list(SPECIFIC_FEATURES)
    )
    rb_imp = _clean_features(rb_train, all_audit_cols)

    full_present = _present_numeric(rb_imp, feature_cols)
    static_present = _present_numeric(rb_imp, static_cols)
    history_present = _present_numeric(rb_imp, history_cols)
    targets_present = _present_numeric(rb_imp, list(TARGETS))
    specific_present = _present_numeric(rb_imp, list(SPECIFIC_FEATURES))

    print(
        f"  features used in audit: full={len(full_present)} "
        f"static={len(static_present)} history={len(history_present)} "
        f"specific={len(specific_present)} targets={len(targets_present)}"
    )

    # ── Correlation matrices ───────────────────────────────────────────────
    print("\nComputing Pearson correlations …")
    full_pearson = rb_imp[full_present].corr(method="pearson")
    static_pearson = rb_imp[static_present].corr(method="pearson")

    print("Computing Spearman correlations (full set, may take a few seconds) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        full_spearman = _spearman_matrix(rb_imp, full_present)

    high_full = _high_corr_pairs(full_pearson, threshold=args.corr_threshold)
    high_static = _high_corr_pairs(static_pearson, threshold=args.corr_threshold)

    # ── VIF + condition number on the static block ─────────────────────────
    print("\nComputing VIF on ATTN_STATIC_FEATURES …")
    vif = _vif(rb_imp, static_present)

    cond_pre, cond_post = _condition_number(rb_imp, static_present)
    print(
        f"  condition number  pre-PCA: {cond_pre:.3g}   post-PCA({RIDGE_PCA_COMPONENTS}): {cond_post:.3g}"
    )

    # ── Pre-registered pairs ───────────────────────────────────────────────
    print("\nPre-registered by-construction pairs:")
    pre_reg = _pre_registered_table(rb_imp, PRE_REGISTERED_PAIRS)
    print(f"    {'a':<42s} {'b':<42s} {'pearson':>9s} {'spearman':>10s}")
    for row in pre_reg:
        p = "—" if row.get("pearson") is None else f"{row['pearson']:9.3f}"
        s = "—" if row.get("spearman") is None else f"{row['spearman']:10.3f}"
        print(f"    {row['a'][:42]:<42s} {row['b'][:42]:<42s} {p:>9s} {s:>10s}")

    # ── Targets vs features (high signal sanity check) ─────────────────────
    print("\nTop 10 features by |Pearson| with rushing_yards (sanity check):")
    if "rushing_yards" in rb_imp.columns:
        rys = rb_imp[full_present + ["rushing_yards"]].corr()["rushing_yards"].drop("rushing_yards")
        top_ry = rys.abs().sort_values(ascending=False).head(10)
        for f, _v in top_ry.items():
            print(f"    {f[:48]:<48s} {rys[f]:>7.3f}")

    # ── Stdout summaries ───────────────────────────────────────────────────
    _print_top(high_full, k=20, label=f"full set, |r| ≥ {args.corr_threshold}")
    _print_top(high_static, k=15, label=f"ATTN_STATIC subset, |r| ≥ {args.corr_threshold}")
    _print_vif(vif)

    # ── Persist outputs ────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "n_rows_rb_train": int(len(rb_train)),
        "n_features_full": len(full_present),
        "n_features_static": len(static_present),
        "n_features_history": len(history_present),
        "missing_from_frame": missing,
        "corr_threshold": args.corr_threshold,
        "high_corr_pairs_full": [
            {
                "a": a,
                "b": b,
                "pearson": r,
                "spearman": float(full_spearman.loc[a, b])
                if a in full_spearman.index and b in full_spearman.columns
                else None,
            }
            for a, b, r in high_full
        ],
        "high_corr_pairs_static": [{"a": a, "b": b, "pearson": r} for a, b, r in high_static],
        "vif_static": vif,
        "condition_number_static": {
            "pre_pca": cond_pre,
            "post_pca": cond_post,
            "pca_components": RIDGE_PCA_COMPONENTS,
        },
        "pre_registered_pairs": pre_reg,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"\n✔ wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    _save_static_heatmap(static_pearson, OUT_HEATMAP, "RB")
    print(f"✔ wrote {OUT_HEATMAP.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
