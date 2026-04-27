"""Multicollinearity audit for the RB attention-NN feature set.

Loads ``data/splits/train.parquet`` (general features pre-computed by SETUP.md's
first-time data pull), runs ``add_rb_specific_features`` to materialise the 14
RB-specific engineered columns, and computes:

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rb.config import (  # noqa: E402
    ATTN_HISTORY_STATS,
    ATTN_STATIC_CATEGORIES,
    INCLUDE_FEATURES,
    RIDGE_PCA_COMPONENTS,
    SPECIFIC_FEATURES,
    TARGETS,
)
from src.rb.data import filter_to_position  # noqa: E402
from src.rb.features import add_specific_features, get_feature_columns  # noqa: E402
from src.shared.weather_features import merge_schedule_features  # noqa: E402

OUT_DIR = PROJECT_ROOT / "analysis_output"
OUT_JSON = OUT_DIR / "rb_feature_audit.json"
OUT_HEATMAP = OUT_DIR / "rb_feature_audit_static_corr.png"

# Hypotheses the plan flagged as "likely redundant by construction" — we test
# each one explicitly so the report can confirm or refute, rather than burying
# the result inside a 43x43 matrix.
PRE_REGISTERED_PAIRS = [
    ("opp_def_rank_vs_pos", "opp_fantasy_pts_allowed_to_pos", "rank() of the other"),
    ("target_share_L3", "target_share_L5", "L5 violates the >0.97-corr drop rule"),
    ("carry_share_L3", "carry_share_L5", "L5 violates the >0.97-corr drop rule"),
    ("team_rb_carry_hhi_L3", "team_rb_target_hhi_L3", "both team-level concentration"),
    (
        "opp_def_pts_allowed_L5",
        "opp_fantasy_pts_allowed_to_pos",
        "different aggregations of opp def quality",
    ),
    ("weighted_opportunities_L3", "opportunity_index_L3", "raw count vs share of same quantity"),
    ("yards_per_carry_L3", "rushing_epa_per_attempt_L3", "two rushing efficiency metrics"),
    ("team_rb_carry_share_L3", "team_rb_target_share_L3", "two RB usage shares on same team"),
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


def _high_corr_pairs(corr: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Return upper-triangle (a, b, r) pairs with |r| >= threshold, sorted by |r| desc."""
    arr = corr.to_numpy()
    rows, cols = np.triu_indices_from(arr, k=1)
    pairs = []
    for i, j in zip(rows, cols, strict=True):
        r = arr[i, j]
        if np.isnan(r):
            continue
        if abs(r) >= threshold:
            pairs.append((corr.index[i], corr.columns[j], float(r)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def _vif(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """VIF_i = 1 / (1 - R^2_i) via sklearn LinearRegression.

    Drops rows with any NaN in ``cols`` first; standardises columns so the R²
    is scale-invariant (matches statsmodels' default behaviour).
    """
    sub = df[cols].dropna()
    if len(sub) < 50 or len(cols) < 2:
        return {c: float("nan") for c in cols}

    scaler = StandardScaler()
    X = scaler.fit_transform(sub.to_numpy(dtype=float))
    out: dict[str, float] = {}
    lr = LinearRegression(n_jobs=1)
    for i, c in enumerate(cols):
        y = X[:, i]
        Xi = np.delete(X, i, axis=1)
        lr.fit(Xi, y)
        r2 = lr.score(Xi, y)
        # Numerical floor: r2 can creep slightly above 1.0 with rank-deficient X.
        r2 = min(r2, 1.0 - 1e-12)
        out[c] = float(1.0 / (1.0 - r2))
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


def _spearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sub = df[cols].dropna()
    rho, _ = spearmanr(sub.to_numpy(dtype=float))
    if np.isscalar(rho):  # only happens if n_cols == 2
        rho = np.array([[1.0, rho], [rho, 1.0]])
    return pd.DataFrame(rho, index=cols, columns=cols)


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


def _save_static_heatmap(corr: pd.DataFrame, path: Path) -> None:
    if corr.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(corr)), max(7, 0.32 * len(corr))))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Pearson r"},
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    ax.set_title("RB ATTN_STATIC_FEATURES correlation (training split)")
    plt.xticks(rotation=70, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _print_top(pairs: list[tuple[str, str, float]], k: int, label: str) -> None:
    print(f"\n  Top {min(k, len(pairs))} pairs by |Pearson| ({label}):")
    print(f"    {'a':<46s} {'b':<46s} {'r':>7s}")
    for a, b, r in pairs[:k]:
        print(f"    {a[:46]:<46s} {b[:46]:<46s} {r:>7.3f}")


def _print_vif(vif: dict[str, float], k: int = 15) -> None:
    items = sorted(vif.items(), key=lambda kv: kv[1], reverse=True)[:k]
    print(f"\n  Top {len(items)} VIF (≥ 5 is a smell, ≥ 10 is bad):")
    print(f"    {'feature':<48s} {'VIF':>10s}")
    for name, v in items:
        print(f"    {name[:48]:<48s} {v:>10.2f}")


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

    full_present = _present_numeric(rb_train, feature_cols)
    static_present = _present_numeric(rb_train, static_cols)
    history_present = _present_numeric(rb_train, history_cols)
    targets_present = _present_numeric(rb_train, list(TARGETS))
    specific_present = _present_numeric(rb_train, list(SPECIFIC_FEATURES))

    print(
        f"  features used in audit: full={len(full_present)} "
        f"static={len(static_present)} history={len(history_present)} "
        f"specific={len(specific_present)} targets={len(targets_present)}"
    )

    # ── Correlation matrices ───────────────────────────────────────────────
    print("\nComputing Pearson correlations …")
    full_pearson = rb_train[full_present].corr(method="pearson")
    static_pearson = rb_train[static_present].corr(method="pearson")

    print("Computing Spearman correlations (full set, may take a few seconds) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        full_spearman = _spearman_matrix(rb_train, full_present)

    high_full = _high_corr_pairs(full_pearson, threshold=args.corr_threshold)
    high_static = _high_corr_pairs(static_pearson, threshold=args.corr_threshold)

    # ── VIF + condition number on the static block ─────────────────────────
    print("\nComputing VIF on ATTN_STATIC_FEATURES …")
    vif = _vif(rb_train, static_present)

    cond_pre, cond_post = _condition_number(rb_train, static_present)
    print(
        f"  condition number  pre-PCA: {cond_pre:.3g}   post-PCA({RIDGE_PCA_COMPONENTS}): {cond_post:.3g}"
    )

    # ── Pre-registered pairs ───────────────────────────────────────────────
    print("\nPre-registered by-construction pairs:")
    pre_reg = _pre_registered_table(rb_train, PRE_REGISTERED_PAIRS)
    print(f"    {'a':<42s} {'b':<42s} {'pearson':>9s} {'spearman':>10s}")
    for row in pre_reg:
        p = "—" if row.get("pearson") is None else f"{row['pearson']:9.3f}"
        s = "—" if row.get("spearman") is None else f"{row['spearman']:10.3f}"
        print(f"    {row['a'][:42]:<42s} {row['b'][:42]:<42s} {p:>9s} {s:>10s}")

    # ── Targets vs features (high signal sanity check) ─────────────────────
    print("\nTop 10 features by |Pearson| with rushing_yards (sanity check):")
    if "rushing_yards" in rb_train.columns:
        rys = (
            rb_train[full_present + ["rushing_yards"]].corr()["rushing_yards"].drop("rushing_yards")
        )
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

    _save_static_heatmap(static_pearson, OUT_HEATMAP)
    print(f"✔ wrote {OUT_HEATMAP.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
