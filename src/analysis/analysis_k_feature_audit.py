"""Multicollinearity audit for the K (Kicker) feature set.

Mirrors ``src/analysis/analysis_rb_feature_audit.py`` but adapted to K's
position-specific data flow (no shared ``data/splits/train.parquet`` —
K's loader hits ``nfl_data_py`` directly with a parquet cache under
``data/raw/``). The K pipeline currently shows negative R² across every
model (Ridge 4.079, NN 4.128, Attn 4.132, LGBM 4.061 — see
``benchmark_history/2026-05-20T06-39-23_4af6a9d.json``); this audit
diagnoses whether multicollinearity is inflating Ridge variance or
whether the issue is simply low signal.

Loads the full K dataset (PBP-reconstructed 2015-2024 + weekly 2025),
runs ``compute_targets`` (sets ``fantasy_points`` so ``compute_features``
can roll on it) then ``compute_features`` on the unsplit frame, and
restricts the audit to the **training** rows (season <= 2023) only —
val/test must not influence drop decisions.

Computes:

  1. Pairwise Pearson + Spearman correlations across the full
     ``ALL_FEATURES`` set (= ``SPECIFIC_FEATURES + CONTEXTUAL_FEATURES``)
     and a focused subset for ``ATTN_STATIC_FEATURES`` (what the
     attention NN's static branch actually consumes).
  2. Variance Inflation Factor (VIF) for every feature, computed as
     ``1 / (1 - R^2_i)`` where ``R^2_i`` comes from regressing feature
     ``i`` against the rest via sklearn ``LinearRegression``.
  3. Condition number of the standardised ``ALL_FEATURES`` design
     matrix. K has no Ridge PCA (no ``RIDGE_PCA_COMPONENTS`` const) —
     one number, with interpretation thresholds: ``> 1e4`` suspect,
     ``> 1e6`` multicollinear; for reference, RB's pre-PCA condition
     number was ``1.8e8``.
  4. Pre-registered "by-construction" collinearity checks (see
     ``PRE_REGISTERED_PAIRS``) — confirms or refutes each named
     hypothesis explicitly so the report doesn't bury the answer.
  5. Sanity check: top-10 features by |Pearson| with ``fg_yards_made``
     (current-row, unshifted, leak-free target column).

Drop policy (conservative — fewer drops > overfit to a noisy audit):

  * Flagged (reported, NOT dropped on this alone):
      ``|Pearson| > 0.9`` OR ``VIF > 10``.
  * Drop candidate (the actual drop set):
      ``|Pearson| > 0.95`` AND ``VIF > 10`` AND the pair appears in
      ``PRE_REGISTERED_PAIRS`` (so redundancy is by-construction, not
      coincidental small-sample).

Within a drop candidate pair we keep the more informative feature:
prefer higher |corr-with-target|, tie-break by shorter rolling window
(L3 over L5), then by simpler/less-derived feature.

Outputs:

  - ``analysis_output/k_feature_audit.json``  (machine-readable findings)
  - ``analysis_output/k_feature_audit_static_corr.png``  (heatmap of
    ``ATTN_STATIC_FEATURES`` Pearson matrix)

Usage::

    python -m src.analysis.analysis_k_feature_audit [--corr-threshold 0.85]

# Last run: 2026-05-20 (4af6a9d). Numbers from any individual run rot as the
# feature set and dataset evolve — re-run the script to refresh, don't trust
# any specific cond / VIF / |Pearson| value quoted in source comments. The
# 2026-05-20 run found no drops warranted (cond well below the 1e4 suspect
# threshold; max VIF and max |Pearson| both below the flag thresholds);
# headline diagnosis was that K's MAE floor reflects low signal, not
# variance inflation. Confirm by re-running rather than reading.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; this script writes PNGs, no GUI needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.k.config import POSITION_CONFIG  # noqa: E402
from src.k.data import load_data, season_split  # noqa: E402
from src.k.features import compute_features  # noqa: E402
from src.k.targets import compute_targets  # noqa: E402

ALL_FEATURES = POSITION_CONFIG.all_features
ATTN_STATIC_FEATURES = POSITION_CONFIG.attn_static_features
CONTEXTUAL_FEATURES = POSITION_CONFIG.contextual_features
SPECIFIC_FEATURES = POSITION_CONFIG.specific_features
TARGETS = POSITION_CONFIG.targets

OUT_DIR = PROJECT_ROOT / "analysis_output"
OUT_JSON = OUT_DIR / "k_feature_audit.json"
OUT_HEATMAP = OUT_DIR / "k_feature_audit_static_corr.png"

# Hypotheses to test explicitly so the audit confirms/refutes each in a
# named table rather than burying the result inside the correlation matrix.
# Each entry is (feature_a, feature_b, rationale).
#
# Coverage (one or both rationales per drop-candidate criterion):
#   - Weather/venue redundancy: is_dome ↔ game_wind, is_dome ↔ game_temp
#   - Vegas redundancy:         implied_team_total ↔ total_line
#   - Distance/probability:     avg_fg_distance_L3 ↔ avg_fg_prob_L3
#                               avg_fg_distance_L3 ↔ long_fg_rate_L3
#   - Accuracy variants:        fg_accuracy_L5 ↔ fg_pct_40plus_L5
#                               fg_accuracy_L5 ↔ q4_fg_rate_L5
#                               fg_accuracy_L5 ↔ xp_accuracy_L5
#   - Volume/points:            fg_attempts_L3 ↔ pat_volume_L3
#                               fg_attempts_L3 ↔ total_k_pts_L3
#                               pat_volume_L3 ↔ total_k_pts_L3
#   - Long-FG metrics:          long_fg_rate_L3 ↔ fg_pct_40plus_L5
#   - Trend/variance:           k_pts_trend ↔ k_pts_std_L3
PRE_REGISTERED_PAIRS = [
    ("is_dome", "game_wind", "dome → wind ≈ 0 (strong negative)"),
    ("is_dome", "game_temp", "dome → controlled ~68°F (partial)"),
    ("implied_team_total", "total_line", "line ≈ team_total + opp_total (related, not redundant)"),
    ("avg_fg_distance_L3", "avg_fg_prob_L3", "longer → lower make prob (inverse)"),
    ("avg_fg_distance_L3", "long_fg_rate_L3", "both distance/leg metrics"),
    ("fg_accuracy_L5", "fg_pct_40plus_L5", "overall vs distance-specific accuracy"),
    ("fg_accuracy_L5", "q4_fg_rate_L5", "Q4 is subset of total"),
    ("fg_accuracy_L5", "xp_accuracy_L5", "common factor: kicker leg quality"),
    ("fg_attempts_L3", "pat_volume_L3", "high-scoring offenses produce both"),
    ("fg_attempts_L3", "total_k_pts_L3", "volume drives points"),
    ("pat_volume_L3", "total_k_pts_L3", "PATs are large fraction of total"),
    ("long_fg_rate_L3", "fg_pct_40plus_L5", "both long-FG metrics, different windows"),
    ("k_pts_trend", "k_pts_std_L3", "trend vs variance of recent points"),
]

# Target column used for the "feature signal" sanity check.
# fg_yards_made is the current-row sum of made-FG kick_distance; it is
# NOT a rolling lag, so it doesn't auto-correlate with total_k_pts_L3
# or k_pts_trend the way fg_yard_points (= fg_yards_made * 0.1) does
# downstream — using it here gives a clean leak-free signal probe.
SANITY_TARGET_PRIMARY = "fg_yards_made"
SANITY_TARGET_FALLBACK = "fg_made"


def _present_numeric(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Filter to columns that are present, numeric, and have non-zero variance."""
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


def _condition_number(df: pd.DataFrame, cols: list[str]) -> float:
    """Return the condition number of the standardised ``df[cols]`` design matrix."""
    sub = df[cols].dropna()
    if len(sub) < 50 or len(cols) < 2:
        return float("nan")
    X = StandardScaler().fit_transform(sub.to_numpy(dtype=float))
    return float(np.linalg.cond(X))


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
    """Save a Pearson-r heatmap of the static feature block.

    Uses ``matplotlib.imshow`` directly rather than ``seaborn.heatmap`` so the
    audit script doesn't pull in ``seaborn`` as a dependency.
    """
    if corr.empty:
        return
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * n), max(7, 0.32 * n)))
    im = ax.imshow(
        corr.to_numpy(),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("K ATTN_STATIC_FEATURES correlation (training split)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Pearson r")
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


def _classify_condition_number(cond: float) -> str:
    if np.isnan(cond):
        return "n/a"
    if cond < 1e2:
        return "well-conditioned"
    if cond < 1e4:
        return "moderate"
    if cond < 1e6:
        return "suspect"
    return "multicollinear"


def _decide_drop(
    a: str, b: str, target_signal: dict[str, float], target_col: str | None
) -> tuple[str, str, str]:
    """Pick which side of a redundant pair to drop. Returns (drop, keep, reason).

    Tie-break ladder (each step exits early once a decision is reached):
      1. Higher |corr-with-target| wins (keep).
      2. Prefer L3 over L5 rolling window.
      3. Final fallback: drop the longer name (proxy for "more derived").
    """
    sa = abs(target_signal.get(a, float("nan")))
    sb = abs(target_signal.get(b, float("nan")))
    if not np.isnan(sa) and not np.isnan(sb) and abs(sa - sb) > 1e-6:
        drop = a if sa < sb else b
        return drop, (b if drop == a else a), f"lower |corr-with-{target_col}|"

    a_l5 = "_L5" in a
    b_l5 = "_L5" in b
    if a_l5 != b_l5:
        drop = a if a_l5 else b
        return drop, (b if drop == a else a), "L5 rolling window (prefer fresher L3)"

    drop = a if len(a) >= len(b) else b
    return drop, (b if drop == a else a), "more derived / longer name"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="Report all pairs with |Pearson| above this (default: 0.85)",
    )
    args = parser.parse_args()

    # ── Load + materialise features ────────────────────────────────────────
    # The K loader handles caching directly via nfl_data_py — no shared
    # data/splits/train.parquet equivalent. Fail loudly with a SETUP.md
    # pointer if the parquet cache is missing.
    try:
        print("Loading kicker data (PBP 2015-2024 + weekly 2025) …")
        k_df = load_data()
    except (FileNotFoundError, OSError) as e:
        print(f"ERROR: K data load failed ({e}).")
        print("  Hint: run SETUP.md's first-time data pull so data/raw/ is populated,")
        print("  or run `python -m src.k.run_pipeline` once to seed kicker_pbp_*.parquet.")
        return 1

    if len(k_df) == 0:
        print("ERROR: load_data returned an empty frame (nflverse outage?).")
        return 1
    print(f"  loaded rows: {len(k_df):,}")

    # IMPORTANT: ordering — compute_targets first (sets `fantasy_points`),
    # then compute_features (reads it for the rolling total-points stats).
    k_df = compute_targets(k_df)
    print("Computing K features on full dataset (in-place) …")
    compute_features(k_df)

    train_df, _val_df, _test_df = season_split(k_df)
    print(f"  training rows used for audit: {len(train_df):,}")

    feature_cols = list(ALL_FEATURES)
    static_cols = list(ATTN_STATIC_FEATURES)

    # Diagnostic: which feature columns are missing from the materialised frame.
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        preview = ", ".join(missing[:5]) + (
            f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        )
        print(f"  WARN: {len(missing)} feature(s) missing from frame: {preview}")

    full_present = _present_numeric(train_df, feature_cols)
    static_present = _present_numeric(train_df, static_cols)
    targets_present = _present_numeric(train_df, list(TARGETS))
    specific_present = _present_numeric(train_df, list(SPECIFIC_FEATURES))
    contextual_present = _present_numeric(train_df, list(CONTEXTUAL_FEATURES))

    print(
        f"  features used in audit: full={len(full_present)} "
        f"static={len(static_present)} specific={len(specific_present)} "
        f"contextual={len(contextual_present)} targets={len(targets_present)}"
    )

    # ── Correlation matrices ───────────────────────────────────────────────
    print("\nComputing Pearson correlations …")
    full_pearson = train_df[full_present].corr(method="pearson")
    static_pearson = train_df[static_present].corr(method="pearson")

    print("Computing Spearman correlations (full set) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        full_spearman = _spearman_matrix(train_df, full_present)

    high_full = _high_corr_pairs(full_pearson, threshold=args.corr_threshold)
    high_static = _high_corr_pairs(static_pearson, threshold=args.corr_threshold)

    # ── VIF + condition number on the full feature set ─────────────────────
    print("\nComputing VIF on ALL_FEATURES …")
    vif = _vif(train_df, full_present)

    cond = _condition_number(train_df, full_present)
    cond_label = _classify_condition_number(cond)
    print(f"  condition number (standardised ALL_FEATURES): {cond:.3g} ({cond_label})")
    print(
        "  thresholds: <1e2 well-conditioned, <1e4 moderate, <1e6 suspect, "
        "≥1e6 multicollinear (RB pre-PCA ref: 1.8e8)."
    )

    # ── Pre-registered pairs ───────────────────────────────────────────────
    print("\nPre-registered by-construction pairs:")
    pre_reg = _pre_registered_table(train_df, PRE_REGISTERED_PAIRS)
    print(f"    {'a':<42s} {'b':<42s} {'pearson':>9s} {'spearman':>10s}")
    for row in pre_reg:
        p = "—" if row.get("pearson") is None else f"{row['pearson']:9.3f}"
        s = "—" if row.get("spearman") is None else f"{row['spearman']:10.3f}"
        print(f"    {row['a'][:42]:<42s} {row['b'][:42]:<42s} {p:>9s} {s:>10s}")

    # ── Sanity: top features by |Pearson| with a leak-free target ──────────
    if SANITY_TARGET_PRIMARY in train_df.columns:
        target_col: str | None = SANITY_TARGET_PRIMARY
        header = f"\nTop 10 features by |Pearson| with {target_col} (sanity check):"
    elif SANITY_TARGET_FALLBACK in train_df.columns:
        target_col = SANITY_TARGET_FALLBACK
        header = (
            f"\nTop 10 features by |Pearson| with {target_col} "
            f"(fallback — {SANITY_TARGET_PRIMARY} missing):"
        )
    else:
        target_col = None
        header = (
            "\nNo clean leak-free target column on frame "
            f"(neither {SANITY_TARGET_PRIMARY} nor {SANITY_TARGET_FALLBACK}); "
            "skipping feature-signal sanity check."
        )
    print(header)

    target_signal: dict[str, float] = {}
    if target_col is not None:
        # corrwith is O(n_features) vs O(n_features²) for a full corr matrix.
        rys = train_df[full_present].corrwith(train_df[target_col])
        target_signal = {f: float(v) for f, v in rys.items()}
        top_ry = rys.abs().sort_values(ascending=False).head(10)
        for f, _v in top_ry.items():
            print(f"    {f[:48]:<48s} {rys[f]:>7.3f}")

    # ── Stdout summaries ───────────────────────────────────────────────────
    _print_top(high_full, k=20, label=f"full set, |r| ≥ {args.corr_threshold}")
    _print_top(high_static, k=15, label=f"ATTN_STATIC subset, |r| ≥ {args.corr_threshold}")
    _print_vif(vif)

    # ── Drop-candidate analysis ────────────────────────────────────────────
    # Conservative: pair must clear |r| > 0.95 AND have at least one side
    # with VIF > 10 AND be one of the pre-registered "by-construction" pairs.
    pre_reg_set = {frozenset({a, b}) for a, b, _ in PRE_REGISTERED_PAIRS}
    drop_candidates = []
    for a, b, r in _high_corr_pairs(full_pearson, threshold=0.95):
        if frozenset({a, b}) not in pre_reg_set:
            continue
        va = vif.get(a, float("nan"))
        vb = vif.get(b, float("nan"))
        if (np.isnan(va) or va <= 10) and (np.isnan(vb) or vb <= 10):
            continue
        drop, keep, reason = _decide_drop(a, b, target_signal, target_col)
        drop_candidates.append(
            {
                "drop": drop,
                "keep": keep,
                "pearson": r,
                "vif_a": va,
                "vif_b": vb,
                "reason": reason,
            }
        )

    print(f"\nDrop candidates (|r|>0.95 AND VIF>10 AND pre-registered): {len(drop_candidates)}")
    for c in drop_candidates:
        print(
            f"    drop {c['drop']:<32s} keep {c['keep']:<32s} "
            f"r={c['pearson']:+.3f}  VIF=({c['vif_a']:.1f}, {c['vif_b']:.1f})  "
            f"({c['reason']})"
        )

    # ── Persist outputs ────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "n_rows_k_train": int(len(train_df)),
        "n_features_full": len(full_present),
        "n_features_static": len(static_present),
        "n_features_specific": len(specific_present),
        "n_features_contextual": len(contextual_present),
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
        "vif_full": vif,
        "condition_number_full": {"value": cond, "label": cond_label},
        "pre_registered_pairs": pre_reg,
        "feature_signal_target": target_col,
        "feature_signal_pearson": target_signal,
        "drop_candidates": drop_candidates,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"\n✔ wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    _save_static_heatmap(static_pearson, OUT_HEATMAP)
    print(f"✔ wrote {OUT_HEATMAP.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
