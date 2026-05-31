"""Parameterized feature-collinearity audit for any position.

K and RB already have bespoke, hypothesis-driven audits
(``analysis_k_feature_audit.py`` / ``analysis_rb_feature_audit.py`` — left
untouched). This is the *discovery* tool for the positions that lack one
(QB, WR, TE, DST), but it works for all six so a single run produces a
consistent cross-position comparison.

For each requested position it materialises the training feature matrix the
way the production pipeline does, then computes:

  1. Pairwise Pearson + Spearman correlations across the **full** modelled
     feature set (what Ridge/LightGBM see) and the **static** subset (what
     the attention NN's static branch consumes, ``cfg.attn_static_features``).
     Reports the top-N worst pairs by ``|Pearson|`` rather than a buried
     matrix — this is discovery, not hypothesis confirmation.
  2. Variance Inflation Factor (VIF) on the static block, ``1/(1 - R^2_i)``
     via sklearn ``LinearRegression`` (avoids a ``statsmodels`` dep). The
     static block is the meaningful place for VIF — it's deliberately
     de-correlated, so a high VIF there is a real smell; the full set is
     saturated with by-construction rolling-window collinearity that the
     pair ranking conveys more legibly.
  3. Condition number of the standardised **full** design matrix, reported on
     BOTH populations and against PCA:
       - imputed (production): NaN/±inf -> 0 like ``feature_build.py:110``, the
         full population the models train on (rookies + early-game rows kept);
       - complete-case: listwise ``dropna()``, the textbook view — but only the
         veteran-heavy subset with complete history (e.g. QB ~59% of rows);
       - prod-PCA: post the PCA Ridge actually runs (``ridge_pca_components``:
         WR=30, DST=20, RB=80; QB/TE/K=None), "—" when the position runs none;
       - hypothetical PCA: components retaining 99% variance, computed for
         EVERY position so you can see what PCA would buy QB/TE/K (which run none).
     Thresholds: ``<1e2`` well-conditioned, ``<1e4`` moderate, ``<1e6`` suspect,
     ``>=1e6`` multicollinear (a skill-position pre-PCA reference is ~1.8e8).
  4. Drop candidates (``|r|>0.95`` AND a side ``VIF>10``) on the static
     block — reported as *candidates to investigate*, never auto-dropped.
  5. A leak-free signal sanity check: top-10 features by ``|Pearson|`` with a
     raw target stat.

Two data-flow families, dispatched by position:
  * **split** (QB/RB/WR/TE): read ``data/splits/train.parquet`` →
    ``filter_to_position`` → ``merge_schedule_features`` →
    ``add_specific_features``; full set = ``get_feature_columns()``.
  * **dedicated** (DST/K): the position self-loads team-/kicker-level data
    (DST ``build_data``; K ``load_data``) → ``compute_targets`` →
    ``compute_features`` → training seasons; full set = ``cfg.all_features``.

Outputs (``analysis_output/`` is gitignored):
  - ``analysis_output/{pos}_feature_audit.json``        (per position)
  - ``analysis_output/{pos}_feature_audit_static_corr.png``
  - ``analysis_output/feature_audit_summary.json``      (cross-position table)

Usage::

    python -m src.analysis.analysis_feature_audit QB WR TE DST
    python -m src.analysis.analysis_feature_audit QB --splits-dir data/splits
    python -m src.analysis.analysis_feature_audit QB RB WR TE K DST  # all six

# Numbers from any individual run rot as the feature set and dataset evolve —
# re-run to refresh; don't trust specific cond / VIF / |r| values quoted in
# source comments or a stale findings doc.
"""

from __future__ import annotations

import argparse
import importlib
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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "analysis_output"

# How each position sources its training matrix. "split" positions live in the
# shared player-level data/splits parquet; "dedicated" positions self-load.
SPLIT_POSITIONS = {"QB", "RB", "WR", "TE"}
DEDICATED_POSITIONS = {"DST", "K"}
ALL_POSITIONS = sorted(SPLIT_POSITIONS | DEDICATED_POSITIONS)

# Leak-free raw stat per position for the feature-signal sanity check and the
# _decide_drop tie-break. Deliberately a raw stat, NOT fantasy_points (which is
# a linear combination of the targets and would auto-correlate).
SIGNAL_TARGETS = {
    "QB": "passing_yards",
    "RB": "rushing_yards",
    "WR": "receiving_yards",
    "TE": "receiving_yards",
    "DST": "def_sacks",
    "K": "fg_yards_made",
}

# Drop-candidate thresholds (conservative — report, don't auto-drop).
DROP_R = 0.95
DROP_VIF = 10.0

# Variance retained by the "what would PCA buy this position" probe. 0.99 mirrors
# the spirit of the configured Ridge PCA components (e.g. RB's 80 ≈ 99%+ variance)
# and is applied uniformly so QB/TE/K (no production PCA) get a comparable number.
PCA_VARIANCE_TARGET = 0.99


# ───────────────────────── shared numeric helpers ──────────────────────────
# (Ported verbatim from analysis_rb_feature_audit.py / analysis_k_feature_audit.py;
#  those scripts keep their own copies — this is the canonical tested version.)


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


def _clean_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with NaN/±inf in the present ``cols`` filled to 0.0.

    Mirrors the production catch-all at ``src/shared/feature_build.py:110``
    (``build_position_features``: ``replace([inf,-inf], nan).fillna(0)``), the
    last NaN-handling step every model's feature matrix passes through. Using
    this instead of listwise ``dropna()`` keeps the audit on the SAME full,
    imputed population the models actually train on — not the veteran-heavy
    complete-case subset (rookies have NaN ``prior_season_*``; a player's
    early games have NaN rolling stats). dropna() silently restricted the QB
    audit to ~59% of rows; production sees 100%. See PR #594.
    """
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    if present:
        out[present] = out[present].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _pca_conditioning(
    df: pd.DataFrame, cols: list[str], variance_threshold: float
) -> tuple[int | None, float]:
    """(#components, condition number) retaining ``variance_threshold`` of variance.

    A threshold-free "what would PCA buy this position" probe — answers it
    uniformly for the positions whose Ridge does NOT run PCA (QB/TE/K) as well
    as those that do. Returns the smallest component count whose cumulative
    explained-variance ratio reaches ``variance_threshold`` and the condition
    number of that PCA-reduced matrix. ``(None, nan)`` if too few rows/cols.
    Caller passes an already-imputed frame to measure the production population.
    """
    sub = df[cols].dropna()
    if len(sub) < 50 or len(cols) < 2:
        return None, float("nan")
    X = StandardScaler().fit_transform(sub.to_numpy(dtype=float))
    cumvar = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    n = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n = max(1, min(n, X.shape[1], X.shape[0]))
    cond = float(np.linalg.cond(PCA(n_components=n).fit_transform(X)))
    return n, cond


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


def _max_abs_offdiag(corr: pd.DataFrame) -> float:
    """Largest |value| off the diagonal of a correlation matrix (nan if < 2 cols).

    Unlike ``_high_corr_pairs``, this is threshold-free — it reports the true
    maximum pairwise |Pearson| even when no pair clears the reporting threshold
    (e.g. K, whose worst pair sits at ~0.78, would otherwise summarise as nan).
    """
    arr = corr.to_numpy()
    if arr.shape[0] < 2:
        return float("nan")
    rows, cols = np.triu_indices_from(arr, k=1)
    vals = np.abs(arr[rows, cols])
    vals = vals[~np.isnan(vals)]
    return float(vals.max()) if vals.size else float("nan")


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


def _condition_number(
    df: pd.DataFrame, cols: list[str], pca_components: int | None
) -> tuple[float, float | None]:
    """(pre-PCA, post-PCA) condition numbers of the standardised ``df[cols]``.

    ``post`` is ``None`` when ``pca_components`` is None — i.e. the position
    feeds Ridge the raw feature matrix (QB/TE/K) rather than a PCA-reduced one.
    """
    sub = df[cols].dropna()
    if len(sub) < 50 or len(cols) < 2:
        return float("nan"), None
    X = StandardScaler().fit_transform(sub.to_numpy(dtype=float))
    cond_pre = float(np.linalg.cond(X))
    if pca_components is None:
        return cond_pre, None
    n_components = min(pca_components, X.shape[1], X.shape[0])
    Xp = PCA(n_components=n_components).fit_transform(X)
    return cond_pre, float(np.linalg.cond(Xp))


def _spearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sub = df[cols].dropna()
    rho, _ = spearmanr(sub.to_numpy(dtype=float))
    if np.isscalar(rho):  # only happens if n_cols == 2
        rho = np.array([[1.0, rho], [rho, 1.0]])
    return pd.DataFrame(rho, index=cols, columns=cols)


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


def _drop_candidates(
    corr: pd.DataFrame,
    vif: dict[str, float],
    target_signal: dict[str, float],
    target_col: str | None,
) -> list[dict]:
    """Discovery drop candidates: |r|>DROP_R AND a side VIF>DROP_VIF.

    No pre-registered-pair gate (unlike the K/RB scripts) — we have no
    hypotheses for the new positions, so any pair clearing both thresholds is
    surfaced for a human to investigate.
    """
    out = []
    for a, b, r in _high_corr_pairs(corr, threshold=DROP_R):
        va = vif.get(a, float("nan"))
        vb = vif.get(b, float("nan"))
        if (np.isnan(va) or va <= DROP_VIF) and (np.isnan(vb) or vb <= DROP_VIF):
            continue
        drop, keep, reason = _decide_drop(a, b, target_signal, target_col)
        out.append(
            {
                "drop": drop,
                "keep": keep,
                "pearson": r,
                "vif_a": va,
                "vif_b": vb,
                "reason": reason,
            }
        )
    return out


def _save_static_heatmap(corr: pd.DataFrame, path: Path, pos: str) -> None:
    """Save a Pearson-r heatmap of the static feature block.

    Uses ``matplotlib.imshow`` directly rather than ``seaborn.heatmap`` so the
    audit doesn't pull in ``seaborn`` (not pinned in requirements.txt).
    """
    if corr.empty:
        return
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * n), max(7, 0.32 * n)))
    im = ax.imshow(
        corr.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal", interpolation="nearest"
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(f"{pos} attn_static_features correlation (training split)")
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
    print(f"\n  Top {len(items)} VIF on static block (>= 5 is a smell, >= 10 is bad):")
    print(f"    {'feature':<48s} {'VIF':>10s}")
    for name, v in items:
        print(f"    {name[:48]:<48s} {v:>10.2f}")


# ───────────────────────── data resolution (per family) ─────────────────────


def _resolve_position(pos: str, splits_dir: Path) -> dict:
    """Materialise the training feature frame + column sets for one position.

    Returns a dict with ``train_df``, ``full_cols``, ``static_cols``,
    ``pca_components``, ``signal_target``. Raises on a data/feature-build
    failure (e.g. a stale split missing a raw column needed by feature
    engineering) — the caller skips-and-continues.
    """
    pl = pos.lower()
    cfg = importlib.import_module(f"src.{pl}.config").POSITION_CONFIG
    static_cols = list(cfg.attn_static_features)
    pca_components = cfg.ridge_pca_components
    signal_target = SIGNAL_TARGETS.get(pos)

    if pos in SPLIT_POSITIONS:
        data_mod = importlib.import_module(f"src.{pl}.data")
        feat_mod = importlib.import_module(f"src.{pl}.features")
        from src.shared.weather_features import merge_schedule_features

        train_path = splits_dir / "train.parquet"
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} not found — run SETUP.md's first-time data pull "
                f"or pass --splits-dir at a checkout that has it."
            )
        df = pd.read_parquet(train_path)
        df = data_mod.filter_to_position(df).copy()
        # Mirror build_position_features: weather/Vegas merge happens BEFORE
        # add_specific_features in production, or the weather_vegas static
        # columns would all be missing.
        merge_schedule_features(df, label="train")
        empty = df.iloc[:0].copy()
        df, _, _ = feat_mod.add_specific_features(df, empty.copy(), empty.copy())
        full_cols = feat_mod.get_feature_columns()
    elif pos == "DST":
        data_mod = importlib.import_module("src.dst.data")
        feat_mod = importlib.import_module("src.dst.features")
        tgt_mod = importlib.import_module("src.dst.targets")
        from src.config import TRAIN_SEASONS

        df = data_mod.build_data()
        df = tgt_mod.compute_targets(df)
        feat_mod.compute_features(df)  # in-place, mirrors run_pipeline
        df = df[df["season"].isin(TRAIN_SEASONS)].copy()
        full_cols = list(cfg.all_features)
    elif pos == "K":
        data_mod = importlib.import_module("src.k.data")
        feat_mod = importlib.import_module("src.k.features")
        tgt_mod = importlib.import_module("src.k.targets")

        df = data_mod.load_data()
        df = tgt_mod.compute_targets(df)
        feat_mod.compute_features(df)  # in-place, mirrors run_pipeline
        df, _, _ = data_mod.season_split(df)
        full_cols = list(cfg.all_features)
    else:  # unreachable given argparse choices, but explicit
        raise ValueError(f"unknown position {pos!r}")

    return {
        "train_df": df,
        "full_cols": full_cols,
        "static_cols": static_cols,
        "pca_components": pca_components,
        "signal_target": signal_target,
    }


# ───────────────────────────── per-position audit ───────────────────────────


def _audit_position(pos: str, splits_dir: Path, corr_threshold: float, top_n: int) -> dict:
    """Run the full audit for one position. Returns the cross-position summary row."""
    print(f"\n{'=' * 72}\n=== feature audit: {pos} ===\n{'=' * 72}")
    resolved = _resolve_position(pos, splits_dir)
    df = resolved["train_df"]
    pca_components = resolved["pca_components"]
    signal_target = resolved["signal_target"]
    print(f"  training rows: {len(df):,}")

    full_cols = resolved["full_cols"]
    static_cols = resolved["static_cols"]

    missing = [c for c in full_cols if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:5]) + (
            f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        )
        print(f"  WARN: {len(missing)} feature(s) missing from frame (stale split?): {preview}")

    # Production imputes NaN/±inf -> 0 (feature_build.py:110) and keeps every
    # row. Build that imputed frame and report BOTH populations:
    #   - complete-case (listwise dropna) — the textbook VIF/redundancy view;
    #   - production (imputed) — what Ridge/NN actually factorize, incl. rookies
    #     and early-game rows whose prior_season_*/rolling_* were NaN -> 0.
    feature_universe = list(dict.fromkeys([*full_cols, *static_cols]))
    present_universe = [c for c in feature_universe if c in df.columns]
    n_rows = len(df)
    if present_universe:
        finite = df[present_universe].replace([np.inf, -np.inf], np.nan)
        n_complete = int((~finite.isna().any(axis=1)).sum())
    else:
        n_complete = n_rows
    df_imp = _clean_features(df, feature_universe)
    pct_complete = 100 * n_complete / n_rows if n_rows else 0.0
    print(
        f"  populations: production(imputed)={n_rows:,} rows (100%)  |  "
        f"complete-case={n_complete:,} ({pct_complete:.0f}%)"
    )

    # Present-numeric on the imputed frame: a column constant among veterans but
    # 0 for rookies *does* vary post-fill, exactly as the model sees it.
    full_present = _present_numeric(df_imp, full_cols)
    static_present = _present_numeric(df_imp, static_cols)
    print(f"  features audited: full={len(full_present)} static={len(static_present)}")

    # ── Correlations (reported on the production/imputed population) ────────
    print("  computing Pearson correlations …")
    full_pearson = df_imp[full_present].corr(method="pearson")
    static_pearson = df_imp[static_present].corr(method="pearson")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        full_spearman = _spearman_matrix(df_imp, full_present)

    high_full = _high_corr_pairs(full_pearson, threshold=corr_threshold)
    high_static = _high_corr_pairs(static_pearson, threshold=corr_threshold)
    full_ge_drop = [p for p in high_full if abs(p[2]) >= DROP_R]
    # True max |Pearson| from the full matrix — not high_full[0], which is nan
    # when no pair clears corr_threshold (e.g. K's worst pair ~0.78 < 0.85).
    max_abs_r = _max_abs_offdiag(full_pearson)

    # ── VIF + condition number ────────────────────────────────────────────
    print("  computing VIF on static block …")
    vif = _vif(df_imp, static_present)
    max_vif = max((v for v in vif.values() if not np.isnan(v)), default=float("nan"))

    # Condition number on BOTH populations, so the imputation effect is visible.
    cond_pre, cond_post = _condition_number(df_imp, full_present, pca_components)
    cond_complete, _ = _condition_number(df.replace([np.inf, -np.inf], np.nan), full_present, None)
    cond_label = _classify_condition_number(cond_pre)
    post_str = "—" if cond_post is None else f"{cond_post:.3g}"
    print(
        f"  cond# full (imputed)   pre-PCA: {cond_pre:.3g} ({cond_label})   "
        f"prod-PCA({pca_components}): {post_str}"
    )
    print(f"  cond# full (complete-case)     : {cond_complete:.3g}")
    # Hypothetical PCA for EVERY position (incl. QB/TE/K which run none in prod):
    # what conditioning would retaining PCA_VARIANCE_TARGET variance achieve?
    hyp_n, hyp_cond = _pca_conditioning(df_imp, full_present, PCA_VARIANCE_TARGET)
    print(
        f"  cond# full hypothetical PCA    : {hyp_cond:.3g} "
        f"({hyp_n} comps @ {PCA_VARIANCE_TARGET:.0%} var)"
    )
    cond_static_pre, _ = _condition_number(df_imp, static_present, None)
    print(f"  cond# static (imputed)         : {cond_static_pre:.3g}")

    # ── Signal sanity check + drop-candidate tie-break input ──────────────
    target_signal: dict[str, float] = {}
    if signal_target and signal_target in df_imp.columns:
        rys = df_imp[full_present].corrwith(df_imp[signal_target])
        target_signal = {f: float(v) for f, v in rys.items() if not np.isnan(v)}
        print(f"\n  Top 10 features by |Pearson| with {signal_target} (signal sanity check):")
        for f, _v in rys.abs().sort_values(ascending=False).head(10).items():
            print(f"    {f[:48]:<48s} {rys[f]:>7.3f}")
    else:
        print(f"\n  signal target {signal_target!r} absent — skipping signal sanity check.")

    # ── Stdout summaries ──────────────────────────────────────────────────
    _print_top(high_full, k=top_n, label=f"full set, |r| >= {corr_threshold}")
    _print_top(high_static, k=15, label=f"static subset, |r| >= {corr_threshold}")
    _print_vif(vif)

    drop_candidates = _drop_candidates(static_pearson, vif, target_signal, signal_target)
    print(f"\n  static drop candidates (|r|>{DROP_R} AND VIF>{DROP_VIF}): {len(drop_candidates)}")
    for c in drop_candidates:
        print(
            f"    drop {c['drop']:<30s} keep {c['keep']:<30s} "
            f"r={c['pearson']:+.3f}  VIF=({c['vif_a']:.1f}, {c['vif_b']:.1f})  ({c['reason']})"
        )

    # ── Persist per-position JSON + heatmap ───────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "position": pos,
        "nan_handling": "impute_zero_like_production (feature_build.py:110)",
        "n_rows_train": int(n_rows),
        "n_rows_complete_case": n_complete,
        "pct_rows_complete_case": round(pct_complete, 1),
        "n_features_full": len(full_present),
        "n_features_static": len(static_present),
        "missing_from_frame": missing,
        "corr_threshold": corr_threshold,
        "ridge_pca_components": pca_components,
        "condition_number_full": {
            "imputed_pre_pca": cond_pre,
            "imputed_prod_pca": cond_post,
            "complete_case_pre_pca": cond_complete,
            "hypothetical_pca_components": hyp_n,
            "hypothetical_pca_cond": hyp_cond,
            "hypothetical_pca_variance": PCA_VARIANCE_TARGET,
            "label": cond_label,
        },
        "condition_number_static": {"imputed_pre_pca": cond_static_pre},
        "max_abs_pearson_full": max_abs_r,
        "n_pairs_ge_0.95_full": len(full_ge_drop),
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
        "max_vif_static": max_vif,
        "signal_target": signal_target,
        "feature_signal_pearson": target_signal,
        "drop_candidates_static": drop_candidates,
    }
    out_json = OUT_DIR / f"{pos.lower()}_feature_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"\n  wrote {out_json.relative_to(PROJECT_ROOT)}")

    out_png = OUT_DIR / f"{pos.lower()}_feature_audit_static_corr.png"
    _save_static_heatmap(static_pearson, out_png, pos)
    print(f"  wrote {out_png.relative_to(PROJECT_ROOT)}")

    return {
        "position": pos,
        "n_features_full": len(full_present),
        "pct_complete_case": round(pct_complete, 0),
        "cond_pre_pca": cond_pre,
        "cond_complete_case": cond_complete,
        "cond_post_pca": cond_post,
        "cond_hypothetical_pca": hyp_cond,
        "cond_label": cond_label,
        "max_vif_static": max_vif,
        "max_abs_pearson_full": max_abs_r,
        "n_pairs_ge_0.95_full": len(full_ge_drop),
        "n_drop_candidates_static": len(drop_candidates),
        "status": "ok",
    }


def _print_summary(rows: list[dict]) -> None:
    print(f"\n\n{'=' * 110}\n=== cross-position collinearity summary ===\n{'=' * 110}")
    print(
        "  cond columns are condition number of the FULL feature matrix. "
        "imp=production population (NaN->0); cc=complete-case (dropna);"
    )
    print(
        "  prodPCA=PCA Ridge actually runs (— if none); hypPCA=hypothetical "
        f"PCA @ {PCA_VARIANCE_TARGET:.0%} var (what PCA would buy a no-PCA position)."
    )
    hdr = (
        f"{'pos':<5s} {'n_feat':>6s} {'%cc':>4s} {'cond_imp':>9s} {'cond_cc':>9s} "
        f"{'prodPCA':>8s} {'hypPCA':>8s} {'label':<16s} {'maxVIF':>8s} {'max|r|':>7s} "
        f"{'#r>=.95':>8s} {'#drop':>6s} {'status':<10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["status"] != "ok":
            print(
                f"{r['position']:<5s} {'—':>6s} {'—':>4s} {'—':>9s} {'—':>9s} {'—':>8s} "
                f"{'—':>8s} {'—':<16s} {'—':>8s} {'—':>7s} {'—':>8s} {'—':>6s} {r['status']:<10s}"
            )
            continue
        post = "—" if r["cond_post_pca"] is None else f"{r['cond_post_pca']:.1e}"
        print(
            f"{r['position']:<5s} {r['n_features_full']:>6d} {r['pct_complete_case']:>3.0f}% "
            f"{r['cond_pre_pca']:>9.1e} {r['cond_complete_case']:>9.1e} {post:>8s} "
            f"{r['cond_hypothetical_pca']:>8.1e} {r['cond_label']:<16s} {r['max_vif_static']:>8.1f} "
            f"{r['max_abs_pearson_full']:>7.3f} {r['n_pairs_ge_0.95_full']:>8d} "
            f"{r['n_drop_candidates_static']:>6d} {r['status']:<10s}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "positions",
        nargs="+",
        help=f"Positions to audit ({' '.join(ALL_POSITIONS)}).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Directory with train.parquet for split positions (default: data/splits).",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="Report all pairs with |Pearson| above this (default: 0.85).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many worst full-set pairs to print (default: 20).",
    )
    args = parser.parse_args()

    requested = [p.upper() for p in args.positions]
    unknown = [p for p in requested if p not in ALL_POSITIONS]
    if unknown:
        parser.error(f"unknown position(s): {unknown}; choose from {ALL_POSITIONS}")

    summary_rows: list[dict] = []
    for pos in requested:
        try:
            summary_rows.append(
                _audit_position(pos, args.splits_dir, args.corr_threshold, args.top_n)
            )
        except Exception as e:  # noqa: BLE001 — skip-and-continue so one stale position doesn't sink the run
            print(f"\n  ERROR auditing {pos}: {type(e).__name__}: {e}")
            print("  → skipping. (Stale splits often miss a raw column needed by feature")
            print("    engineering — refresh via SETUP.md and re-run.)")
            summary_rows.append({"position": pos, "status": f"skipped ({type(e).__name__})"})

    _print_summary(summary_rows)

    OUT_DIR.mkdir(exist_ok=True)
    summary_path = OUT_DIR / "feature_audit_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=False))
    print(f"\nwrote {summary_path.relative_to(PROJECT_ROOT)}")

    ok = [r for r in summary_rows if r["status"] == "ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
