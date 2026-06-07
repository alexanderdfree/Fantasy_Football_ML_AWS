"""Generic feature-collinearity statistics shared by the per-position feature
audits (``analysis_feature_audit`` / ``analysis_rb_feature_audit`` /
``analysis_k_feature_audit``). Pure numpy/pandas/scipy/sklearn — no data loaders,
no position imports — cheap to import. Extracted to kill the byte-identical copies
that lived in all three audit scripts (reuse sweep, finding A).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


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
