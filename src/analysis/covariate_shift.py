"""Train -> val/test covariate-shift guard.

Catches the class of bug where a feature is in-distribution in *train* but degenerate /
out-of-distribution in *test*, so the train-fit ``StandardScaler`` maps it far into the
tail and the attention NN extrapolates badly. The canonical instance: commit ``9409e7e``
filled missing depth with ``-1``; the 2025 test season had **zero** depth-chart coverage,
so every test row was the ``-1`` sentinel, which standardized to ≈ −4σ and regressed QB
attention NN 6.513 -> 8.066 MAE (TODO.md archive). A standing diagnostic flags that
**before** a GPU run is wasted.

**Primary metric — post-scaler OOD-z.** Per whitelisted feature, fit mean/std on train
(exactly as the pipeline's ``StandardScaler`` does, zero-variance -> scale 1) and measure
the *pre-clip* standardized test values: ``mean_z``, ``frac_beyond_4σ`` (the ``±4`` matches
``FEATURE_CLIP``), ``max_abs_z``. Flag if ``|mean_z| > 1.0`` or ``frac_beyond_4σ > 0.5``.
The depth_chart bug gives ``mean_z ≈ −4`` (trips instantly); an in-distribution or imputed
feature gives ``mean_z ≈ 0``; a feature that is *constant in both* train and test
auto-exempts (zero variance -> z = 0, the same mechanism that spared K).

**Secondary metric — PSI** (population stability index, train-decile bins, ε-floored) is
reported as descriptive context, not the hard gate (it is noisy on small / discrete K/DST
test sets).

The diagnostic runs on the **post-``build_position_features``** matrix — i.e. after the
pipeline's own imputations (including the depth_chart fix) — so it measures the *residual*
risk the model actually faces. On the current splits ``depth_chart_rank`` therefore shows
``mean_z ≈ 0`` (the fix holds); the guard exists to catch the *next* un-neutralized feature.

Usage::

    python -m src.analysis.covariate_shift [POS ...] [--top 20] [--allow week,...] [--fail-on-shift]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MIN_GAMES_PER_SEASON, SPLITS_DIR  # noqa: E402
from src.shared.position import Position  # noqa: E402

# NOTE: ``build_position_features`` is imported lazily inside
# ``shift_report_for_position`` — it pulls the nflverse data-loader chain, which the pure
# stat cores (``ood_scaler_stats`` / ``population_stability_index`` / ``compute_feature_shift``)
# and their unit tests must not require.

OUT_DIR = PROJECT_ROOT / "analysis_output"

# Thresholds. OOD_CLIP_K mirrors FEATURE_CLIP ((-4, 4) in src/shared/feature_build.py) so the
# "beyond clip" fraction matches what the NN's scaler would actually saturate.
OOD_CLIP_K: float = 4.0
MEAN_Z_THRESHOLD: float = 1.0
FRAC_BEYOND_K_THRESHOLD: float = 0.5
PSI_BINS: int = 10
PSI_EPS: float = 1e-6
PSI_WARN: float = 0.25
LOW_N_THRESHOLD: int = 50  # below this many test rows, PSI is descriptive-only

# Features that are legitimately shifted and must not gate the split refresh. ``week`` is
# included because the test season's weeks are a subset of the train weeks, giving a benign
# nonzero mean_z if ``week`` is ever whitelisted as a feature. Each entry should carry a
# one-line reason, mirroring refresh-splits.yml's runtime-added carve-out.
DEFAULT_ALLOWLIST: dict[str, set[str]] = {p: {"week"} for p in Position.values()}


def ood_scaler_stats(
    train_vals: np.ndarray, other_vals: np.ndarray, *, k: float = OOD_CLIP_K
) -> dict:
    """Standardize ``other_vals`` by train mean/std and summarize the tail.

    Replicates ``sklearn.StandardScaler``: a zero-variance train feature uses ``scale=1``,
    so a feature constant in both train and ``other`` yields ``mean_z=0`` (auto-exempt).
    Returns ``{mean_z, max_abs_z, frac_beyond_k, train_std, constant}``.
    """
    train = np.asarray(train_vals, dtype=float)
    other = np.asarray(other_vals, dtype=float)
    train = train[np.isfinite(train)]
    other = other[np.isfinite(other)]
    if train.size == 0 or other.size == 0:
        return {
            "mean_z": float("nan"),
            "max_abs_z": float("nan"),
            "frac_beyond_k": float("nan"),
            "train_std": float("nan"),
            "constant": False,
        }
    mean = train.mean()
    std = train.std(ddof=0)
    constant = std <= 1e-12
    scale = std if not constant else 1.0
    z = (other - mean) / scale
    return {
        "mean_z": float(z.mean()),
        "max_abs_z": float(np.abs(z).max()),
        "frac_beyond_k": float((np.abs(z) > k).mean()),
        "train_std": float(std),
        "constant": bool(constant),
    }


def population_stability_index(
    train_vals: np.ndarray, test_vals: np.ndarray, *, bins: int = PSI_BINS, eps: float = PSI_EPS
) -> dict:
    """PSI of ``test_vals`` vs ``train_vals`` using train-decile bins.

    Bands: <0.1 none, 0.1-0.25 moderate, >0.25 significant. A constant train feature
    short-circuits to PSI 0 (``constant=True``) — avoids the div-by-zero an empty-variance
    binning would hit.
    """
    train = np.asarray(train_vals, dtype=float)
    test = np.asarray(test_vals, dtype=float)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if train.size == 0 or test.size == 0:
        return {"psi": float("nan"), "band": "undefined", "constant": False}

    edges = np.unique(np.quantile(train, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:  # constant (or near-constant) train feature
        return {"psi": 0.0, "band": "none", "constant": True}
    # Open the end bins so test values outside the train range are still counted.
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf

    train_counts, _ = np.histogram(train, bins=edges)
    test_counts, _ = np.histogram(test, bins=edges)
    train_prop = np.clip(train_counts / train_counts.sum(), eps, None)
    test_prop = np.clip(test_counts / test_counts.sum(), eps, None)
    psi = float(np.sum((test_prop - train_prop) * np.log(test_prop / train_prop)))
    band = "none" if psi < 0.1 else ("moderate" if psi < PSI_WARN else "significant")
    return {"psi": psi, "band": band, "constant": False}


def compute_feature_shift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    val_df: pd.DataFrame | None = None,
    allowlist: set[str] | None = None,
) -> list[dict]:
    """Per-feature train->test (and train->val) shift records, worst-first by ``|mean_z|``.

    A feature is ``flagged`` when it is NOT in ``allowlist`` and either ``|mean_z| > 1.0`` or
    ``frac_beyond_4σ > 0.5`` on the train->test comparison. A missing feature column is
    reported as ``missing=True`` and flagged (the gate should fail loudly).
    """
    allowlist = allowlist or set()
    low_n = len(test_df) < LOW_N_THRESHOLD
    records: list[dict] = []
    for col in feature_cols:
        if col not in train_df.columns or col not in test_df.columns:
            records.append({"feature": col, "missing": True, "flagged": col not in allowlist})
            continue
        ood = ood_scaler_stats(train_df[col].to_numpy(), test_df[col].to_numpy())
        psi = population_stability_index(train_df[col].to_numpy(), test_df[col].to_numpy())
        rec = {
            "feature": col,
            "missing": False,
            "mean_z": ood["mean_z"],
            "max_abs_z": ood["max_abs_z"],
            "frac_beyond_4sigma": ood["frac_beyond_k"],
            "train_std": ood["train_std"],
            "constant": ood["constant"],
            "psi": psi["psi"],
            "psi_band": psi["band"],
            "low_n": low_n,
            "allowlisted": col in allowlist,
        }
        if val_df is not None and col in val_df.columns:
            rec["mean_z_val"] = ood_scaler_stats(train_df[col].to_numpy(), val_df[col].to_numpy())[
                "mean_z"
            ]
        ood_flag = (abs(ood["mean_z"]) > MEAN_Z_THRESHOLD) or (
            ood["frac_beyond_k"] > FRAC_BEYOND_K_THRESHOLD
        )
        rec["flagged"] = bool(ood_flag and col not in allowlist)
        records.append(rec)

    records.sort(
        key=lambda r: abs(r.get("mean_z", 0.0)) if np.isfinite(r.get("mean_z", 0.0)) else 0.0,
        reverse=True,
    )
    return records


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()
    return df


def shift_report_for_position(
    position: str,
    *,
    splits_dir: str | Path = SPLITS_DIR,
    allowlist: set[str] | None = None,
) -> dict:
    """Build the post-``build_position_features`` shift report for one position.

    Replicates the pipeline's train preparation exactly (filter -> min-games filter ->
    compute targets -> ``build_position_features``) so the standardized values match what the
    NN's scaler sees. Requires ``data/splits/{train,val,test}.parquet``.
    """
    from src.shared.feature_build import build_position_features
    from src.shared.registry import get_config

    splits_dir = Path(splits_dir)
    allowlist = allowlist if allowlist is not None else DEFAULT_ALLOWLIST.get(position, set())
    cfg = get_config(position)

    train_df = _load_split(splits_dir / "train.parquet")
    val_df = _load_split(splits_dir / "val.parquet")
    test_df = _load_split(splits_dir / "test.parquet")

    pos_train = cfg["filter_fn"](train_df)
    pos_val = cfg["filter_fn"](val_df)
    pos_test = cfg["filter_fn"](test_df)

    games = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    # Per-position floor (None → global), matching pipeline.py so the shift report
    # reflects production's actual train population after the min-games relaxation.
    min_games = cfg.get("min_games_per_season")
    if min_games is None:
        min_games = MIN_GAMES_PER_SEASON
    pos_train = pos_train[games >= min_games].copy()

    pos_train = cfg["compute_targets_fn"](pos_train)
    pos_val = cfg["compute_targets_fn"](pos_val)
    pos_test = cfg["compute_targets_fn"](pos_test)

    feature_cols = cfg["get_feature_columns_fn"]()
    pos_train, pos_val, pos_test = build_position_features(
        pos_train, pos_val, pos_test, cfg, feature_cols
    )

    features = compute_feature_shift(
        pos_train, pos_test, feature_cols, val_df=pos_val, allowlist=allowlist
    )
    return {
        "position": position,
        "n_train": int(len(pos_train)),
        "n_val": int(len(pos_val)),
        "n_test": int(len(pos_test)),
        "n_features": len(feature_cols),
        "allowlist": sorted(allowlist),
        "thresholds": {
            "mean_z": MEAN_Z_THRESHOLD,
            "frac_beyond_4sigma": FRAC_BEYOND_K_THRESHOLD,
            "psi_warn": PSI_WARN,
        },
        "features": features,
    }


def gate_check(report: dict) -> tuple[bool, list[dict]]:
    """Return ``(ok, offenders)`` — ok is False if any feature is flagged (non-allowlisted)."""
    offenders = [f for f in report["features"] if f.get("flagged")]
    return (len(offenders) == 0, offenders)


def format_shift_table(report: dict, *, top: int = 20) -> str:
    feats = report["features"]
    flagged = [f for f in feats if f.get("flagged")]
    lines = [
        f"Covariate shift {report['position']} — train={report['n_train']} "
        f"val={report['n_val']} test={report['n_test']} features={report['n_features']}",
        f"  flagged: {len(flagged)}  (|mean_z|>{MEAN_Z_THRESHOLD} or "
        f"frac_beyond_4σ>{FRAC_BEYOND_K_THRESHOLD})",
        f"  {'feature':<44s} {'mean_z':>8s} {'frac>4σ':>8s} {'psi':>7s} {'band':>11s}  flag",
    ]
    for f in feats[:top]:
        if f.get("missing"):
            lines.append(
                f"  {f['feature'][:44]:<44s} {'MISSING':>8s}{'':>8s}{'':>7s}{'':>11s}  FLAG"
            )
            continue
        mark = "FLAG" if f.get("flagged") else ("allow" if f.get("allowlisted") else "")
        lines.append(
            f"  {f['feature'][:44]:<44s} {f['mean_z']:>8.3f} {f['frac_beyond_4sigma']:>8.3f} "
            f"{f['psi']:>7.3f} {f['psi_band']:>11s}  {mark}"
        )
    if len(feats) > top:
        lines.append(f"  … {len(feats) - top} more (worst-first; see JSON)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "positions", nargs="*", default=Position.values(), help="Positions (default: all six)"
    )
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--top", type=int, default=20, help="Rows to print per position")
    parser.add_argument(
        "--allow", default="", help="Comma-separated extra features to exempt from the gate"
    )
    parser.add_argument(
        "--fail-on-shift",
        action="store_true",
        help="Exit non-zero if any non-allowlisted feature is flagged",
    )
    args = parser.parse_args(argv)

    extra_allow = {a.strip() for a in args.allow.split(",") if a.strip()}
    OUT_DIR.mkdir(exist_ok=True)
    any_flagged = False
    for pos in args.positions:
        allowlist = DEFAULT_ALLOWLIST.get(pos, set()) | extra_allow
        report = shift_report_for_position(pos, splits_dir=args.splits_dir, allowlist=allowlist)
        print(f"\n{'#' * 60}\n# COVARIATE SHIFT {pos}\n{'#' * 60}")
        print(format_shift_table(report, top=args.top))
        ok, offenders = gate_check(report)
        if not ok:
            any_flagged = True
            for off in offenders:
                print(
                    f"::warning::[{pos}] covariate shift: {off['feature']} "
                    f"(mean_z={off.get('mean_z', float('nan')):.3f})"
                )
        out_path = OUT_DIR / f"covariate_shift_{pos}.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"✔ wrote {out_path.relative_to(PROJECT_ROOT)}")

    if args.fail_on_shift and any_flagged:
        print("\nERROR: covariate-shift gate failed (flagged features above).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
