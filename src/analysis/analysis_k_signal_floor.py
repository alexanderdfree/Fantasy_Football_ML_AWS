"""Signal-floor diagnostic for the K (Kicker) model.

The just-shipped multicollinearity audit (``analysis_k_feature_audit.py``)
showed K's feature matrix is well-conditioned (cond=8.33, max VIF=8.49,
max |Pearson|=0.778), and yet every production model lands at MAE ≈ 4.06–4.13
with R² ≈ 0 on test. That left an open question: **is K's MAE floor noise,
or is there exploitable signal that the current feature set / model class
just isn't capturing?**

This script answers that by:

  1. Computing **naive baselines** (constant-mean, kicker rolling mean,
     team kicker rolling mean, Vegas-only Ridge) and measuring how close
     the production Ridge / NN / Attn / LGBM models get to them.
  2. Re-fitting a quick Ridge and LGBM **inside the script** to get
     per-row predictions on val + test (the benchmark JSON only carries
     aggregate MAE).
  3. Slicing test-set Ridge errors by ``is_dome``, ``is_home``,
     ``surface``, wind quartile, week bucket, Vegas-implied-total quartile,
     and the top-10 kickers — flagging any segment with MAE outside
     [3.0, 5.0] as a candidate for Phase-2 modelling.
  4. Emitting a categorical recommendation based on the gap between the
     best naive baseline and the best refit model.

If the best refit model beats the best naive baseline by less than
≈0.05 MAE, K is at the noise floor and additional feature engineering
on the existing data won't move the needle — Phase 2 should be capped or
abandoned in favor of new data sources.

Outputs:

  - ``analysis_output/k_signal_floor.json``  (machine-readable findings)
  - ``analysis_output/k_signal_floor_segments.png``  (segment-MAE bar
    chart for the Ridge model on test)

Usage::

    python -m src.analysis.analysis_k_signal_floor [--quick]

``--quick`` skips the LGBM refit (saves ~10s) for use during script
iteration; LGBM is the production winner so the default run includes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.k.config import POSITION_CONFIG  # noqa: E402

ALL_FEATURES = POSITION_CONFIG.all_features
TARGETS = POSITION_CONFIG.targets
from src.k.data import load_data, season_split  # noqa: E402
from src.k.features import compute_features  # noqa: E402
from src.k.targets import compute_targets  # noqa: E402
from src.shared.aggregate_targets import predictions_to_fantasy_points  # noqa: E402

OUT_DIR = PROJECT_ROOT / "analysis_output"
OUT_JSON = OUT_DIR / "k_signal_floor.json"
OUT_PNG = OUT_DIR / "k_signal_floor_segments.png"

BENCHMARK_PATH = PROJECT_ROOT / "benchmark_history" / "2026-05-20T06-39-23_4af6a9d.json"
# Hard-coded fallback if the benchmark JSON is missing (e.g. CI shard without
# the file). Matches the post-mortem in this script's docstring.
BENCHMARK_FALLBACK = {"ridge": 4.079, "nn": 4.128, "attn": 4.132, "lgbm": 4.061}

# Recommendation gap thresholds (test MAE units = fantasy points).
GAP_PROCEED = 0.15
GAP_CAPPED = 0.05
GAP_BUG = 0.05  # model worse than naive by this margin → investigate

# Per-segment MAE flag thresholds — outside this window means the model is
# systematically off in that slice and worth chasing in Phase 2.
SEGMENT_LOW = 3.0
SEGMENT_HIGH = 5.0


# ── Surface normalisation ─────────────────────────────────────────────────
# Raw `surface` values include many turf variants and stray strings like
# "grass " (trailing space) and "". Bucket them so segment counts aren't
# scattered across 10+ near-duplicate labels.
_TURF_VARIANTS = frozenset(
    {"fieldturf", "matrixturf", "sportturf", "astroturf", "astroplay", "a_turf"}
)


def _normalise_surface(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "other_or_na"
    s = str(v).strip().lower()
    if s == "grass":
        return "grass"
    if s in _TURF_VARIANTS:
        return "turf"
    return "other_or_na"


# ── Baselines ─────────────────────────────────────────────────────────────


def _grouped_rolling_mean(
    full_df: pd.DataFrame,
    group_col: str,
    sort_cols: list[str],
    window: int | None,
    fill: float,
) -> pd.Series:
    """Shift(1) rolling/expanding mean of ``fantasy_points`` grouped by
    ``group_col``. ``window=None`` = expanding (career-long); ``window=N`` =
    last N games. Returns a Series aligned to ``full_df``'s original index.
    """
    df = full_df.sort_values(sort_cols)
    grp = df.groupby(group_col)["fantasy_points"]
    if window is None:
        # expanding() on a groupby returns a MultiIndex (group, original).
        # Drop the group level to realign on the input frame's index.
        rolling = grp.expanding().mean().shift(1).reset_index(level=0, drop=True)
    else:
        rolling = grp.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    return rolling.fillna(fill).reindex(full_df.index)


def _vegas_only_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str] = ("implied_team_total", "is_dome"),
) -> tuple[np.ndarray, np.ndarray] | None:
    """Tiny Ridge fitted on just (implied_team_total, is_dome). Tests whether
    Vegas-line + venue alone capture K's floor.
    """
    cols = list(cols)
    if not all(c in train_df.columns for c in cols):
        return None
    train_sub = train_df[cols + ["fantasy_points"]].dropna()
    if len(train_sub) < 50:
        return None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_sub[cols].to_numpy(dtype=float))
    y_train = train_sub["fantasy_points"].to_numpy(dtype=float)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    def _predict(df: pd.DataFrame) -> np.ndarray:
        # Fill with train medians so we predict for every row (matches the
        # production pipeline's NaN-handling); standardise with the train
        # scaler.
        sub = df[cols].copy()
        for c in cols:
            sub[c] = sub[c].fillna(train_sub[c].median())
        X = scaler.transform(sub.to_numpy(dtype=float))
        return model.predict(X)

    return _predict(val_df), _predict(test_df)


# ── In-script Ridge / LGBM refits ─────────────────────────────────────────


def _prep_design_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple:
    """Return (X_train, X_val, X_test, scaler) standardised on train means.

    NaN-fills with train-column means before standardisation (matches the
    K production pipeline's ``fill_nans_with_train_means``). Returns ndarrays
    aligned to the original train/val/test row order.
    """
    train_means = train_df[feature_cols].mean()

    def _fill(df: pd.DataFrame) -> np.ndarray:
        return df[feature_cols].fillna(train_means).to_numpy(dtype=float)

    X_train = _fill(train_df)
    X_val = _fill(val_df)
    X_test = _fill(test_df)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def _refit_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one Ridge per target, aggregate to fantasy points. Returns
    (val_preds_fp, test_preds_fp).
    """
    X_train, X_val, X_test, _ = _prep_design_matrix(train_df, val_df, test_df, feature_cols)
    val_preds_dict: dict[str, np.ndarray] = {}
    test_preds_dict: dict[str, np.ndarray] = {}
    for target in TARGETS:
        y_train = train_df[target].fillna(0).to_numpy(dtype=float)
        m = Ridge(alpha=1.0)
        m.fit(X_train, y_train)
        # Clamp to non-negative — matches the K NN's per-head clamp and the
        # raw-stat semantics (a negative predicted "miss count" is invalid).
        val_preds_dict[target] = np.clip(m.predict(X_val), 0.0, None)
        test_preds_dict[target] = np.clip(m.predict(X_test), 0.0, None)
    val_fp = predictions_to_fantasy_points("K", val_preds_dict)
    test_fp = predictions_to_fantasy_points("K", test_preds_dict)
    return val_fp, test_fp


def _refit_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit one LightGBM per target with quick defaults. Returns
    (val_preds_fp, test_preds_fp), or None if lightgbm is unavailable.
    """
    try:
        import lightgbm as lgb  # local import — soft dependency
    except ImportError:
        return None

    # LGBM handles NaN natively, but use the same fill as Ridge for parity.
    # Pass DataFrames (not ndarrays) to fit + predict so LGBM/sklearn don't
    # warn about feature-name drift (it auto-generates `Column_N` names from
    # an ndarray fit, then complains on every predict).
    train_means = train_df[feature_cols].mean()
    X_train = train_df[feature_cols].fillna(train_means).astype(float)
    X_val = val_df[feature_cols].fillna(train_means).astype(float)
    X_test = test_df[feature_cols].fillna(train_means).astype(float)

    val_preds_dict: dict[str, np.ndarray] = {}
    test_preds_dict: dict[str, np.ndarray] = {}
    for target in TARGETS:
        y_train = train_df[target].fillna(0).to_numpy(dtype=float)
        m = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=30,
            reg_lambda=2.0,
            reg_alpha=0.1,
            objective="huber",
            n_jobs=1,
            verbose=-1,
        )
        m.fit(X_train, y_train)
        val_preds_dict[target] = np.clip(m.predict(X_val), 0.0, None)
        test_preds_dict[target] = np.clip(m.predict(X_test), 0.0, None)
    val_fp = predictions_to_fantasy_points("K", val_preds_dict)
    test_fp = predictions_to_fantasy_points("K", test_preds_dict)
    return val_fp, test_fp


# ── Per-segment MAE breakdown ─────────────────────────────────────────────


_QUARTILE_LABELS = ["Q1_low", "Q2", "Q3", "Q4_high"]


def _quartile_cut(values: pd.Series, train_values: pd.Series) -> pd.Series:
    """Bin ``values`` into the quartiles of ``train_values``.

    Edges are open-ended on both sides (``-inf`` / ``+inf``) so out-of-range
    test points still land in a bin. Returns an "na"-filled Series if the
    train quartiles collapse (e.g. constant column).
    """
    edges = train_values.quantile([0.25, 0.5, 0.75]).to_list()
    uniq = sorted(set(edges))
    if len(uniq) < 1:
        return pd.Series(["na"] * len(values), index=values.index)
    bounded = [-np.inf] + uniq + [np.inf]
    return pd.cut(
        values, bins=bounded, labels=_QUARTILE_LABELS[: len(bounded) - 1], include_lowest=True
    )


def _week_bucket(week: int) -> str:
    if week <= 6:
        return "weeks_1_6"
    if week <= 12:
        return "weeks_7_12"
    return "weeks_13_18"


def _segment_mae(
    test_df: pd.DataFrame,
    ground_truth: np.ndarray,
    preds: np.ndarray,
    train_df: pd.DataFrame,
    position_mean: float,
) -> dict[str, dict]:
    """Compute MAE in each segment slice. Cuts for quartile-binned features
    are derived from ``train_df`` to avoid leakage.

    Returns a dict shaped like the JSON ``segments`` field.
    """
    abs_err = np.abs(ground_truth - preds)
    pos_mean_err = np.abs(ground_truth - position_mean)

    df = test_df.reset_index(drop=True).copy()
    df["_abs_err"] = abs_err
    df["_pos_mean_err"] = pos_mean_err

    out: dict[str, dict] = {}

    def _emit(seg_name: str, groups: pd.DataFrame) -> None:
        out[seg_name] = {}
        for key, sub in groups.groupby(groups.iloc[:, 0], dropna=False, observed=False):
            label = "NaN" if pd.isna(key) else str(key)
            out[seg_name][label] = {
                "n": int(len(sub)),
                "mae": float(sub["_abs_err"].mean()),
                "mae_vs_position_mean": float(sub["_pos_mean_err"].mean()),
            }

    # is_dome / is_home — clean integer columns
    for col in ("is_dome", "is_home"):
        if col in df.columns:
            _emit(col, df[[col, "_abs_err", "_pos_mean_err"]])

    # surface (normalised)
    if "surface" in df.columns:
        df["_surface_norm"] = df["surface"].map(_normalise_surface)
        _emit("surface", df[["_surface_norm", "_abs_err", "_pos_mean_err"]])

    # game_wind quartile (cut on train quartiles)
    if "game_wind" in df.columns and train_df["game_wind"].notna().sum() > 100:
        df["_wind_q"] = _quartile_cut(df["game_wind"], train_df["game_wind"])
        _emit("game_wind_quartile", df[["_wind_q", "_abs_err", "_pos_mean_err"]])

    # week bucket
    if "week" in df.columns:
        df["_week_bucket"] = df["week"].apply(_week_bucket)
        _emit("week_bucket", df[["_week_bucket", "_abs_err", "_pos_mean_err"]])

    # implied_team_total quartile (cut on train)
    if "implied_team_total" in df.columns and train_df["implied_team_total"].notna().sum() > 100:
        df["_itt_q"] = _quartile_cut(df["implied_team_total"], train_df["implied_team_total"])
        _emit("implied_total_quartile", df[["_itt_q", "_abs_err", "_pos_mean_err"]])

    # Top-10 kickers by sample count on test
    if "player_id" in df.columns:
        counts = df["player_id"].value_counts().head(10)
        top10 = df[df["player_id"].isin(counts.index)].copy()
        top10["_pid_label"] = top10["player_id"].astype(str)
        out["top_10_kickers"] = {}
        for pid, sub in top10.groupby("_pid_label", observed=False):
            out["top_10_kickers"][pid] = {
                "n": int(len(sub)),
                "mae": float(sub["_abs_err"].mean()),
                "mae_vs_position_mean": float(sub["_pos_mean_err"].mean()),
            }

    return out


def _flag_segments(segments: dict, low: float, high: float) -> list[dict]:
    """Return entries with MAE outside [low, high]."""
    flagged = []
    for seg_name, vals in segments.items():
        for label, stats in vals.items():
            if stats["n"] < 20:  # too small to interpret
                continue
            mae = stats["mae"]
            if mae < low or mae > high:
                flagged.append(
                    {
                        "segment": seg_name,
                        "value": label,
                        "n": stats["n"],
                        "mae": mae,
                        "mae_vs_position_mean": stats["mae_vs_position_mean"],
                    }
                )
    flagged.sort(key=lambda r: r["mae"], reverse=True)
    return flagged


def _plot_segments(segments: dict, position_mean_mae: float, path: Path) -> None:
    """Bar chart: model MAE per segment value with the position-mean MAE line."""
    rows = []
    for seg_name, vals in segments.items():
        if seg_name == "top_10_kickers":
            continue  # too many cats; keep the chart readable
        for label, stats in vals.items():
            if stats["n"] < 20:
                continue
            rows.append((f"{seg_name}={label}", stats["mae"], stats["n"]))
    if not rows:
        return
    labels = [r[0] for r in rows]
    maes = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(labels)), 5))
    ax.bar(range(len(labels)), maes, color="steelblue")
    ax.axhline(
        position_mean_mae,
        color="orange",
        linestyle="--",
        label=f"position_mean baseline ({position_mean_mae:.3f})",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("MAE (fantasy points)")
    ax.set_title("K Ridge model MAE by segment (test split)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ── Recommendation logic ──────────────────────────────────────────────────


def _decide(best_naive_mae: float, best_model_mae: float) -> tuple[str, str]:
    """Return (category, one-line rationale)."""
    gap = best_naive_mae - best_model_mae
    if gap > GAP_PROCEED:
        return (
            "proceed_to_phase_2",
            f"best model beats naive by {gap:+.3f} > {GAP_PROCEED}; signal exists.",
        )
    if gap > GAP_CAPPED:
        return (
            "phase_2_capped",
            f"best model beats naive by {gap:+.3f} (≤ {GAP_PROCEED}); small but real headroom.",
        )
    if abs(gap) <= GAP_CAPPED:
        return (
            "accept_noise_floor",
            f"|gap| {abs(gap):.3f} ≤ {GAP_CAPPED}; K is at the noise floor.",
        )
    return (
        "bug_investigate",
        f"best model worse than naive by {-gap:+.3f} > {GAP_BUG}; something is wrong.",
    )


# ── Stdout formatters ─────────────────────────────────────────────────────


def _print_baselines(baselines: dict) -> None:
    print("\nNaive baselines (fantasy-point MAE):")
    print(f"    {'baseline':<24s} {'val_mae':>9s} {'test_mae':>9s}")
    for name, vals in baselines.items():
        v = vals.get("val_mae")
        t = vals.get("test_mae")
        v_s = "—" if v is None else f"{v:9.3f}"
        t_s = "—" if t is None else f"{t:9.3f}"
        print(f"    {name:<24s} {v_s:>9s} {t_s:>9s}")


def _print_models(in_script: dict, from_bench: dict) -> None:
    print("\nIn-script refits (fantasy-point MAE):")
    print(f"    {'model':<10s} {'val_mae':>9s} {'test_mae':>9s}")
    for name, vals in in_script.items():
        v = vals.get("val_mae")
        t = vals.get("test_mae")
        v_s = "—" if v is None else f"{v:9.3f}"
        t_s = "—" if t is None else f"{t:9.3f}"
        print(f"    {name:<10s} {v_s:>9s} {t_s:>9s}")
    print("\nFrom production benchmark (test MAE, for reference):")
    for name in ("ridge", "nn", "attn", "lgbm"):
        if name in from_bench:
            print(f"    {name:<10s} {from_bench[name]:>9.3f}")


def _print_segments(segments: dict) -> None:
    print("\nPer-segment MAE (test, in-script Ridge):")
    print(f"    {'segment':<24s} {'value':<24s} {'n':>5s} {'mae':>8s} {'vs_pm':>8s}")
    for seg_name, vals in segments.items():
        if seg_name == "top_10_kickers":
            continue
        for label, stats in vals.items():
            print(
                f"    {seg_name:<24s} {label[:24]:<24s} {stats['n']:>5d} "
                f"{stats['mae']:>8.3f} {stats['mae_vs_position_mean']:>8.3f}"
            )


# ── Benchmark JSON loader ────────────────────────────────────────────────


def _load_benchmark_mae(path: Path) -> dict[str, float]:
    """Return the K MAE numbers from a benchmark_history JSON. Falls back to
    the hard-coded values from the docstring if the file is missing.
    """
    if not path.exists():
        return dict(BENCHMARK_FALLBACK)
    try:
        d = json.loads(path.read_text())
        for entry in d.get("results", []):
            if entry.get("position") == "K":
                return {
                    "ridge": float(entry.get("ridge_mae", BENCHMARK_FALLBACK["ridge"])),
                    "nn": float(entry.get("nn_mae", BENCHMARK_FALLBACK["nn"])),
                    "attn": float(entry.get("attn_nn_mae", BENCHMARK_FALLBACK["attn"])),
                    "lgbm": float(entry.get("lgbm_mae", BENCHMARK_FALLBACK["lgbm"])),
                }
    except (json.JSONDecodeError, OSError):
        pass
    return dict(BENCHMARK_FALLBACK)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the LGBM refit (saves ~10s during script iteration)",
    )
    args = parser.parse_args()

    # ── Load + materialise features ────────────────────────────────────────
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

    # Order matters: compute_targets writes `fantasy_points` (signed total),
    # which compute_features reads for the rolling total-points columns.
    k_df = compute_targets(k_df)
    print("Computing K features on full dataset (in-place) …")
    compute_features(k_df)

    train_df, val_df, test_df = season_split(k_df)
    n_rows = {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))}
    print(f"  rows: train={n_rows['train']}, val={n_rows['val']}, test={n_rows['test']}")
    if n_rows["test"] == 0 or n_rows["val"] == 0:
        print("ERROR: val or test split is empty; cannot evaluate baselines.")
        return 1

    # Filter to features actually present (defensive — the audit found all
    # features present on the materialised frame, but keep this guard so a
    # future column rename doesn't crash the script).
    feature_cols = [c for c in ALL_FEATURES if c in train_df.columns]
    missing = [c for c in ALL_FEATURES if c not in train_df.columns]
    if missing:
        print(f"  WARN: {len(missing)} feature(s) missing from frame: {missing}")

    val_fp = val_df["fantasy_points"].to_numpy(dtype=float)
    test_fp = test_df["fantasy_points"].to_numpy(dtype=float)

    # ── Naive baselines ────────────────────────────────────────────────────
    baselines: dict[str, dict] = {}

    # 1. position_mean (constant: train fantasy-points mean)
    position_mean = float(train_df["fantasy_points"].mean())
    baselines["position_mean"] = {
        "val_mae": float(np.mean(np.abs(val_fp - position_mean))),
        "test_mae": float(np.mean(np.abs(test_fp - position_mean))),
        "predicted_value": position_mean,
    }

    # 2/3/4. Rolling baselines — fit on the prior-row chain across the full
    # frame, then read off val/test rows by index. shift(1) inside the rolling
    # fn means a 2024 row uses 2015–2023-prior data; eval rows never see their
    # own value.
    for name, group_col, sort_cols, window in [
        ("kicker_career_mean", "player_id", ["player_id", "season", "week"], None),
        ("kicker_L8_mean", "player_id", ["player_id", "season", "week"], 8),
        ("team_kicker_mean", "recent_team", ["recent_team", "season", "week"], 16),
    ]:
        rolling = _grouped_rolling_mean(k_df, group_col, sort_cols, window, position_mean)
        val_pred = rolling.loc[val_df.index].to_numpy()
        test_pred = rolling.loc[test_df.index].to_numpy()
        baselines[name] = {
            "val_mae": float(np.mean(np.abs(val_fp - val_pred))),
            "test_mae": float(np.mean(np.abs(test_fp - test_pred))),
        }

    # 5. vegas_only Ridge
    vegas_preds = _vegas_only_ridge(train_df, val_df, test_df)
    if vegas_preds is None:
        baselines["vegas_only"] = {"val_mae": None, "test_mae": None, "note": "skipped"}
    else:
        vp_val, vp_test = vegas_preds
        baselines["vegas_only"] = {
            "val_mae": float(np.mean(np.abs(val_fp - vp_val))),
            "test_mae": float(np.mean(np.abs(test_fp - vp_test))),
        }

    _print_baselines(baselines)

    # ── In-script Ridge ────────────────────────────────────────────────────
    print("\nRefitting Ridge on ALL_FEATURES (alpha=1.0, one model per target) …")
    ridge_val_fp, ridge_test_fp = _refit_ridge(train_df, val_df, test_df, feature_cols)
    in_script = {
        "ridge": {
            "val_mae": float(np.mean(np.abs(val_fp - ridge_val_fp))),
            "test_mae": float(np.mean(np.abs(test_fp - ridge_test_fp))),
        }
    }

    # ── In-script LGBM ────────────────────────────────────────────────────
    if not args.quick:
        print("Refitting LightGBM on ALL_FEATURES (n_est=100 per target) …")
        lgbm = _refit_lgbm(train_df, val_df, test_df, feature_cols)
        if lgbm is None:
            in_script["lgbm"] = {"val_mae": None, "test_mae": None, "note": "lightgbm unavailable"}
        else:
            lgbm_val_fp, lgbm_test_fp = lgbm
            in_script["lgbm"] = {
                "val_mae": float(np.mean(np.abs(val_fp - lgbm_val_fp))),
                "test_mae": float(np.mean(np.abs(test_fp - lgbm_test_fp))),
            }
    else:
        in_script["lgbm"] = {"val_mae": None, "test_mae": None, "note": "skipped (--quick)"}

    benchmark = _load_benchmark_mae(BENCHMARK_PATH)
    _print_models(in_script, benchmark)

    # Sanity vs production Ridge — must be in the ballpark or the diagnostic
    # is measuring something different from the production pipeline.
    prod_ridge = benchmark.get("ridge")
    if prod_ridge is not None:
        script_ridge = in_script["ridge"]["test_mae"]
        delta = abs(script_ridge - prod_ridge)
        if delta > 0.20:
            print(
                f"\nWARN: in-script Ridge test MAE ({script_ridge:.3f}) differs from "
                f"production benchmark ({prod_ridge:.3f}) by {delta:.3f} (>0.20). "
                f"Investigate feature-set / split / NaN-fill mismatch."
            )

    # ── Per-segment MAE on test (Ridge) ────────────────────────────────────
    print("\nComputing per-segment MAE (Ridge predictions on test) …")
    segments = _segment_mae(
        test_df=test_df,
        ground_truth=test_fp,
        preds=ridge_test_fp,
        train_df=train_df,
        position_mean=position_mean,
    )
    _print_segments(segments)

    flagged = _flag_segments(segments, low=SEGMENT_LOW, high=SEGMENT_HIGH)
    if flagged:
        print(f"\nFlagged segments (MAE < {SEGMENT_LOW} or > {SEGMENT_HIGH}):")
        for f in flagged:
            print(
                f"    {f['segment']:<24s} {f['value'][:24]:<24s} "
                f"n={f['n']:<5d} mae={f['mae']:.3f} vs_pm={f['mae_vs_position_mean']:.3f}"
            )
    else:
        print(f"\nNo flagged segments (all MAE within [{SEGMENT_LOW}, {SEGMENT_HIGH}]).")

    # ── Recommendation ─────────────────────────────────────────────────────
    # Best on test only — val is used for parity sanity, test drives the call.
    naive_test_maes = {
        name: vals["test_mae"]
        for name, vals in baselines.items()
        if vals.get("test_mae") is not None
    }
    model_test_maes = {
        name: vals["test_mae"]
        for name, vals in in_script.items()
        if vals.get("test_mae") is not None
    }
    # Also include the benchmark Ridge/NN/Attn/LGBM in the "best model" pool —
    # if a production model beats every naive baseline by a hair, that's still
    # the call.
    for name, mae in benchmark.items():
        model_test_maes[f"benchmark_{name}"] = mae

    best_naive_name, best_naive_mae = min(naive_test_maes.items(), key=lambda kv: kv[1])
    best_model_name, best_model_mae = min(model_test_maes.items(), key=lambda kv: kv[1])

    recommendation, rationale = _decide(best_naive_mae, best_model_mae)
    summary = (
        f"K signal-floor diagnostic on test (2025, n={n_rows['test']}): "
        f"best naive baseline = {best_naive_name} at MAE {best_naive_mae:.3f}; "
        f"best model = {best_model_name} at MAE {best_model_mae:.3f} "
        f"(gap {best_naive_mae - best_model_mae:+.3f}). {rationale} "
        f"Flagged segments: {len(flagged)}."
    )

    print("\n" + "=" * 72)
    print(f"RECOMMENDATION: {recommendation}")
    print("=" * 72)
    print(f"  best naive: {best_naive_name} (test MAE {best_naive_mae:.3f})")
    print(f"  best model: {best_model_name} (test MAE {best_model_mae:.3f})")
    print(f"  gap (naive - model): {best_naive_mae - best_model_mae:+.3f}")
    print(f"  rationale: {rationale}")
    print(f"  flagged segments: {len(flagged)}")

    # ── Persist ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "n_rows": n_rows,
        "baselines": baselines,
        "models_in_script": in_script,
        "models_from_benchmark": {
            **{k: float(v) for k, v in benchmark.items()},
            "source": str(BENCHMARK_PATH.relative_to(PROJECT_ROOT))
            if BENCHMARK_PATH.exists()
            else "fallback (file missing)",
        },
        "segments": segments,
        "flagged_segments": flagged,
        "best_naive": {"name": best_naive_name, "test_mae": best_naive_mae},
        "best_model": {"name": best_model_name, "test_mae": best_model_mae},
        "recommendation": recommendation,
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"\n✔ wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    _plot_segments(
        segments,
        position_mean_mae=baselines["position_mean"]["test_mae"],
        path=OUT_PNG,
    )
    if OUT_PNG.exists():
        print(f"✔ wrote {OUT_PNG.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
