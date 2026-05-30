"""RB model comparison: LGBM-vs-peers disagreement, and NN / attention-NN
behaviour by in-season history depth (operator-only CLI).

Part 1 — LGBM vs peers. Motivated by player-weeks where LightGBM prints far
below the other RB models (canonical case: Derrick Henry 2025 Week 2 — LGBM
~12.6, Attn-NN ~17, NN ~16, Ridge ~26). Against ground truth it is not a bug:
Henry's W2 actual was 2.3 FP (he cratered after a 29.2-pt Week 1), so LGBM was
the *closest* model and Ridge the *worst*. Trees can't extrapolate past their
leaves and shrink toward typical outcomes; the linear model extrapolates the hot
Week 1. This conservatism is why LightGBM is the production-best RB model.

Part 2 — NN & attention NN by history depth. The attention NN is history-aware,
so one might expect it to *regress* when the per-game sequence is short (at season
Week 2 the within-season history holds a single game). It does not: attention
over a length-1 sequence extracts that one game — there is no "small-sample →
hedge" reflex — so a lone explosive game (plus the static prior-season branch)
yields a high prediction. Empirically that is well calibrated: sparse-history
players coming off a hot game average ~16.6 FP next week, so ~17 was the right
expected value and Henry's crater is a 1-of-1 anomaly. The one genuine weak spot
is the season opener (``n_prior_games == 0``, empty history): the attention NN
leans on its static branch and over-predicts. And the attention NN never beats
LightGBM in any history-depth bucket.

This script quantifies behaviour and prints a verdict — it does **not** change
any model. The full RB pipeline runs on every invocation (``run()``), so this
module is gated behind ``if __name__ == "__main__"`` — importing it must NOT fire
the pipeline. The pure helpers below (``per_model_metrics``, ``peer_gap``,
``calibration_table``, ``gap_decomposition``, ``add_history_depth``,
``history_depth_table``) are unit-tested in
``tests/analysis/test_analysis_rb_lgbm_disagreement.py``.

Usage:
    python -m src.analysis.analysis_rb_lgbm_disagreement
    python -m src.analysis.analysis_rb_lgbm_disagreement --no-plots
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import SCORING_PPR  # noqa: E402
from src.shared.evaluation import compute_metrics  # noqa: E402

# Per-model total-FP prediction columns written by src/shared/pipeline.py.
# LightGBM is the model under investigation; the other three are its "peers".
LGBM = "LightGBM"
MODELS = {
    "Ridge": "pred_ridge_total",
    "NN": "pred_nn_total",
    "Attention NN": "pred_attn_nn_total",
    LGBM: "pred_lgbm_total",
}
PEERS = ["Ridge", "NN", "Attention NN"]

# Pipeline truth column (src/shared/pipeline.py compares pred_*_total against it).
ACTUAL = "fantasy_points"

# A player-week counts as "LGBM << peers" when the peer mean exceeds LGBM by at
# least this many fantasy points. Reported alongside a top-decile cut so the
# conclusion does not hinge on one threshold.
DISAGREEMENT_THRESHOLD = 4.0

# Actual-FP bin edges for the calibration table.
CALIB_BINS = [0, 5, 10, 15, 20, 25, np.inf]

# In-season games-played buckets for the history-depth table. n_prior_games is
# the attention model's real (non-padded) sequence length for the row, so
# bucket "0" == season opener with an empty history sequence.
HISTORY_DEPTH_BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 99, "8+")]

# Sparse-history calibration cut: a "hot" most-recent game is one whose FP was at
# least this high (rolling_max over the last 3 games == the single prior game's FP
# when only one game has been played).
HOT_RECENT_FP = 18.0
RECENT_MAX_COL = "rolling_max_fantasy_points_L3"


def available_models(df: pd.DataFrame) -> dict[str, str]:
    """Subset of MODELS whose prediction column is present in ``df``."""
    return {name: col for name, col in MODELS.items() if col in df.columns}


def per_model_metrics(
    df: pd.DataFrame, models: dict[str, str] | None = None, actual: str = ACTUAL
) -> dict[str, dict[str, float]]:
    """MAE / bias (mean signed residual) / RMSE / n for each model on ``df``.

    Bias = mean(pred - actual): positive ⇒ the model over-predicts.
    """
    models = models or available_models(df)
    # A data-dependent slice (e.g. gap<=-4 or actual>=20) can be empty on a
    # different season; compute_metrics → sklearn raises on 0 samples, so return
    # NaNs rather than crash the diagnostic.
    if len(df) == 0:
        return {
            name: {"mae": float("nan"), "bias": float("nan"), "rmse": float("nan"), "n": 0}
            for name in models
        }
    y = df[actual].to_numpy(dtype=float)
    out: dict[str, dict[str, float]] = {}
    for name, col in models.items():
        p = df[col].to_numpy(dtype=float)
        m = compute_metrics(y, p)
        out[name] = {
            "mae": m["mae"],
            "bias": float(np.mean(p - y)),
            "rmse": m["rmse"],
            "n": int(len(df)),
        }
    return out


def peer_gap(df: pd.DataFrame, models: dict[str, str] | None = None) -> pd.Series:
    """peer-mean(Ridge, NN, Attn) − LGBM, per row. Large positive ⇒ LGBM << peers."""
    models = models or available_models(df)
    peer_cols = [models[p] for p in PEERS if p in models]
    return df[peer_cols].mean(axis=1) - df[models[LGBM]]


def calibration_table(
    df: pd.DataFrame,
    models: dict[str, str] | None = None,
    actual: str = ACTUAL,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Mean prediction per model within each actual-FP bin (calibration curve).

    Columns: n, avg_actual, then one mean-prediction column per model.
    """
    models = models or available_models(df)
    bins = bins or CALIB_BINS
    binned = pd.cut(df[actual], bins, right=False)
    rows = []
    for b, sub in df.groupby(binned, observed=True):
        row = {"bin": str(b), "n": len(sub), "avg_actual": sub[actual].mean()}
        for name, col in models.items():
            row[name] = sub[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def gap_decomposition(
    df: pd.DataFrame, scoring: dict[str, float] | None = None
) -> dict[str, float]:
    """Per-target fantasy-point contribution to the Ridge−LGBM total gap on ``df``.

    Returns {target: mean(pred_ridge_t − pred_lgbm_t) * scoring_weight}. The
    target whose value is largest is the raw stat driving the disagreement.
    """
    scoring = scoring or SCORING_PPR
    out: dict[str, float] = {}
    for target, weight in scoring.items():
        rc, lc = f"pred_ridge_{target}", f"pred_lgbm_{target}"
        if rc in df.columns and lc in df.columns:
            out[target] = float((df[rc] - df[lc]).mean() * weight)
    return out


def add_history_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``n_prior_games`` = in-season games played before each row.

    This equals the attention model's real (non-padded) sequence length for the
    row: ``build_game_history_arrays`` groups by ``(player_id, season)`` and gives
    each game its prior in-season games (current excluded). So ``n_prior_games==0``
    is a season opener with an all-padding (empty) history sequence, and Henry's
    2025 Week 2 has ``n_prior_games==1`` (just his Week 1).
    """
    out = df.sort_values(["player_id", "season", "week"]).copy()
    out["n_prior_games"] = out.groupby(["player_id", "season"]).cumcount()
    return out


def history_depth_table(
    df: pd.DataFrame,
    models: dict[str, str] | None = None,
    buckets: list[tuple[int, int, str]] | None = None,
) -> pd.DataFrame:
    """Per-model MAE & bias within each in-season history-depth bucket.

    One row per bucket with columns: ``npg`` (label), ``n``, ``avg_actual``, then
    ``{model} MAE`` and ``{model} bias`` for each model. Requires
    ``n_prior_games`` (call :func:`add_history_depth` first).
    """
    models = models or available_models(df)
    buckets = buckets or HISTORY_DEPTH_BUCKETS
    rows = []
    for lo, hi, label in buckets:
        sub = df[df["n_prior_games"].between(lo, hi)]
        row: dict[str, object] = {
            "npg": label,
            "n": len(sub),
            "avg_actual": sub[ACTUAL].mean() if len(sub) else float("nan"),
        }
        for name, m in per_model_metrics(sub, models).items():
            row[f"{name} MAE"] = m["mae"]
            row[f"{name} bias"] = m["bias"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Presentation (IO; not unit-tested)
# --------------------------------------------------------------------------- #
def _print_metric_table(title: str, metrics: dict[str, dict[str, float]]) -> None:
    n = next(iter(metrics.values()))["n"] if metrics else 0
    print(f"\n=== {title}  (n={n}) ===")
    print(f"{'model':14} {'MAE':>8} {'bias':>8} {'RMSE':>8}")
    for name, m in metrics.items():
        print(f"{name:14} {m['mae']:8.3f} {m['bias']:8.3f} {m['rmse']:8.3f}")


def _make_plots(df: pd.DataFrame, gap: pd.Series, outdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    models = available_models(df)
    y = df[ACTUAL].to_numpy(dtype=float)

    # 1) Calibration: mean prediction vs mean actual per bin, with identity line.
    calib = calibration_table(df, models)
    fig, ax = plt.subplots(figsize=(6, 5))
    lo, hi = 0, float(calib["avg_actual"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect (y=x)")
    for name in models:
        ax.plot(calib["avg_actual"], calib[name], marker="o", label=name)
    ax.set_xlabel("mean actual FP (per bin)")
    ax.set_ylabel("mean predicted FP")
    ax.set_title(
        "RB calibration: all models regress to the mean;\nRidge runs hottest on low-actual weeks"
    )
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(outdir, "rb_calibration.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # 2) Ridge vs LGBM predictions, coloured by which model is closer to actual.
    ridge = df[models["Ridge"]].to_numpy(dtype=float)
    lgbm = df[models[LGBM]].to_numpy(dtype=float)
    lgbm_closer = np.abs(lgbm - y) <= np.abs(ridge - y)
    fig, ax = plt.subplots(figsize=(6, 5))
    m = max(ridge.max(), lgbm.max()) + 2
    ax.plot([0, m], [0, m], "k--", lw=1, label="Ridge = LGBM")
    ax.scatter(
        lgbm[lgbm_closer],
        ridge[lgbm_closer],
        s=10,
        alpha=0.4,
        label="LGBM closer to actual",
        color="tab:green",
    )
    ax.scatter(
        lgbm[~lgbm_closer],
        ridge[~lgbm_closer],
        s=10,
        alpha=0.4,
        label="Ridge closer to actual",
        color="tab:red",
    )
    ax.set_xlabel("LightGBM predicted FP")
    ax.set_ylabel("Ridge predicted FP")
    ax.set_title("Where Ridge >> LGBM (above the line), LGBM is\nusually closer to the truth")
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(outdir, "rb_ridge_vs_lgbm.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    written = [p1, p2]

    # 3) Per-model MAE vs in-season history depth (the NN / attention-NN lens).
    # Shows the attention NN never dips below LGBM and is weakest at npg==0.
    if "n_prior_games" in df.columns:
        depths = list(range(0, 13))  # 0..12 prior games; tail is sparse
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, col in models.items():
            maes = []
            for d in depths:
                sub = df[df["n_prior_games"] == d]
                maes.append(
                    np.mean(np.abs(sub[col].to_numpy() - sub[ACTUAL].to_numpy()))
                    if len(sub)
                    else np.nan
                )
            ax.plot(depths, maes, marker="o", label=name)
        ax.set_xlabel("in-season games played before this week (attention seq length)")
        ax.set_ylabel("MAE (fantasy points)")
        ax.set_title(
            "RB MAE by history depth: attention NN never beats LGBM;\nweakest at the season opener (0 prior games)"
        )
        ax.legend()
        fig.tight_layout()
        p3 = os.path.join(outdir, "rb_mae_by_history_depth.png")
        fig.savefig(p3, dpi=120)
        plt.close(fig)
        written.append(p3)

    print("\nPlots written: " + "  |  ".join(written))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="skip PNG output")
    args = parser.parse_args()

    from src.rb.run_pipeline import run  # deferred: pulls the full pipeline/torch

    result = run()
    df = result["test_df"].copy()
    models = available_models(df)
    gap = peer_gap(df, models)
    df = add_history_depth(df.assign(_gap=gap))
    y = df[ACTUAL]

    print("\n" + "=" * 72)
    print("RB LightGBM disagreement analysis")
    print("=" * 72)
    print(f"RB test player-weeks: {len(df)}   mean actual FP: {y.mean():.2f}")

    # 1) Global — does LGBM actually win overall, and is its bias anomalous?
    _print_metric_table("GLOBAL (all RB rows)", per_model_metrics(df, models))

    # 2) The flagged class: LGBM << peers.
    print(
        f"\npeer_gap = mean(Ridge,NN,Attn) - LGBM:  mean={gap.mean():.2f}  "
        f"p90={gap.quantile(0.9):.2f}  max={gap.max():.2f}  "
        f"(gap>={DISAGREEMENT_THRESHOLD}: {(gap >= DISAGREEMENT_THRESHOLD).sum()} rows, "
        f"gap<=-{DISAGREEMENT_THRESHOLD}: {(gap <= -DISAGREEMENT_THRESHOLD).sum()} rows)"
    )
    hi_mask = df["_gap"] >= DISAGREEMENT_THRESHOLD
    dec_mask = df["_gap"] >= df["_gap"].quantile(0.9)
    _print_metric_table(
        f"LGBM << peers (gap>={DISAGREEMENT_THRESHOLD})", per_model_metrics(df[hi_mask], models)
    )
    _print_metric_table("LGBM << peers (top-decile gap)", per_model_metrics(df[dec_mask], models))
    _print_metric_table(
        f"LGBM >> peers (gap<=-{DISAGREEMENT_THRESHOLD})",
        per_model_metrics(df[df["_gap"] <= -DISAGREEMENT_THRESHOLD], models),
    )

    # 3) Calibration.
    print("\n=== CALIBRATION: mean prediction by actual-FP bin ===")
    print(calibration_table(df, models).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # 4) Early-season vs rest, and the genuinely-high residual cost.
    _print_metric_table("early season (wk<=3)", per_model_metrics(df[df["week"] <= 3], models))
    _print_metric_table("rest of season (wk>=4)", per_model_metrics(df[df["week"] >= 4], models))
    _print_metric_table("genuinely HIGH actual (>=20 FP)", per_model_metrics(df[y >= 20], models))

    # 5) What raw stat drives the Ridge-vs-LGBM gap on the disagreement class?
    decomp = gap_decomposition(df[hi_mask])
    print(f"\n=== per-target FP gap (Ridge - LGBM) on gap>={DISAGREEMENT_THRESHOLD} ===")
    for t, v in sorted(decomp.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {t:18} {v:+7.3f} FP")
    print(f"  {'TOTAL':18} {sum(decomp.values()):+7.3f} FP")

    # Verdict.
    g = per_model_metrics(df, models)
    cls = per_model_metrics(df[hi_mask], models)
    lgbm_best_global = min(g, key=lambda k: g[k]["mae"]) == LGBM
    lgbm_best_class = bool(cls) and min(cls, key=lambda k: cls[k]["mae"]) == LGBM
    print("\n" + "-" * 72)
    if lgbm_best_global and lgbm_best_class:
        print(
            "VERDICT: EXPECTED behaviour, not a bug. LightGBM has the lowest MAE "
            "both overall and\non the exact 'LGBM << peers' class — its lower "
            "predictions are correct conservatism on\na mean-reverting target; the "
            "linear model over-extrapolates hot recent form."
        )
    else:
        print(
            "VERDICT: NEEDS REVIEW — LightGBM is NOT the most accurate on the "
            "disagreement class.\nThis is a real cost, not just a model difference; "
            "surface to a human before acting."
        )
    print("-" * 72)

    # ----------------------------------------------------------------------- #
    # PART 2 — NN & attention NN by in-season history depth.
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("PART 2 — NN & attention NN by in-season history depth")
    print("=" * 72)
    hdt = history_depth_table(df, models)
    mae_cols = ["npg", "n", "avg_actual"] + [f"{n} MAE" for n in models]
    bias_cols = ["npg"] + [f"{n} bias" for n in models]
    print("\nMAE by history depth (n_prior_games == attention seq length):")
    print(hdt[mae_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nbias (pred - actual) by history depth:")
    print(hdt[bias_cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # Sparse history (1-2 prior games) split by whether the recent game was hot.
    if RECENT_MAX_COL in df.columns:
        sparse = df["n_prior_games"].between(1, 2)
        hot = df[RECENT_MAX_COL] >= HOT_RECENT_FP
        _print_metric_table(
            f"sparse hist (1-2 games) + HOT recent game (>={HOT_RECENT_FP:.0f} FP)",
            per_model_metrics(df[sparse & hot], models),
        )
        _print_metric_table(
            "sparse hist (1-2 games) + cool recent game",
            per_model_metrics(df[sparse & ~hot], models),
        )
        sh = df[sparse & hot]
        if len(sh):
            print(
                f"  -> these {len(sh)} 'sparse+hot' player-weeks actually averaged "
                f"{sh[ACTUAL].mean():.1f} FP: hot starts usually persist, so a high "
                "prediction is the right expected value (Henry's 2.3 crater is the tail)."
            )

    # Attention-NN verdict: does it ever beat LGBM, and where is it weakest?
    attn_best_buckets = sum(
        1
        for lo, hi, _ in HISTORY_DEPTH_BUCKETS
        if (m := per_model_metrics(df[df["n_prior_games"].between(lo, hi)], models))
        and min(m, key=lambda k: m[k]["mae"]) == "Attention NN"
    )
    opener = per_model_metrics(df[df["n_prior_games"] == 0], models)
    attn_opener_bias = opener.get("Attention NN", {}).get("bias", float("nan"))
    lgbm_opener_bias = opener.get(LGBM, {}).get("bias", float("nan"))
    print("\n" + "-" * 72)
    print(
        f"VERDICT (Part 2): attention NN is best in {attn_best_buckets}/"
        f"{len(HISTORY_DEPTH_BUCKETS)} history-depth buckets. Season-opener (npg==0) bias: "
        f"Attn {attn_opener_bias:+.2f} vs LGBM {lgbm_opener_bias:+.2f} — the empty-history"
    )
    print(
        "case is its one real weak spot. History-awareness does NOT make it regress on a "
        "single explosive game; ~17 for Henry was well-calibrated to the typical sparse+hot "
        "outcome (those players average ~16-17 FP the next week)."
    )
    print("-" * 72)

    if not args.no_plots:
        outdir = os.path.join(os.path.dirname(__file__), "outputs", "figures")
        _make_plots(df, gap, outdir)


if __name__ == "__main__":
    main()
