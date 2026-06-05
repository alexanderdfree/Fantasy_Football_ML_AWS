"""Week-by-week and subgroup accuracy analysis, focused on the Attention NN.

Read-only diagnostic. Runs each position's production pipeline (one or more
seeds), pulls the per-row ``result["test_df"]`` (which carries the actual
``fantasy_points`` plus every model's ``pred_*_total`` column), and slices the
errors three ways:

  1. Week-by-week    — per-model MAE & signed bias for each test-season week.
  2. Subgroups       — actual-score tier, week-phase (early/mid/late), and
                       position, judged by *bias* (the project lesson: on
                       skewed FP targets, "is the model worse on X" is a bias
                       question, not an MAE one) plus bias-corrected MAE.
  3. ATTN head-to-head — where the Attention NN wins / loses vs its best peer
                       (Ridge / NN / LightGBM), per slice.

Multi-seed by default (42,123): a single-seed NN overall-MAE is noise, so ATTN
slice metrics are reported as mean +/- std across seeds and a slice "win" is
only flagged when its direction is consistent across seeds.

Usage:
    python -m src.analysis.attn_weekly_accuracy
    python -m src.analysis.attn_weekly_accuracy --positions QB RB --seeds 42
    python -m src.analysis.attn_weekly_accuracy --report report.md --figdir /tmp/figs
    python -m src.analysis.attn_weekly_accuracy --cache .attn_cache  # reuse test_dfs
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cohort_analysis import (  # noqa: E402
    ACTUAL,
    available_models,
    per_model_metrics,
)
from src.config import POSITIONS  # noqa: E402

# The four trained models we compare; ATTN is the subject of the study.
MODELS = {
    "Ridge": "pred_ridge_total",
    "NN": "pred_nn_total",
    "Attention NN": "pred_attn_nn_total",
    "LightGBM": "pred_lgbm_total",
}
ATTN = "Attention NN"
PEERS = ["Ridge", "NN", "LightGBM"]

# Week-phase buckets across an 18-game regular season + playoffs.
EARLY, MID, LATE = "early (W1-4)", "mid (W5-13)", "late (W14+)"


# --------------------------------------------------------------------------- #
# Pipeline driving / caching
# --------------------------------------------------------------------------- #
def _run_position(pos: str, seed: int) -> pd.DataFrame:
    """Run one position's production pipeline and return its test_df slice."""
    mod = importlib.import_module(f"src.{pos.lower()}.run_pipeline")
    result = mod.run(seed=seed)
    df = result["test_df"].copy()
    df["position"] = pos
    df["seed"] = seed
    keep = ["position", "seed", "player_id", "season", "week", ACTUAL] + [
        c for c in MODELS.values() if c in df.columns
    ]
    # Keep a couple of role/volume columns when present for richer subgrouping.
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def collect(positions, seeds, cache_dir: str | None) -> pd.DataFrame:
    """Gather per-row test predictions for every (position, seed), with cache."""
    frames = []
    for pos in positions:
        for seed in seeds:
            cpath = None
            if cache_dir:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                cpath = Path(cache_dir) / f"{pos.lower()}_seed{seed}.parquet"
                if cpath.exists():
                    print(f"  [cache] {pos} seed={seed}")
                    frames.append(pd.read_parquet(cpath))
                    continue
            print(f"  [run]   {pos} seed={seed} ...", flush=True)
            df = _run_position(pos, seed)
            if cpath is not None:
                df.to_parquet(cpath, index=False)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Aggregation helpers (seed-aware)
# --------------------------------------------------------------------------- #
def _models_in(df: pd.DataFrame) -> dict[str, str]:
    return available_models(df, MODELS)


def _metric_by(df: pd.DataFrame, by, metric: str) -> pd.DataFrame:
    """Per-seed per-model `metric` grouped by `by`, then mean+/-std over seeds.

    Returns a tidy frame: columns [*by, model, mean, std, n].
    """
    models = _models_in(df)
    rows = []
    by = list(by)
    for keys, sub in df.groupby(["seed", *by], observed=True):
        seed = keys[0]
        gkeys = keys[1:]
        pm = per_model_metrics(sub, models)
        for name, m in pm.items():
            rows.append(
                {
                    **dict(zip(by, gkeys, strict=False)),
                    "seed": seed,
                    "model": name,
                    "value": m[metric],
                    "n": m["n"],
                }
            )
    per_seed = pd.DataFrame(rows)
    agg = (
        per_seed.groupby([*by, "model"], observed=True)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("n", "mean"))
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def _pivot(agg: pd.DataFrame, index, value="mean") -> pd.DataFrame:
    return agg.pivot_table(index=index, columns="model", values=value, observed=True)


def _attn_vs_best_peer(agg_mae: pd.DataFrame, index) -> pd.DataFrame:
    """For each slice: ATTN MAE, best-peer MAE, delta (ATTN - best peer)."""
    wide = _pivot(agg_mae, index, "mean")
    peers = [p for p in PEERS if p in wide.columns]
    out = pd.DataFrame(index=wide.index)
    out["attn_mae"] = wide.get(ATTN)
    out["best_peer"] = wide[peers].idxmin(axis=1)
    out["best_peer_mae"] = wide[peers].min(axis=1)
    out["attn_minus_peer"] = out["attn_mae"] - out["best_peer_mae"]
    out["attn_wins"] = out["attn_minus_peer"] < 0
    return out


def _fmt(x, nd=3):
    return "nan" if pd.isna(x) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def _section_overall(df, out):
    out.append("## Overall (test season, pooled across seeds)\n")
    agg = _metric_by(df, ["position"], "mae")
    seeds = sorted(df["seed"].unique())
    out.append(f"Seeds: {seeds}.  Rows/seed: {len(df) // max(len(seeds), 1)}.\n")
    for pos in list(df["position"].unique()) + ["ALL"]:
        sub = df if pos == "ALL" else df[df["position"] == pos]
        a = _metric_by(sub, [], "mae") if pos == "ALL" else agg[agg["position"] == pos]
        if pos == "ALL":
            a = a.assign(position="ALL")
        wide = a.pivot_table(index="position", columns="model", values="mean")
        line = "  ".join(f"{m}={_fmt(wide[m].iloc[0])}" for m in MODELS if m in wide.columns)
        out.append(f"- **{pos}**: {line}")
    out.append("")


def _section_weekly(df, out, figdir):
    out.append("## Week-by-week MAE (ATTN vs best peer)\n")
    agg = _metric_by(df, ["week"], "mae")
    cmp = _attn_vs_best_peer(agg, "week")
    out.append("| week | ATTN MAE | best peer | peer MAE | ATTN-peer | ATTN wins |")
    out.append("|---|---|---|---|---|---|")
    for wk, r in cmp.iterrows():
        out.append(
            f"| {wk} | {_fmt(r.attn_mae)} | {r.best_peer} | {_fmt(r.best_peer_mae)} "
            f"| {_fmt(r.attn_minus_peer)} | {'YES' if r.attn_wins else 'no'} |"
        )
    wins = int(cmp["attn_wins"].sum())
    out.append(f"\nATTN beats its best peer in **{wins}/{len(cmp)}** test weeks.\n")

    # Per-week signed bias (over/under prediction drift through the season).
    bias = _metric_by(df, ["week"], "bias")
    bwide = _pivot(bias, "week", "mean")
    out.append("### Per-week signed bias (mean pred - actual; + = over-predict)\n")
    out.append("| week | " + " | ".join(m for m in MODELS if m in bwide.columns) + " |")
    out.append("|---|" + "---|" * sum(m in bwide.columns for m in MODELS))
    for wk in bwide.index:
        cells = " | ".join(_fmt(bwide.loc[wk, m]) for m in MODELS if m in bwide.columns)
        out.append(f"| {wk} | {cells} |")
    out.append("")
    if figdir:
        _plot_weekly(agg, bias, figdir, out)


def _section_score_tier(df, out):
    out.append("## Subgroup: actual-score tier (judged by BIAS)\n")
    out.append(
        "On skewed FP targets, low-scoring slices have lower MAE regardless of "
        "model quality, so the honest 'worse on X?' signal is *bias* and "
        "bias-corrected MAE, not raw MAE.\n"
    )
    # Per-position quantile tiers so a QB's 'high' isn't a K's 'high'.
    df = df.copy()
    df["score_tier"] = (
        df.groupby(["position", "seed"], observed=True)[ACTUAL]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]))
        .astype(str)
    )
    bias = _metric_by(df, ["score_tier"], "bias")
    bwide = _pivot(bias, "score_tier", "mean")
    out.append("Signed bias by quartile (Q1=lowest actual, Q4=highest):\n")
    out.append("| tier | " + " | ".join(m for m in MODELS if m in bwide.columns) + " |")
    out.append("|---|" + "---|" * sum(m in bwide.columns for m in MODELS))
    for tier in ["Q1", "Q2", "Q3", "Q4"]:
        if tier in bwide.index:
            cells = " | ".join(_fmt(bwide.loc[tier, m]) for m in MODELS if m in bwide.columns)
            out.append(f"| {tier} | {cells} |")
    out.append("")


def _section_week_phase(df, out):
    out.append("## Subgroup: season phase\n")
    df = df.copy()
    df["phase"] = np.where(df["week"] <= 4, EARLY, np.where(df["week"] <= 13, MID, LATE))
    agg = _metric_by(df, ["position", "phase"], "mae")
    cmp = _attn_vs_best_peer(agg, ["position", "phase"])
    out.append("| position | phase | ATTN MAE | best peer | peer MAE | ATTN-peer |")
    out.append("|---|---|---|---|---|---|")
    for (pos, ph), r in cmp.iterrows():
        out.append(
            f"| {pos} | {ph} | {_fmt(r.attn_mae)} | {r.best_peer} "
            f"| {_fmt(r.best_peer_mae)} | {_fmt(r.attn_minus_peer)} |"
        )
    out.append("")


def _section_attn_position(df, out):
    out.append("## ATTN head-to-head by position\n")
    agg = _metric_by(df, ["position"], "mae")
    cmp = _attn_vs_best_peer(agg, "position")
    out.append("| position | ATTN MAE | best peer | peer MAE | ATTN-peer | ATTN wins |")
    out.append("|---|---|---|---|---|---|")
    for pos, r in cmp.iterrows():
        out.append(
            f"| {pos} | {_fmt(r.attn_mae)} | {r.best_peer} | {_fmt(r.best_peer_mae)} "
            f"| {_fmt(r.attn_minus_peer)} | {'YES' if r.attn_wins else 'no'} |"
        )
    out.append("")


def _plot_weekly(agg_mae, agg_bias, figdir, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(figdir).mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    mwide = _pivot(agg_mae, "week", "mean")
    bwide = _pivot(agg_bias, "week", "mean")
    for m in MODELS:
        if m in mwide.columns:
            lw = 2.5 if m == ATTN else 1.2
            ax1.plot(mwide.index, mwide[m], marker="o", ms=3, lw=lw, label=m)
            ax2.plot(bwide.index, bwide[m], marker="o", ms=3, lw=lw, label=m)
    ax1.set_title("Week-by-week MAE")
    ax1.set_xlabel("week")
    ax1.set_ylabel("MAE (fantasy pts)")
    ax1.legend()
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_title("Week-by-week signed bias")
    ax2.set_xlabel("week")
    ax2.set_ylabel("mean(pred - actual)")
    ax2.legend()
    fig.tight_layout()
    path = Path(figdir) / "attn_weekly_accuracy.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    out.append(f"![weekly]({path})\n")
    print(f"  figure -> {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positions", nargs="+", default=POSITIONS)
    p.add_argument("--seeds", default="42,123", help="comma-separated seeds")
    p.add_argument("--cache", default=None, help="dir to cache/reuse per-run test_dfs")
    p.add_argument("--report", default=None, help="write markdown report here")
    p.add_argument("--figdir", default=None, help="dir for the weekly figure")
    args = p.parse_args()

    positions = [x.upper() for x in args.positions]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    print(f"Positions: {positions}  Seeds: {seeds}")

    df = collect(positions, seeds, args.cache)
    print(f"Collected {len(df)} prediction rows across {df['position'].nunique()} positions.")

    out: list[str] = [
        "# Week-by-week & subgroup accuracy — Attention NN focus\n",
        f"Positions: {positions}.  Seeds: {seeds}.  "
        f"Test rows/seed: {len(df) // max(len(seeds), 1)}.\n",
    ]
    _section_overall(df, out)
    _section_attn_position(df, out)
    _section_weekly(df, out, args.figdir)
    _section_score_tier(df, out)
    _section_week_phase(df, out)

    text = "\n".join(out)
    print("\n" + text)
    if args.report:
        Path(args.report).write_text(text)
        print(f"\nReport written -> {args.report}")


if __name__ == "__main__":
    main()
