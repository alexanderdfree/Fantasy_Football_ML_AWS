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

# Expert projection sources (opt-in via --experts). NFL.com covers QB/RB/WR/TE/K
# (no DST); Sleeper covers QB/RB/WR/TE/DST (no K). Both are re-scored to the same
# PPR fantasy_points the models target. Columns are attached by attach_experts().
EXPERTS = {
    "NFL.com": "pred_nflcom_total",
    "Sleeper": "pred_sleeper_total",
}
MODELS_AND_EXPERTS = {**MODELS, **EXPERTS}
# Offense skill positions are the only ones BOTH experts project, so they are the
# fair common ground for a single all-lines-comparable weekly chart.
OFFENSE = ["QB", "RB", "WR", "TE"]

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
# Expert projections (NFL.com + Sleeper), opt-in
# --------------------------------------------------------------------------- #
def attach_experts(df: pd.DataFrame, scoring_format: str = "ppr") -> pd.DataFrame:
    """Left-merge NFL.com + Sleeper projected totals (same PPR scoring) onto df.

    Reuses the project repo's expert loaders/projectors so the projections land
    in the exact fantasy_points scale the models target, and inherit their
    placeholder filtering (unprojected roster rows are already dropped, so a
    non-null expert column means a genuine projection). Adds ``pred_nflcom_total``
    and ``pred_sleeper_total``; rows a source does not project stay NaN.
    """
    from src.analysis.analysis_expert_comparison import (
        _project_nflcom_expert,
        _project_sleeper_to_ppr,
    )
    from src.analysis.sleeper_loader import load_sleeper_with_gsis_id
    from src.data.nflcom_loader import load_nflcom_with_gsis_id

    seasons = sorted(int(s) for s in df["season"].unique())
    print(f"  loading expert projections for seasons {seasons} ...", flush=True)
    nfl_raw = load_nflcom_with_gsis_id(seasons=seasons)
    sleeper_raw = load_sleeper_with_gsis_id(seasons=seasons)

    def _projected(raw, project_fn, out_col):
        parts = []
        for pos in df["position"].unique():
            proj = project_fn(raw, pos, scoring_format)
            if proj is None or proj.empty:
                continue
            parts.append(proj.rename(columns={"expert_pred_total": out_col}))
        if not parts:
            return pd.DataFrame(columns=["player_id", "season", "week", out_col])
        out = pd.concat(parts, ignore_index=True)[["player_id", "season", "week", out_col]]
        out["player_id"] = out["player_id"].astype(str)
        out["season"] = out["season"].astype(int)
        out["week"] = out["week"].astype(int)
        return out.drop_duplicates(["player_id", "season", "week"])

    df = df.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    nfl = _projected(nfl_raw, _project_nflcom_expert, EXPERTS["NFL.com"])
    sleeper = _projected(sleeper_raw, _project_sleeper_to_ppr, EXPERTS["Sleeper"])
    df = df.merge(nfl, on=["player_id", "season", "week"], how="left")
    df = df.merge(sleeper, on=["player_id", "season", "week"], how="left")
    n_nfl = int(df[EXPERTS["NFL.com"]].notna().sum())
    n_sl = int(df[EXPERTS["Sleeper"]].notna().sum())
    print(f"  matched: NFL.com {n_nfl}/{len(df)} rows, Sleeper {n_sl}/{len(df)} rows")
    return df


# --------------------------------------------------------------------------- #
# Aggregation helpers (seed-aware)
# --------------------------------------------------------------------------- #
def _models_in(df: pd.DataFrame, models: dict[str, str] | None = None) -> dict[str, str]:
    return available_models(df, models or MODELS)


def _metric_by(
    df: pd.DataFrame, by, metric: str, models: dict[str, str] | None = None
) -> pd.DataFrame:
    """Per-seed per-model `metric` grouped by `by`, then mean+/-std over seeds.

    Returns a tidy frame: columns [*by, model, mean, std, n].
    """
    models = _models_in(df, models)
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


def _plot_weekly(
    agg_mae, agg_bias, figdir, out, *, models=None, fname="attn_weekly_accuracy.png", title=""
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = models or MODELS
    Path(figdir).mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    mwide = _pivot(agg_mae, "week", "mean")
    bwide = _pivot(agg_bias, "week", "mean")
    for m in models:
        if m in mwide.columns:
            # Experts dashed, ATTN bold, other models thin solid.
            is_expert = m in EXPERTS
            lw = 2.5 if m == ATTN else (2.0 if is_expert else 1.2)
            ls = "--" if is_expert else "-"
            ax1.plot(mwide.index, mwide[m], marker="o", ms=3, lw=lw, ls=ls, label=m)
            ax2.plot(bwide.index, bwide[m], marker="o", ms=3, lw=lw, ls=ls, label=m)
    suffix = f" — {title}" if title else ""
    ax1.set_title(f"Week-by-week MAE{suffix}")
    ax1.set_xlabel("week")
    ax1.set_ylabel("MAE (fantasy pts)")
    ax1.legend()
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_title(f"Week-by-week signed bias{suffix}")
    ax2.set_xlabel("week")
    ax2.set_ylabel("mean(pred - actual)")
    ax2.legend()
    fig.tight_layout()
    path = Path(figdir) / fname
    fig.savefig(path, dpi=140)
    plt.close(fig)
    out.append(f"![weekly]({path})\n")
    print(f"  figure -> {path}")


def _section_experts_weekly(df, out, figdir):
    """Week-by-week chart of the 4 models + NFL.com + Sleeper on the common subset.

    Experts only project startable players, so MAE here runs higher than the
    full-population numbers above; every line is computed on the identical
    matched subset (offense skill positions, rows BOTH experts project) so the
    six lines are directly comparable.
    """
    out.append("## Week-by-week: 4 models + NFL.com + Sleeper (matched subset)\n")
    n_col, s_col = EXPERTS["NFL.com"], EXPERTS["Sleeper"]
    if n_col not in df.columns or s_col not in df.columns:
        out.append("_Expert columns absent — run with `--experts`._\n")
        return
    sub = df[df["position"].isin(OFFENSE)].dropna(subset=[n_col, s_col]).copy()
    if sub.empty:
        out.append("_No player-weeks with both NFL.com and Sleeper projections._\n")
        return
    out.append(
        f"Common ground = offense skill positions {OFFENSE} where both experts "
        f"project (NFL.com has no DST, Sleeper has no K). "
        f"Matched rows/seed: {len(sub) // sub['seed'].nunique()}. "
        "MAE is higher than the full-population sections because experts only "
        "cover high-volume startable players.\n"
    )
    agg = _metric_by(sub, ["week"], "mae", MODELS_AND_EXPERTS)
    bias = _metric_by(sub, ["week"], "bias", MODELS_AND_EXPERTS)
    mwide = _pivot(agg, "week", "mean")
    order = [m for m in MODELS_AND_EXPERTS if m in mwide.columns]
    out.append("### Per-week MAE\n")
    out.append("| week | " + " | ".join(order) + " | best |")
    out.append("|---|" + "---|" * (len(order) + 1))
    for wk in mwide.index:
        cells = " | ".join(_fmt(mwide.loc[wk, m]) for m in order)
        best = min(order, key=lambda m: mwide.loc[wk, m])
        out.append(f"| {wk} | {cells} | {best} |")
    # Season-long matched-subset MAE per source.
    overall = _metric_by(sub, [], "mae", MODELS_AND_EXPERTS).set_index("model")["mean"]
    out.append("\n### Season-long MAE on the matched subset (lower = better)\n")
    ranked = overall.sort_values()
    out.append("| rank | source | MAE |")
    out.append("|---|---|---|")
    for i, (name, v) in enumerate(ranked.items(), 1):
        tag = " *(expert)*" if name in EXPERTS else ""
        out.append(f"| {i} | {name}{tag} | {_fmt(v)} |")
    out.append("")
    if figdir:
        _plot_weekly(
            agg,
            bias,
            figdir,
            out,
            models=MODELS_AND_EXPERTS,
            fname="weekly_models_vs_experts.png",
            title="models vs NFL.com & Sleeper (matched offense subset)",
        )


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
    p.add_argument(
        "--experts",
        action="store_true",
        help="also overlay NFL.com + Sleeper projections (needs network on first fetch)",
    )
    p.add_argument(
        "--scoring", default="ppr", help="scoring format for experts (ppr/half_ppr/standard)"
    )
    args = p.parse_args()

    positions = [x.upper() for x in args.positions]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    print(f"Positions: {positions}  Seeds: {seeds}")

    df = collect(positions, seeds, args.cache)
    print(f"Collected {len(df)} prediction rows across {df['position'].nunique()} positions.")
    if args.experts:
        df = attach_experts(df, args.scoring)

    out: list[str] = [
        "# Week-by-week & subgroup accuracy — Attention NN focus\n",
        f"Positions: {positions}.  Seeds: {seeds}.  "
        f"Test rows/seed: {len(df) // max(len(seeds), 1)}.\n",
    ]
    _section_overall(df, out)
    _section_attn_position(df, out)
    _section_weekly(df, out, args.figdir)
    if args.experts:
        _section_experts_weekly(df, out, args.figdir)
    _section_score_tier(df, out)
    _section_week_phase(df, out)

    text = "\n".join(out)
    print("\n" + text)
    if args.report:
        Path(args.report).write_text(text)
        print(f"\nReport written -> {args.report}")


if __name__ == "__main__":
    main()
