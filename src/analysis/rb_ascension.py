"""How does the RB model handle backups ascending into a workhorse role?

When a starter goes down, a low-usage back can jump from ~5 to ~22 touches in a
single week. Every RB volume/role feature is a ``shift=1`` rolling aggregate
(``src/shared/feature_build.py::rolling_agg``) or the prior-game attention
sequence, so on the ascension week itself the model's inputs *still encode
backup usage* — it cannot anticipate the breakout, only react to it. The one
forward-looking signal, ``depth_chart_rank`` (current-week depth charts publish
pre-game), is weak (median rank 2 even on ascension weeks) and is **entirely
absent for the 2025 test season** (``src/shared/feature_build.py`` imputes the
``-1`` sentinel to the train mean).

This operator-only diagnostic quantifies that. Two modes:

* **default (data-only):** pull nflverse weekly + injuries + depth charts, build
  an ascension cohort, and print (1) the LAG GAP — what the lagged inputs encode
  vs realized output on week W; (2) CONVERGENCE — how fast the lagged features
  catch up over W..W+3; (3) depth-chart coverage; (4) concrete examples. No
  model, no splits, no torch.
* **``--with-model-error``:** additionally run the RB pipeline and report literal
  per-model (Ridge / NN / Attention NN / LightGBM) MAE and signed bias on the
  ascension vs established cohorts, reusing ``src.shared.error_analysis``.

The lag gap is an *upper bound* on how well any model anchored on these inputs
can score the ascension week — it isolates the information content of the
features, which is why the data-only mode is meaningful without a pipeline run.

The pipeline (``run()``) is imported lazily inside ``main()`` so importing this
module never fires training/torch. The pure helpers are unit-tested in
``tests/analysis/test_rb_ascension.py``.

Usage:
    python -m src.analysis.rb_ascension
    python -m src.analysis.rb_ascension --with-model-error
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import SEASONS, TEST_SEASONS  # noqa: E402
from src.rb.data import compute_team_rb_totals  # noqa: E402
from src.shared.error_analysis import compute_stratum_metrics  # noqa: E402
from src.shared.feature_build import rolling_agg, safe_divide  # noqa: E402

# --- Cohort definition (touches = carries + targets, i.e. "opportunities") ---
BACKUP_OPP = 8.0  # mean opportunities/game over the prior 3 games to be a "backup"
WORKHORSE_OPP = 18.0  # opportunities in week W to count as a "workhorse" game
MIN_PRIOR_GAMES = 2  # need >=2 of the prior 3 weeks played to establish a baseline

# Per-model total-FP prediction columns written by src/shared/pipeline.py.
MODELS = {
    "Ridge": "pred_ridge_total",
    "NN": "pred_nn_total",
    "Attention NN": "pred_attn_nn_total",
    "LightGBM": "pred_lgbm_total",
}
ACTUAL = "fantasy_points"  # pipeline truth column (pred_*_total is compared to it)

# role_change stratum bucket labels.
ASCENSION = "ascension"
ESTABLISHED = "established"
UNKNOWN = "unknown"

# Columns label_ascension_rows() needs on a pipeline test_df. The L3 rolling
# means are model features (always present); carries/targets are raw weekly
# columns the RB feature code and attention history both read, so they survive
# onto test_df even though they are not themselves whitelisted features.
_PRIOR3_COLS = ["rolling_mean_carries_L3", "rolling_mean_targets_L3"]
_REALIZED_COLS = ["carries", "targets"]

_GRP = ["player_id", "season"]


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def prepare_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular-season RB rows and attach the shifted prior-game
    features the model keys on (computed with the exact pipeline convention:
    ``rolling_agg`` defaults to ``shift=1`` so week W sees only weeks < W).

    Adds: ``opp`` (carries+targets), ``prior3_opp``, ``prior3_fp``,
    ``prior3_fp_ppr``, ``prior3_carries``, ``lastwk_fp``, ``prior3_games``,
    ``carry_share_l3`` (player / team-RB carries, both shifted-L3 sums — i.e.
    ``team_rb_carry_share_L3``), and ``game_carry_share`` (realized, week W).
    """
    df = weekly[(weekly["position"] == "RB") & (weekly["season_type"] == "REG")].copy()
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    df["opp"] = df["carries"].fillna(0) + df["targets"].fillna(0)

    totals = compute_team_rb_totals(df)
    df = df.merge(totals, on=["recent_team", "season", "week"], how="left")

    df["prior3_opp"] = rolling_agg(df, "opp", _GRP, 3, agg="mean")
    df["prior3_fp"] = rolling_agg(df, "fantasy_points", _GRP, 3, agg="mean")
    df["prior3_fp_ppr"] = rolling_agg(df, "fantasy_points_ppr", _GRP, 3, agg="mean")
    df["prior3_carries"] = rolling_agg(df, "carries", _GRP, 3, agg="mean")
    df["lastwk_fp"] = rolling_agg(df, "fantasy_points", _GRP, 1, agg="mean")
    df["prior3_games"] = rolling_agg(df, "opp", _GRP, 3, agg="count").fillna(0)

    p_car = rolling_agg(df, "carries", _GRP, 3, agg="sum")
    t_car = rolling_agg(df, "team_rb_carries", _GRP, 3, agg="sum")
    df["carry_share_l3"] = safe_divide(p_car, t_car)
    df["game_carry_share"] = safe_divide(df["carries"], df["team_rb_carries"])
    return df


def find_ascension_events(
    prepared: pd.DataFrame,
    *,
    backup_opp: float = BACKUP_OPP,
    workhorse_opp: float = WORKHORSE_OPP,
    min_prior_games: int = MIN_PRIOR_GAMES,
) -> pd.DataFrame:
    """Rows where a prior backup (<= ``backup_opp`` opp/g over the prior 3 games)
    posts a workhorse week (>= ``workhorse_opp`` opportunities at week W)."""
    mask = (
        (prepared["prior3_games"] >= min_prior_games)
        & (prepared["prior3_opp"] <= backup_opp)
        & (prepared["opp"] >= workhorse_opp)
    )
    return prepared[mask].copy()


def add_injury_attribution(events: pd.DataFrame, prepared: pd.DataFrame, inj_out: set) -> pd.Series:
    """Boolean Series: was the team's prior lead RB Out/Doubtful or inactive at
    week W? ``inj_out`` is a set of ``(season, week, gsis_id)`` for
    Out/Doubtful report statuses. Approximates "ascended *because* a starter
    went down" — a lead back existed (>=8 prior carries) and is gone at W.
    """

    def _lead_back_down(row: pd.Series) -> bool:
        s, w, t, pid = row["season"], row["week"], row["recent_team"], row["player_id"]
        prior = prepared[
            (prepared["recent_team"] == t)
            & (prepared["season"] == s)
            & (prepared["week"].between(w - 3, w - 1))
            & (prepared["player_id"] != pid)
        ]
        if prior.empty:
            return False
        carries_by_back = prior.groupby("player_id")["carries"].sum()
        lead = carries_by_back.idxmax()
        if carries_by_back.loc[lead] < 8:  # no real lead back existed
            return False
        at_w = prepared[
            (prepared["player_id"] == lead) & (prepared["season"] == s) & (prepared["week"] == w)
        ]
        absent = at_w.empty or at_w["opp"].iloc[0] <= 2
        return bool(absent or (s, w, lead) in inj_out)

    if events.empty:
        return pd.Series([], dtype=bool)
    return events.apply(_lead_back_down, axis=1)


def label_ascension_rows(
    df: pd.DataFrame,
    *,
    backup_opp: float = BACKUP_OPP,
    workhorse_opp: float = WORKHORSE_OPP,
) -> pd.Series:
    """Label each pipeline ``test_df`` row ``ascension`` / ``established``.

    Uses the lagged L3 means (prior-3g opportunities) + realized week-W
    carries/targets already on ``test_df``. Degrades to ``unknown`` for every
    row if a required column is missing (mirrors
    ``error_analysis.add_stratification_columns``' defensive style) so this can
    be dropped into any position's stratum list without crashing.
    """
    needed = _PRIOR3_COLS + _REALIZED_COLS
    if any(c not in df.columns for c in needed):
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)
    prior3_opp = df["rolling_mean_carries_L3"].fillna(0) + df["rolling_mean_targets_L3"].fillna(0)
    realized_opp = df["carries"].fillna(0) + df["targets"].fillna(0)
    is_asc = (prior3_opp <= backup_opp) & (realized_opp >= workhorse_opp)
    return pd.Series(np.where(is_asc, ASCENSION, ESTABLISHED), index=df.index, dtype=object)


def convergence_table(
    events: pd.DataFrame, prepared: pd.DataFrame, max_offset: int = 3
) -> pd.DataFrame:
    """For offsets k=0..max_offset from each ascension week W, the mean realized
    FP vs the lagged-input FP and carry share — shows how fast (and whether it
    overshoots) the model's inputs catch up after the breakout.
    """
    keys = events[["player_id", "season", "week"]]
    rows = []
    for k in range(max_offset + 1):
        nxt = keys.copy()
        nxt["week"] = nxt["week"] + k
        m = nxt.merge(prepared, on=["player_id", "season", "week"], how="inner")
        rows.append(
            {
                "offset": f"W+{k}",
                "n": len(m),
                "realized_fp": m["fantasy_points"].mean(),
                "l3_input_fp": m["prior3_fp"].mean(),
                "carry_share_l3": m["carry_share_l3"].mean(),
                "realized_share": m["game_carry_share"].mean(),
            }
        )
    return pd.DataFrame(rows)


def cohort_model_table(df: pd.DataFrame, models: dict[str, str] | None = None) -> pd.DataFrame:
    """Per-model total-FP MAE / bias / n on the ``ascension`` vs ``established``
    buckets, via ``error_analysis.compute_stratum_metrics``. Expects a
    ``role_change`` column and the ``pred_*_total`` columns on ``df``.
    """
    models = models or {n: c for n, c in MODELS.items() if c in df.columns}
    out = []
    for name, col in models.items():
        metrics = compute_stratum_metrics(df, ACTUAL, col, "role_change")
        for _, r in metrics.iterrows():
            out.append(
                {
                    "model": name,
                    "bucket": str(r["role_change"]),
                    "n": int(r["n"]),
                    "mae": r["mae"],
                    "bias": r["bias"],
                }
            )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Presentation (IO; not unit-tested)
# --------------------------------------------------------------------------- #
def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _print_lag_gap(events: pd.DataFrame) -> None:
    _hr("LAG GAP at the ascension week (what the inputs encode vs realized)")
    g = events
    pairs = [
        ("Realized opportunities (car+tgt) @ W", g["opp"].mean()),
        ("Prior-3g opportunities (model input)", g["prior3_opp"].mean()),
        ("Realized game carry share @ W", g["game_carry_share"].mean()),
        ("carry_share_L3 (model input)", g["carry_share_l3"].mean()),
        (None, None),
        ("Realized fantasy pts @ W (std)", g["fantasy_points"].mean()),
        ("  L3-mean baseline (volume-anchored)", g["prior3_fp"].mean()),
        ("  Last-week baseline", g["lastwk_fp"].mean()),
        ("Realized fantasy pts @ W (PPR)", g["fantasy_points_ppr"].mean()),
        ("  L3-mean baseline (PPR)", g["prior3_fp_ppr"].mean()),
    ]
    for label, val in pairs:
        print("" if label is None else f"  {label:<42} {val:6.2f}")
    gap = (g["fantasy_points"] - g["prior3_fp"]).mean()
    gap_ppr = (g["fantasy_points_ppr"] - g["prior3_fp_ppr"]).mean()
    realized = g["fantasy_points"].mean()
    pct = gap / realized if realized else float("nan")
    print(f"\n  => structural under-prediction (std): {gap:5.2f} FP/game ({pct:.0%} of realized)")
    print(f"  => structural under-prediction (PPR): {gap_ppr:5.2f} FP/game")


def _print_convergence(events: pd.DataFrame, prepared: pd.DataFrame) -> None:
    _hr("CONVERGENCE — lagged input vs realized, weeks W..W+3")
    tbl = convergence_table(events, prepared)
    print(
        f"  {'offset':<8}{'N':>5}{'realized FP':>14}{'L3-input FP':>14}"
        f"{'carry_sh L3':>13}{'realized sh':>13}"
    )
    for _, r in tbl.iterrows():
        print(
            f"  {r['offset']:<8}{int(r['n']):>5}{r['realized_fp']:>14.2f}"
            f"{r['l3_input_fp']:>14.2f}{r['carry_share_l3']:>13.2f}{r['realized_share']:>13.2f}"
        )


def _print_depth_coverage(events: pd.DataFrame) -> None:
    _hr("depth_chart_rank — the only forward-looking signal")
    from src.data import nfl_source

    dc = nfl_source.depth_charts(SEASONS)
    dc = dc[dc["formation"] == "Offense"].copy()
    dc["depth_team"] = pd.to_numeric(dc["depth_team"], errors="coerce")
    dc_rb = dc[dc["position"] == "RB"]
    agg = dc_rb.groupby(["gsis_id", "season", "week"]).agg(rank=("depth_team", "min")).reset_index()
    cov = dc_rb.groupby("season")["week"].count()
    print("  RB depth-chart rows per season (0 => forward signal absent):")
    for s in SEASONS:
        n = int(cov.get(s, 0))
        tag = (
            "  <- TEST: forward signal DEAD"
            if s in set(TEST_SEASONS) and n == 0
            else ("  <- TEST" if s in set(TEST_SEASONS) else "")
        )
        print(f"    {s}: {n:5d}{tag}")
    ek = events[["player_id", "season", "week"]].rename(columns={"player_id": "gsis_id"})
    caught = ek.merge(agg, on=["gsis_id", "season", "week"], how="left")
    covered = caught[~caught["season"].isin(set(TEST_SEASONS))]
    has_rank = covered["rank"].notna()
    if len(covered):
        print(
            f"\n  Of {len(covered)} non-test events: {has_rank.mean():.0%} had a depth-chart row."
        )
        if has_rank.any():
            r = covered.loc[has_rank, "rank"]
            print(
                f"    among those, {(r <= 2).mean():.0%} listed rank<=2 (chart 'caught' the "
                f"promotion), median rank {r.median():.0f} (>=2 ⇒ often still listed as backup)."
            )


def _print_examples(events: pd.DataFrame) -> None:
    _hr("CONCRETE EXAMPLES (largest input->output gaps)")
    name_col = "player_display_name" if "player_display_name" in events.columns else "player_name"
    ex = events.assign(gap=events["fantasy_points"] - events["prior3_fp"]).sort_values(
        "gap", ascending=False
    )
    cols = [
        name_col,
        "recent_team",
        "season",
        "week",
        "prior3_opp",
        "opp",
        "prior3_fp",
        "fantasy_points",
        "injury_linked",
    ]
    cols = [c for c in cols if c in ex.columns]
    show = (
        ex[cols]
        .head(12)
        .rename(
            columns={
                name_col: "player",
                "prior3_opp": "pr3_opp",
                "opp": "opp_W",
                "prior3_fp": "L3_fp",
                "fantasy_points": "real_fp",
                "injury_linked": "inj",
            }
        )
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string(index=False, float_format=lambda x: f"{x:5.1f}"))


def _print_model_error(df: pd.DataFrame) -> None:
    _hr("LITERAL PER-MODEL ERROR on ascension vs established (RB test set)")
    df = df.copy()
    df["role_change"] = label_ascension_rows(df)
    counts = df["role_change"].value_counts().to_dict()
    print(f"  cohort sizes: {counts}")
    if counts.get(ASCENSION, 0) == 0:
        print(
            "  No ascension rows in the test set under the current thresholds — nothing to score."
        )
        return
    tbl = cohort_model_table(df)
    print(f"\n  {'model':<14}{'bucket':<13}{'N':>5}{'MAE':>9}{'bias':>9}")
    for _, r in tbl.sort_values(["model", "bucket"]).iterrows():
        print(
            f"  {r['model']:<14}{r['bucket']:<13}{int(r['n']):>5}{r['mae']:>9.3f}{r['bias']:>+9.3f}"
        )
    print(
        "\n  (bias = mean(pred - actual); large negative on the ascension bucket = "
        "systematic under-prediction of the breakout week.)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RB backup->workhorse ascension diagnostic")
    parser.add_argument(
        "--with-model-error",
        action="store_true",
        help="also run the RB pipeline and report per-model MAE/bias on the cohort",
    )
    args = parser.parse_args()

    from src.data import nfl_source  # deferred: pulls nflreadpy

    print("Loading nflverse weekly + injuries (2012-2025) ...", flush=True)
    weekly = nfl_source.weekly_data(SEASONS)
    prepared = prepare_weekly(weekly)
    events = find_ascension_events(prepared)

    inj = nfl_source.injuries(SEASONS)
    inj = inj[inj["report_status"].isin(["Out", "Doubtful"])]
    inj_out = set(zip(inj["season"], inj["week"], inj["gsis_id"], strict=False))
    events["injury_linked"] = add_injury_attribution(events, prepared, inj_out)

    _hr(
        f"RB ASCENSION COHORT (backup <= {BACKUP_OPP:g} prior3 opp/g  ->  "
        f">= {WORKHORSE_OPP:g} opp in week W)"
    )
    print(
        f"Total events {SEASONS[0]}-{SEASONS[-1]}: {len(events)}   "
        f"injury-linked (lead back Out/inactive): {events['injury_linked'].mean():.0%}"
    )
    per_season = events.groupby("season").size()
    print("\nEvents per season:")
    for s in SEASONS:
        tag = "  <- TEST" if s in set(TEST_SEASONS) else ""
        print(f"  {s}: {int(per_season.get(s, 0)):3d}{tag}")

    _print_lag_gap(events)
    _print_convergence(events, prepared)
    _print_depth_coverage(events)
    _print_examples(events)

    if args.with_model_error:
        from src.rb.run_pipeline import run  # deferred: pulls the full pipeline/torch

        print("\nRunning RB pipeline for literal per-model error ...", flush=True)
        result = run()
        _print_model_error(result["test_df"])


if __name__ == "__main__":
    main()
