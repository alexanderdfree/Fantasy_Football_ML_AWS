"""Unified cohort/subgroup diagnostics for fantasy-football model errors.

This is the consolidated home for read-only cohort analysis. It replaces the
older one-off CLIs for RB ascension, late-week effects, RB sparse-history/LGBM
disagreement, rookie metrics, and injury/return metrics while keeping their
tested pure helpers available for compatibility wrappers.

The rookie and injury-return cohorts are labels for error analysis only. They do
not add model features and do not revisit the rejected draft-capital feature.
Pipeline imports stay lazy so importing this module does not train models or
pull torch.

Usage:
    python -m src.analysis.cohort_analysis
    python -m src.analysis.cohort_analysis rookie --no-model
    python -m src.analysis.cohort_analysis ascension --positions RB --with-model-error
    python -m src.analysis.cohort_analysis sparse_history --positions RB --with-model-error --deep-dive
    python -m src.analysis.cohort_analysis scoring_tier --with-model-error --tier-topn 24 --deep-dive
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import POSITIONS, SCORING_PPR, SEASONS, SPLITS_DIR, TEST_SEASONS  # noqa: E402
from src.rb.data import compute_team_rb_totals  # noqa: E402
from src.shared.error_analysis import compute_stratum_metrics  # noqa: E402
from src.shared.evaluation import compute_metrics  # noqa: E402
from src.shared.feature_build import rolling_agg, safe_divide  # noqa: E402

ACTUAL = "fantasy_points"
TRUE_COL = ACTUAL
MODELS = {
    "Ridge": "pred_ridge_total",
    "NN": "pred_nn_total",
    "Attention NN": "pred_attn_nn_total",
    "LightGBM": "pred_lgbm_total",
}
LGBM = "LightGBM"
PEERS = ["Ridge", "NN", "Attention NN"]

UNKNOWN = "unknown"

# Ascension constants.
BACKUP_OPP = 8.0
WORKHORSE_OPP = 18.0
MIN_PRIOR_GAMES = 2
ASCENSION = "ascension"
ESTABLISHED = "established"
ROLE_CHANGE = "role_change"
_PRIOR3_COLS = ["rolling_mean_carries_L3", "rolling_mean_targets_L3"]
_REALIZED_COLS = ["carries", "targets"]
_GRP = ["player_id", "season"]

# Late-week constants.
EARLY, PENULT, FINAL = "early", "penult", "final"
BUCKET_ORDER = [EARLY, PENULT, FINAL]
DEFAULT_ESTABLISHED_GAMES = 8
DEFAULT_TOP_K = 12
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

# Rookie constants.
ROOKIE_EARLY = "rookie_early"
ROOKIE_REST = "rookie_rest"
VETERAN = "veteran"
ROOKIE_BUCKET = "rookie_bucket"
EARLY_GAMES = 3
DEFAULT_ROOKIE_POSITIONS = ["QB", "RB", "WR", "TE"]

# Sparse-history / LGBM-disagreement constants.
DISAGREEMENT_THRESHOLD = 4.0
CALIB_BINS = [0, 5, 10, 15, 20, 25, np.inf]
HISTORY_DEPTH_BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 99, "8+")]
HOT_RECENT_FP = 18.0
RECENT_MAX_COL = "rolling_max_fantasy_points_L3"
HISTORY_BUCKET = "history_bucket"

# Injury-return constants.
ANALYSIS_NAME = "cohort_analysis"
HISTORY_DIR = os.path.join("benchmark_history", "analysis")
DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
SMALL_N = 40
RETURN_FLAG = "is_returning_from_absence"
DAYS_REST = "days_rest"
GAME_STATUS = "game_status"
INJURY_RETURN_BUCKET = "injury_return_bucket"

# Scoring-tier (a-priori "commonly drafted" elite) constants. The tier is defined
# by prior-season mean fantasy points (an ADP-like, known-before-kickoff proxy)
# ranked within (season, position), NOT by points actually scored that week — so
# the slice is what you would have *drafted*, free of hindsight leakage.
TIER_ELITE = "elite_top_drafted"
TIER_FIELD = "field"
TIER_BUCKET = "tier_bucket"
DEFAULT_TIER_TOPN = 24
TIER_POSITIONS = ["QB", "RB", "WR", "TE"]

# New consolidated cohorts.
COMMITTEE = "committee"
NON_COMMITTEE = "non_committee"
TRADED = "midseason_trade"
STABLE_TEAM = "stable_team"
SUSPENSION_RETURN = "suspension_return"


@dataclass(frozen=True)
class CohortContext:
    min_season: pd.Series | None = None
    early_games: int = EARLY_GAMES
    # (player_id, season) -> prior-season mean fantasy points; the a-priori
    # scoring-tier proxy, precomputed from all split frames so a position's
    # test_df (test seasons only) can still see each player's S-1 expectation.
    prior_season_fp: pd.Series | None = None
    tier_topn: int = DEFAULT_TIER_TOPN


@dataclass(frozen=True)
class CohortSpec:
    name: str
    description: str
    positions: tuple[str, ...]
    label_col: str
    cohort_value: str
    label_fn: Callable[[pd.DataFrame, CohortContext], pd.Series]
    feasible: bool = True


SUBGROUP_SPECS: list[tuple[str, str, str | None, Callable[[pd.DataFrame], pd.Series]]] = [
    ("global", "GLOBAL (all rows)", None, lambda df: pd.Series(True, index=df.index)),
    (
        "settled",
        "settled - no week gap (days_rest=7)",
        RETURN_FLAG,
        lambda df: df[RETURN_FLAG] == 0,
    ),
    ("returning", "returning - any gap >1 wk", RETURN_FLAG, lambda df: df[RETURN_FLAG] == 1),
    (
        "ret_1wk",
        "returning, exactly 1 wk missed (days_rest=14)",
        DAYS_REST,
        lambda df: df[DAYS_REST] == 14,
    ),
    (
        "ret_2wk",
        "returning, 2+ wk missed (days_rest>14)",
        DAYS_REST,
        lambda df: df[DAYS_REST] > 14,
    ),
    ("healthy", "healthy (game_status=1.0)", GAME_STATUS, lambda df: df[GAME_STATUS] >= 1.0),
    (
        "questionable",
        "Questionable+ (game_status<1.0)",
        GAME_STATUS,
        lambda df: df[GAME_STATUS] < 1.0,
    ),
]


def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _flag(n: int) -> str:
    if n == 0:
        return "  [EMPTY - not measurable in test]"
    if n < SMALL_N:
        return f"  [small-n<{SMALL_N}: noisy, do not act on alone]"
    return ""


# --------------------------------------------------------------------------- #
# Shared model/error helpers
# --------------------------------------------------------------------------- #
def available_models(df: pd.DataFrame, models: dict[str, str] | None = None) -> dict[str, str]:
    """Subset of ``models`` whose prediction column is present in ``df``."""
    models = models or MODELS
    return {name: col for name, col in models.items() if col in df.columns}


def _prediction_columns(df: pd.DataFrame) -> dict[str, str]:
    """Dynamic prediction-column discovery with the historical short NN label."""
    try:
        from src.analysis.significance import pred_columns_from_test_df

        cols = pred_columns_from_test_df(df)
        if cols:
            return {"NN" if name == "Neural Net" else name: col for name, col in cols.items()}
    except Exception:
        pass
    return available_models(df)


def per_model_metrics(
    df: pd.DataFrame, models: dict[str, str] | None = None, actual: str = ACTUAL
) -> dict[str, dict[str, float]]:
    """MAE / signed bias / RMSE / n for each model on ``df``.

    Bias = mean(pred - actual): positive means over-prediction.
    """
    models = models or available_models(df)
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


def bucket_model_table(
    df: pd.DataFrame,
    bucket_col: str,
    models: dict[str, str] | None = None,
    *,
    actual: str = ACTUAL,
) -> pd.DataFrame:
    """Uniform per-model MAE/RMSE/bias/n by bucket plus dMAE vs global."""
    models = models or _prediction_columns(df)
    global_metrics = per_model_metrics(df, models, actual)
    out = []
    for name, col in models.items():
        for bucket, sub in df.groupby(bucket_col, observed=True, sort=True):
            m = per_model_metrics(sub, {name: col}, actual)[name]
            out.append(
                {
                    "model": name,
                    "bucket": str(bucket),
                    "n": int(m["n"]),
                    "mae": m["mae"],
                    "dmae": m["mae"] - global_metrics[name]["mae"],
                    "rmse": m["rmse"],
                    "bias": m["bias"],
                }
            )
    return pd.DataFrame(out)


def bias_corrected_mae(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str
) -> pd.Series:
    """Per-group mean(|error - mean(error)|), the bias-removed MAE."""
    tmp = df[[group_col, y_true_col, y_pred_col]].dropna().copy()
    tmp["_e"] = tmp[y_pred_col] - tmp[y_true_col]
    centered = tmp["_e"] - tmp.groupby(group_col, observed=True)["_e"].transform("mean")
    tmp["_abs_centered"] = centered.abs()
    return tmp.groupby(group_col, observed=True)["_abs_centered"].mean()


def best_model(df: pd.DataFrame, models: dict[str, str] | None = None) -> tuple[str | None, float]:
    """Lowest-overall-MAE model present on ``df``."""
    models = models or available_models(df)
    best_name, best_mae = None, float("inf")
    for name, col in models.items():
        mae = (df[col] - df[ACTUAL]).abs().mean()
        if mae < best_mae:
            best_name, best_mae = name, mae
    return best_name, (best_mae if best_name is not None else float("nan"))


# --------------------------------------------------------------------------- #
# Ascension helpers and deep dive
# --------------------------------------------------------------------------- #
def prepare_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular-season RB rows and attach shifted prior-game features."""
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
    """Rows where a prior backup posts a workhorse week."""
    mask = (
        (prepared["prior3_games"] >= min_prior_games)
        & (prepared["prior3_opp"] <= backup_opp)
        & (prepared["opp"] >= workhorse_opp)
    )
    return prepared[mask].copy()


def add_injury_attribution(events: pd.DataFrame, prepared: pd.DataFrame, inj_out: set) -> pd.Series:
    """Was the team's prior lead RB Out/Doubtful or inactive at week W?"""

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
        if carries_by_back.loc[lead] < 8:
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
    min_prior_games: int = MIN_PRIOR_GAMES,
) -> pd.Series:
    """Label each row ``ascension`` / ``established`` / ``unknown``.

    When player/season/week keys are present, require at least
    ``min_prior_games`` earlier in-season games so season openers with empty
    rolling history do not masquerade as backup-to-workhorse transitions.
    """
    needed = _PRIOR3_COLS + _REALIZED_COLS
    if any(c not in df.columns for c in needed):
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)
    prior3_opp = df["rolling_mean_carries_L3"].fillna(0) + df["rolling_mean_targets_L3"].fillna(0)
    realized_opp = df["carries"].fillna(0) + df["targets"].fillna(0)
    is_asc = (prior3_opp <= backup_opp) & (realized_opp >= workhorse_opp)
    prior_games = _prior_game_counts(df)
    if prior_games is not None:
        is_asc &= prior_games >= min_prior_games
    return pd.Series(np.where(is_asc, ASCENSION, ESTABLISHED), index=df.index, dtype=object)


def convergence_table(
    events: pd.DataFrame, prepared: pd.DataFrame, max_offset: int = 3
) -> pd.DataFrame:
    """Lagged-input convergence after each ascension week W."""
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


def ascension_cohort_model_table(
    df: pd.DataFrame, models: dict[str, str] | None = None
) -> pd.DataFrame:
    """Historical ascension-only MAE/bias table, expected by compatibility tests."""
    models = models or {n: c for n, c in MODELS.items() if c in df.columns}
    out = []
    for name, col in models.items():
        metrics = compute_stratum_metrics(df, ACTUAL, col, ROLE_CHANGE)
        for _, r in metrics.iterrows():
            out.append(
                {
                    "model": name,
                    "bucket": str(r[ROLE_CHANGE]),
                    "n": int(r["n"]),
                    "mae": r["mae"],
                    "bias": r["bias"],
                }
            )
    return pd.DataFrame(out)


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
    _hr("CONVERGENCE - lagged input vs realized, weeks W..W+3")
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


def _offense_depth_ranks(depth: pd.DataFrame) -> pd.DataFrame:
    """Collapse canonical depth rows to one offensive rank per player-week."""
    off = depth[depth["formation"] == "Offense"].copy()
    off["depth_team"] = pd.to_numeric(off["depth_team"], errors="coerce")
    return off.groupby(["gsis_id", "season", "week"]).agg(rank=("depth_team", "min")).reset_index()


def _depth_chart_ranks(seasons: list[int], schedules: pd.DataFrame) -> pd.DataFrame:
    """Load depth ranks through the same normalization path production uses."""
    from src.data import nfl_source
    from src.data.loader import _DEPTH_CANONICAL_COLS, _normalize_espn_depth

    old = [s for s in seasons if s <= 2024]
    new = [s for s in seasons if s >= 2025]
    parts = []
    if old:
        parts.append(nfl_source.depth_charts(old)[_DEPTH_CANONICAL_COLS])
    for s in new:
        url = (
            "https://github.com/nflverse/nflverse-data/releases/download/"
            f"depth_charts/depth_charts_{s}.parquet"
        )
        parts.append(_normalize_espn_depth(pd.read_parquet(url), schedules, s))
    return _offense_depth_ranks(pd.concat(parts, ignore_index=True))


def _print_depth_coverage(events: pd.DataFrame) -> None:
    _hr("depth_chart_rank - forward-looking signal (loaded as production does)")
    from src.data import nfl_source

    ranks = _depth_chart_ranks(SEASONS, nfl_source.schedules(SEASONS))
    cov = ranks.groupby("season").size()
    print("  Offensive player-weeks with a depth_chart_rank, per season (0 => absent):")
    for s in SEASONS:
        tag = "  <- TEST (ESPN-format via #370)" if s in set(TEST_SEASONS) else ""
        print(f"    {s}: {int(cov.get(s, 0)):6d}{tag}")
    ek = events[["player_id", "season", "week"]].rename(columns={"player_id": "gsis_id"})
    caught = ek.merge(ranks, on=["gsis_id", "season", "week"], how="left")
    has_rank = caught["rank"].notna()
    print(
        f"\n  Of {len(caught)} ascension events: {has_rank.mean():.0%} had a depth_chart_rank @ W"
    )
    if has_rank.any():
        r = caught.loc[has_rank, "rank"]
        print(
            f"    among those: {(r <= 2).mean():.0%} listed rank<=2, median rank {r.median():.0f}."
        )
        t = caught[caught["season"].isin(set(TEST_SEASONS))]
        if len(t):
            print(
                f"    2025 test season: {t['rank'].notna().mean():.0%} of {len(t)} events covered."
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


def ascension_data_diagnostic(*, deep_dive: bool = True) -> pd.DataFrame:
    """Run the data-only ascension diagnostic and return the event rows."""
    from src.data import nfl_source

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

    if deep_dive:
        _print_lag_gap(events)
        _print_convergence(events, prepared)
        _print_depth_coverage(events)
        _print_examples(events)
    return events


# --------------------------------------------------------------------------- #
# Rookie helpers
# --------------------------------------------------------------------------- #
def player_min_season(frames: list[pd.DataFrame | None]) -> pd.Series:
    """Map ``player_id`` to earliest season across usable split frames."""
    parts = [
        f[["player_id", "season"]]
        for f in frames
        if f is not None and "player_id" in f.columns and "season" in f.columns
    ]
    if not parts:
        return pd.Series(dtype="int64", name="season")
    return pd.concat(parts, ignore_index=True).groupby("player_id")["season"].min()


def player_prior_season_fp(frames: list[pd.DataFrame | None]) -> pd.Series:
    """Map ``(player_id, season)`` to the player's PRIOR-season mean fantasy points.

    This is the a-priori "commonly drafted" expectation proxy: it aggregates each
    player's season-(S-1) mean ``fantasy_points`` and attaches it to season S, so a
    test-season row can be ranked by what was known before the season started. It
    reconstructs the signal that ``prior_season_mean_fantasy_points`` carried before
    PR #191 dropped it as a model feature; here it is used only for slicing, never
    fed to a model.
    """
    needed = ["player_id", "season", "fantasy_points"]
    parts = [f[needed] for f in frames if f is not None and all(c in f.columns for c in needed)]
    if not parts:
        return pd.Series(dtype=float, name="prior_season_fp")
    allrows = pd.concat(parts, ignore_index=True)
    season_fp = allrows.groupby(["player_id", "season"], as_index=False)["fantasy_points"].mean()
    # Shift each season's mean forward one year: S-1's mean is S's prior expectation.
    season_fp["season"] = season_fp["season"] + 1
    out = season_fp.set_index(["player_id", "season"])["fantasy_points"]
    out.name = "prior_season_fp"
    return out


def label_scoring_tier_rows(
    df: pd.DataFrame,
    prior_season_fp: pd.Series | None,
    *,
    top_n: int = DEFAULT_TIER_TOPN,
) -> pd.Series:
    """Label rows ``elite_top_drafted`` / ``field`` / ``unknown`` by a-priori rank.

    A player-season is ``elite_top_drafted`` when its prior-season mean fantasy
    points ranks in the top ``top_n`` within its ``(season, position)``; players
    with a prior season but a lower rank are ``field``; rows with no prior-season
    expectation (rookies / first observed season) are ``unknown`` rather than being
    forced into ``field``, so the elite-vs-field contrast stays a veteran comparison.
    """
    needed = ["player_id", "season", "position"]
    if (
        prior_season_fp is None
        or len(prior_season_fp) == 0
        or any(c not in df.columns for c in needed)
    ):
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)

    keys = pd.MultiIndex.from_arrays([df["player_id"], df["season"]])
    proxy = pd.Series(prior_season_fp.reindex(keys).to_numpy(), index=df.index)

    labels = pd.Series(UNKNOWN, index=df.index, dtype=object)
    valid = proxy.notna()
    if not valid.any():
        return labels

    # Rank once per player-season (not per game) so a player's number of test rows
    # does not bias the cutoff; ascending=False puts the highest scorers at rank 1.
    ps = df.loc[valid, ["player_id", "season", "position"]].copy()
    ps["_proxy"] = proxy[valid]
    ps = ps.drop_duplicates(["player_id", "season"])
    ps["_rank"] = ps.groupby(["season", "position"])["_proxy"].rank(method="first", ascending=False)
    elite_keys = set(map(tuple, ps.loc[ps["_rank"] <= top_n, ["player_id", "season"]].to_numpy()))

    row_keys = list(zip(df["player_id"], df["season"], strict=True))
    is_elite = pd.Series([k in elite_keys for k in row_keys], index=df.index)
    labels[valid & ~is_elite] = TIER_FIELD
    labels[valid & is_elite] = TIER_ELITE
    return labels


def label_rookie_rows(
    test_df: pd.DataFrame,
    min_season: pd.Series,
    early_games: int = EARLY_GAMES,
) -> pd.Series:
    """Label rows ``rookie_early`` / ``rookie_rest`` / ``veteran``."""
    needed = ["player_id", "season", "week"]
    if any(c not in test_df.columns for c in needed) or min_season is None or len(min_season) == 0:
        return pd.Series([UNKNOWN] * len(test_df), index=test_df.index, dtype=object)

    debut = test_df["player_id"].map(min_season)
    is_rookie = (test_df["season"].eq(debut) & debut.notna()).to_numpy()

    tmp = test_df[_GRP + ["week"]].copy()
    tmp["_pos"] = np.arange(len(tmp))
    tmp = tmp.sort_values(_GRP + ["week"])
    tmp["_game_idx"] = tmp.groupby(_GRP).cumcount()
    game_idx = tmp.sort_values("_pos")["_game_idx"].to_numpy()

    out = np.where(
        ~is_rookie,
        VETERAN,
        np.where(game_idx < early_games, ROOKIE_EARLY, ROOKIE_REST),
    )
    return pd.Series(out, index=test_df.index, dtype=object)


def rookie_cohort_model_table(
    df: pd.DataFrame, models: dict[str, str] | None = None
) -> pd.DataFrame:
    """Rookie-specific table with bias-corrected MAE."""
    models = available_models(df, models)
    out = []
    for name, col in models.items():
        metrics = compute_stratum_metrics(df, ACTUAL, col, ROOKIE_BUCKET)
        corr = bias_corrected_mae(df, ACTUAL, col, ROOKIE_BUCKET)
        for _, r in metrics.iterrows():
            out.append(
                {
                    "model": name,
                    "bucket": str(r[ROOKIE_BUCKET]),
                    "n": int(r["n"]),
                    "mae": r["mae"],
                    "mae_corr": float(corr.get(r[ROOKIE_BUCKET], float("nan"))),
                    "rmse": r["rmse"],
                    "bias": r["bias"],
                }
            )
    return pd.DataFrame(out)


def rookie_gap(df: pd.DataFrame, model_col: str) -> tuple[float, float, float]:
    """``(rookie_mae, veteran_mae, rookie_early_mae)`` for one model column."""

    def _mae(mask: pd.Series) -> float:
        if not mask.any():
            return float("nan")
        return (df.loc[mask, model_col] - df.loc[mask, ACTUAL]).abs().mean()

    bucket = df[ROOKIE_BUCKET]
    return (
        _mae(bucket.isin([ROOKIE_EARLY, ROOKIE_REST])),
        _mae(bucket.eq(VETERAN)),
        _mae(bucket.eq(ROOKIE_EARLY)),
    )


def _print_cohort_sizes(
    test_df: pd.DataFrame, min_season: pd.Series, positions: list[str], early_games: int
) -> None:
    _hr(f"ROOKIE COHORT SIZES (test season {TEST_SEASONS}) - data-only, no model")
    test = test_df.copy()
    if "position" in test.columns:
        test = test[test["position"].isin(positions)].copy()
    test[ROOKIE_BUCKET] = label_rookie_rows(test, min_season, early_games)

    rmask = test[ROOKIE_BUCKET].isin([ROOKIE_EARLY, ROOKIE_REST])
    n, rk = len(test), int(rmask.sum())
    pct = rk / n if n else float("nan")
    print(f"  all {'/'.join(positions)}: {n} rows, rookies {rk} ({pct:.0%})  [expect ~14%]")

    print(f"\n  {'pos':<5}{'rows':>7}{'rookies':>9}{'%':>6}{'rookie FP':>12}{'vet FP':>9}")
    for pos in positions:
        sub = test[test["position"] == pos] if "position" in test.columns else test
        if not len(sub):
            continue
        sub_rmask = sub[ROOKIE_BUCKET].isin([ROOKIE_EARLY, ROOKIE_REST])
        srk = int(sub_rmask.sum())
        r_fp = sub.loc[sub_rmask, ACTUAL].mean() if srk and ACTUAL in sub.columns else float("nan")
        vmask = sub[ROOKIE_BUCKET].eq(VETERAN)
        v_fp = (
            sub.loc[vmask, ACTUAL].mean() if vmask.any() and ACTUAL in sub.columns else float("nan")
        )
        print(f"  {pos:<5}{len(sub):>7}{srk:>9}{(srk / len(sub)):>6.0%}{r_fp:>12.2f}{v_fp:>9.2f}")
    print("\n  (realized rookie FP should sit below veteran FP - rookies score less on average.)")


def _print_rookie_model_error(
    pos: str, test_df: pd.DataFrame, min_season: pd.Series, early_games: int
) -> None:
    _hr(f"{pos}: PER-MODEL ERROR on rookie vs veteran (test set)")
    df = test_df.copy()
    df[ROOKIE_BUCKET] = label_rookie_rows(df, min_season, early_games)
    counts = df[ROOKIE_BUCKET].value_counts().to_dict()
    print(f"  cohort sizes: {counts}")
    if counts.get(ROOKIE_EARLY, 0) + counts.get(ROOKIE_REST, 0) == 0:
        print("  No rookie rows in the test set - nothing to score.")
        return

    models = available_models(df)
    tbl = rookie_cohort_model_table(df, models)
    print(f"\n  {'model':<14}{'bucket':<13}{'N':>5}{'MAE':>8}{'MAEbc':>8}{'RMSE':>8}{'bias':>8}")
    for _, r in tbl.sort_values(["model", "bucket"]).iterrows():
        print(
            f"  {r['model']:<14}{r['bucket']:<13}{int(r['n']):>5}"
            f"{r['mae']:>8.3f}{r['mae_corr']:>8.3f}{r['rmse']:>8.3f}{r['bias']:>+8.3f}"
        )

    best_name, best_mae = best_model(df, models)
    if best_name is not None:
        r_mae, v_mae, e_mae = rookie_gap(df, models[best_name])
        e_corr = float(
            bias_corrected_mae(df, ACTUAL, models[best_name], ROOKIE_BUCKET).get(
                ROOKIE_EARLY, float("nan")
            )
        )
        print(
            f"\n  => best model = {best_name} (overall MAE {best_mae:.3f}); "
            f"rookie {r_mae:.3f} vs veteran {v_mae:.3f} (Delta {r_mae - v_mae:+.3f}); "
            f"rookie_early MAE {e_mae:.3f} -> {e_corr:.3f} bias-corrected "
            f"({e_mae - e_corr:+.3f} recoverable by calibration)."
        )


# --------------------------------------------------------------------------- #
# Late-week helpers
# --------------------------------------------------------------------------- #
def _load_splits(splits_dir: str | Path = SPLITS_DIR) -> list[pd.DataFrame]:
    """Load train/val/test split parquets, REG-only."""
    frames = []
    for name in ("train", "val", "test"):
        path = Path(splits_dir) / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing split: {path}")
        df = pd.read_parquet(path)
        if "season_type" in df.columns:
            df = df[df["season_type"] == "REG"].copy()
        frames.append(df)
    return frames


def _load_split_subset(splits_dir: str | Path, which: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for name in which:
        path = Path(splits_dir) / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing split: {path}")
        df = pd.read_parquet(path)
        if "season_type" in df.columns:
            df = df[df["season_type"] == "REG"].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def assign_final_week_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add season-aware final-week bucket columns."""
    df = df.copy()
    df["final_week"] = df.groupby("season")["week"].transform("max")
    df["week_bucket"] = np.where(
        df["week"] == df["final_week"],
        FINAL,
        np.where(df["week"] == df["final_week"] - 1, PENULT, EARLY),
    )
    df["era"] = np.where(df["season"] <= 2020, "2012-2020(17wk)", "2021-2025(18wk)")
    return df


def _ordered_positions(df: pd.DataFrame) -> list[str]:
    present = set(df["position"].dropna().unique())
    return [p for p in POSITIONS if p in present]


def _composition_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n  {title}")
    print(f"    {'pos':<5}{'bucket':<8}{'n':>7}{'mean':>9}{'median':>9}{'dMean':>9}{'dMean%':>8}")
    for pos in ["ALL", *_ordered_positions(df)]:
        sub = df if pos == "ALL" else df[df["position"] == pos]
        early_mean = sub.loc[sub["week_bucket"] == EARLY, TRUE_COL].mean()
        for bucket in BUCKET_ORDER:
            cell = sub[sub["week_bucket"] == bucket]
            if cell.empty:
                continue
            mean = cell[TRUE_COL].mean()
            median = cell[TRUE_COL].median()
            d = mean - early_mean
            dpct = 100.0 * d / early_mean if early_mean else float("nan")
            tag = "*" if bucket == FINAL else ""
            print(
                f"    {pos:<5}{bucket + tag:<8}{len(cell):>7}{mean:>9.2f}"
                f"{median:>9.2f}{d:>9.2f}{dpct:>7.1f}%"
            )


def _player_relative_drop(df: pd.DataFrame, established_games: int) -> pd.DataFrame:
    recs = []
    for (_pid, _season), g in df.groupby(["player_id", "season"], sort=False):
        if len(g) < established_games:
            continue
        fw = int(g["final_week"].iloc[0])
        early = g[g["week"] <= fw - 2]
        if early.empty:
            continue
        base = early[TRUE_COL].mean()
        pos = g["position"].iloc[0]
        fin = g[g["week"] == fw]
        if not fin.empty:
            recs.append((pos, FINAL, fin[TRUE_COL].mean() - base))
        pen = g[g["week"] == fw - 1]
        if not pen.empty:
            recs.append((pos, PENULT, pen[TRUE_COL].mean() - base))
    return pd.DataFrame(recs, columns=["position", "bucket", "drop"])


def _player_relative_table(drops: pd.DataFrame) -> None:
    print(
        "\n  Player-relative drop vs own early-week baseline "
        "(established players; negative = rested/declined)"
    )
    print(f"    {'pos':<5}{'bucket':<8}{'n_players':>10}{'meanDrop':>10}{'medianDrop':>12}")
    order = {b: i for i, b in enumerate(BUCKET_ORDER)}
    grouped = (
        drops.groupby(["position", "bucket"])["drop"].agg(["count", "mean", "median"]).reset_index()
    )
    grouped["pos_rank"] = grouped["position"].map(
        lambda p: POSITIONS.index(p) if p in POSITIONS else 99
    )
    grouped["b_rank"] = grouped["bucket"].map(order)
    for _, r in grouped.sort_values(["pos_rank", "b_rank"]).iterrows():
        print(
            f"    {r['position']:<5}{r['bucket']:<8}{int(r['count']):>10}"
            f"{r['mean']:>10.2f}{r['median']:>12.2f}"
        )


def _era_contrast_table(df: pd.DataFrame) -> None:
    print("\n  Era contrast: final-week mean minus early-week mean (per position)")
    eras = sorted(df["era"].unique())
    print(f"    {'pos':<5}" + "".join(f"{e:>20}" for e in eras))
    for pos in _ordered_positions(df):
        sub = df[df["position"] == pos]
        cells = []
        for era in eras:
            e = sub[sub["era"] == era]
            early = e.loc[e["week_bucket"] == EARLY, TRUE_COL].mean()
            fin = e.loc[e["week_bucket"] == FINAL, TRUE_COL].mean()
            cells.append(f"{(fin - early):>20.2f}" if not np.isnan(fin) else f"{'-':>20}")
        print(f"    {pos:<5}" + "".join(cells))


def stage1_label_anomaly(splits_dir: str | Path, established_games: int) -> None:
    print("=" * 78)
    print("STAGE 1 - raw-label anomaly in the season's final weeks (no models)")
    print("  'final' = season-aware last week (17 for 2012-2020, 18 for 2021+).")
    print("  Reminder: wk18 is not a fantasy week; wk17 is the fantasy championship.")
    print("  Scope: QB/RB/WR/TE only (raw-split fantasy_points is skill-only; K approximately 0,")
    print("  DST absent) - K/DST get correct totals in model-error mode.")
    print("=" * 78)

    all_df = assign_final_week_buckets(_load_split_subset(splits_dir, ("train", "val", "test")))
    all_df = all_df[all_df["position"].isin(SKILL_POSITIONS)].copy()
    fw_by_season = all_df.groupby("season")["final_week"].first().to_dict()
    print("\n  Sanity - final (max) week by season:")
    print("    " + ", ".join(f"{s}:{w}" for s, w in sorted(fw_by_season.items())))

    _composition_table(all_df, "Table A1 - composition, ALL seasons 2012-2025")

    test_df = all_df[all_df["season"] == all_df["season"].max()]
    _composition_table(
        test_df, f"Table A2 - composition, TEST season {int(test_df['season'].iloc[0])} only"
    )

    _era_contrast_table(all_df)

    drops = _player_relative_drop(all_df, established_games)
    _player_relative_table(drops)


def _week_bucket_eval(week: int) -> str:
    if week <= 16:
        return "wk1-16"
    return "wk17" if week == 17 else "wk18"


def _week_hit_spear(wdf: pd.DataFrame, pred_col: str, top_k: int) -> tuple[float, float] | None:
    from scipy.stats import spearmanr

    if len(wdf) < top_k:
        return None
    actual_top = set(wdf.nlargest(top_k, TRUE_COL)["player_id"])
    pred_top = set(wdf.nlargest(top_k, pred_col)["player_id"])
    hit = len(actual_top & pred_top) / top_k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = spearmanr(wdf[pred_col], wdf[TRUE_COL])
    return hit, corr


def _avg_weekly_ranking(df: pd.DataFrame, pred_col: str, top_k: int) -> tuple[float, float]:
    pairs = [hs for _w, wdf in df.groupby("week") if (hs := _week_hit_spear(wdf, pred_col, top_k))]
    if not pairs:
        return float("nan"), float("nan")
    corrs = [c for _, c in pairs if not np.isnan(c)]
    return (
        float(np.mean([h for h, _ in pairs])),
        float(np.mean(corrs)) if corrs else float("nan"),
    )


def _ranking_by_bucket(test_df: pd.DataFrame, pred_col: str, top_k: int) -> dict[str, dict]:
    per: dict[str, list[tuple[float, float]]] = {}
    for week, wdf in test_df.groupby("week"):
        hs = _week_hit_spear(wdf, pred_col, top_k)
        if hs is not None:
            per.setdefault(_week_bucket_eval(int(week)), []).append(hs)
    out = {}
    for bucket, vals in per.items():
        corrs = [c for _, c in vals if not np.isnan(c)]
        out[bucket] = {
            "hit": float(np.mean([h for h, _ in vals])),
            "spear": float(np.mean(corrs)) if corrs else float("nan"),
        }
    return out


def stage2_prediction_degradation(positions: list[str], top_k: int) -> None:
    print("=" * 78)
    print("STAGE 2 - does the final-week anomaly hurt PREDICTIONS? (production models)")
    print("  Buckets: wk1-16 (baseline) | wk17 (championship) | wk18 (dead/rest)")
    print("=" * 78)

    buckets = ["wk1-16", "wk17", "wk18"]
    for pos in positions:
        print(f"\n### {pos}: running pipeline (trains models)...")
        result = _run_position(pos, seed=42)
        test_df = result["test_df"].copy()
        if TRUE_COL not in test_df.columns:
            print(f"  ! {pos}: no '{TRUE_COL}' in test_df; skipping.")
            continue
        test_df["wk_bucket"] = test_df["week"].map(_week_bucket_eval)
        pred_cols = _prediction_columns(test_df)
        if not pred_cols:
            print(f"  ! {pos}: no prediction columns on test_df; skipping.")
            continue

        base_mask = test_df["wk_bucket"] == "wk1-16"
        best = min(
            pred_cols,
            key=lambda n: compute_metrics(
                test_df.loc[base_mask, TRUE_COL], test_df.loc[base_mask, pred_cols[n]]
            )["mae"],
        )

        print(
            f"  {'model':<14}{'bucket':<8}{'n':>6}{'MAE':>8}{'dMAE':>8}"
            f"{'RMSE':>8}{'R2':>7}{'top' + str(top_k):>7}{'spear':>7}"
        )
        takeaways = {}
        for name, col in pred_cols.items():
            ranking = _ranking_by_bucket(test_df, col, top_k)
            base_mae = compute_metrics(
                test_df.loc[base_mask, TRUE_COL], test_df.loc[base_mask, col]
            )["mae"]
            star = " *" if name == best else ""
            for bucket in buckets:
                sub = test_df[test_df["wk_bucket"] == bucket]
                if sub.empty:
                    continue
                m = compute_metrics(sub[TRUE_COL], sub[col])
                rk = ranking.get(bucket, {"hit": float("nan"), "spear": float("nan")})
                print(
                    f"  {name + star:<14}{bucket:<8}{len(sub):>6}{m['mae']:>8.3f}"
                    f"{(m['mae'] - base_mae):>8.3f}{m['rmse']:>8.3f}{m['r2']:>7.2f}"
                    f"{rk['hit']:>7.2f}{rk['spear']:>7.2f}"
                )
                if name == best:
                    takeaways[bucket] = (m["mae"] - base_mae, rk["hit"])

        base_hit = takeaways.get("wk1-16", (0.0, float("nan")))[1]
        d17 = takeaways.get("wk17", (float("nan"), float("nan")))
        d18 = takeaways.get("wk18", (float("nan"), float("nan")))
        print(
            f"  -> best={best}: DeltaMAE wk17={d17[0]:+.3f} wk18={d18[0]:+.3f} | "
            f"top{top_k} wk17={d17[1]:.2f} wk18={d18[1]:.2f} (base {base_hit:.2f})"
        )


def _drop_final_week(df: pd.DataFrame) -> pd.DataFrame:
    fw = df.groupby("season")["week"].transform("max")
    return df[df["week"] != fw].copy()


def run_ablation(
    positions: list[str], splits_dir: str | Path, top_k: int, eval_max_week: int
) -> None:
    from src.shared.pipeline import _read_split
    from src.shared.registry import get_runner

    print("=" * 78)
    print("ABLATION - does the season's FINAL WEEK in TRAINING help or hurt?")
    print("  KEEP = train on all weeks | CUT = drop each season's final week (train+val).")
    print(f"  Both eval on the SAME test set, weeks 1-{eval_max_week} (deployment-relevant).")
    print("  Same seed. dMAE = CUT - KEEP: positive => cutting HURTS => keep the rows.")
    print("=" * 78)

    train = _read_split(f"{splits_dir}/train.parquet")
    val = _read_split(f"{splits_dir}/val.parquet")
    test = _read_split(f"{splits_dir}/test.parquet")
    train_cut, val_cut = _drop_final_week(train), _drop_final_week(val)
    dropped = len(train) - len(train_cut)
    print(
        f"\n  train rows: keep={len(train)} cut={len(train_cut)} "
        f"(-{dropped}, {100 * dropped / len(train):.1f}%) | val: keep={len(val)} cut={len(val_cut)}"
    )

    for pos in positions:
        runner = get_runner(pos)
        print(f"\n### {pos}: training KEEP (all weeks)...")
        res_keep = runner(train, val, test, seed=42)
        print(f"### {pos}: training CUT (no final week)...")
        res_cut = runner(train_cut, val_cut, test, seed=42)

        tk, tc = res_keep["test_df"], res_cut["test_df"]
        pred_cols = _prediction_columns(tk)
        sub_k = tk[tk["week"] <= eval_max_week]
        sub_c = tc[tc["week"] <= eval_max_week]
        best = min(
            pred_cols, key=lambda n: compute_metrics(sub_k[TRUE_COL], sub_k[pred_cols[n]])["mae"]
        )
        print(
            f"  eval weeks 1-{eval_max_week}: n_keep={len(sub_k)} n_cut={len(sub_c)} (should match)"
        )
        print(
            f"  {'model':<14}{'MAE_keep':>9}{'MAE_cut':>9}{'dMAE':>8}{'hit_keep':>9}{'hit_cut':>9}"
        )
        for name, col in pred_cols.items():
            mk = compute_metrics(sub_k[TRUE_COL], sub_k[col])["mae"]
            mc = compute_metrics(sub_c[TRUE_COL], sub_c[col])["mae"]
            hk = _avg_weekly_ranking(sub_k, col, top_k)[0]
            hc = _avg_weekly_ranking(sub_c, col, top_k)[0]
            star = " *" if name == best else ""
            print(f"  {name + star:<14}{mk:>9.3f}{mc:>9.3f}{(mc - mk):>+8.3f}{hk:>9.2f}{hc:>9.2f}")
        bk = compute_metrics(sub_k[TRUE_COL], sub_k[pred_cols[best]])["mae"]
        bc = compute_metrics(sub_c[TRUE_COL], sub_c[pred_cols[best]])["mae"]
        verdict = "KEEP better" if bc > bk else ("CUT better" if bc < bk else "tie")
        print(f"  -> best={best}: dMAE={bc - bk:+.4f} on wk1-{eval_max_week}  =>  {verdict}")


# --------------------------------------------------------------------------- #
# Sparse-history / LGBM-disagreement helpers
# --------------------------------------------------------------------------- #
def peer_gap(df: pd.DataFrame, models: dict[str, str] | None = None) -> pd.Series:
    """peer-mean(Ridge, NN, Attn) - LGBM per row."""
    models = models or available_models(df)
    peer_cols = [models[p] for p in PEERS if p in models]
    return df[peer_cols].mean(axis=1) - df[models[LGBM]]


def calibration_table(
    df: pd.DataFrame,
    models: dict[str, str] | None = None,
    actual: str = ACTUAL,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Mean prediction per model within each actual-FP bin."""
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
    """Per-target fantasy-point contribution to the Ridge-LGBM total gap."""
    scoring = scoring or SCORING_PPR
    out: dict[str, float] = {}
    for target, weight in scoring.items():
        rc, lc = f"pred_ridge_{target}", f"pred_lgbm_{target}"
        if rc in df.columns and lc in df.columns:
            out[target] = float((df[rc] - df[lc]).mean() * weight)
    return out


def add_history_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy with ``n_prior_games`` by player-season week order."""
    out = df.sort_values(["player_id", "season", "week"]).copy()
    out["n_prior_games"] = out.groupby(["player_id", "season"]).cumcount()
    return out


def history_depth_table(
    df: pd.DataFrame,
    models: dict[str, str] | None = None,
    buckets: list[tuple[int, int, str]] | None = None,
) -> pd.DataFrame:
    """Per-model MAE and bias within each in-season history-depth bucket."""
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

    calib = calibration_table(df, models)
    fig, ax = plt.subplots(figsize=(6, 5))
    lo, hi = 0, float(calib["avg_actual"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect (y=x)")
    for name in models:
        ax.plot(calib["avg_actual"], calib[name], marker="o", label=name)
    ax.set_xlabel("mean actual FP (per bin)")
    ax.set_ylabel("mean predicted FP")
    ax.set_title("RB calibration by actual-FP bin")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(outdir, "rb_calibration.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)

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
        label="LGBM closer",
        color="tab:green",
    )
    ax.scatter(
        lgbm[~lgbm_closer],
        ridge[~lgbm_closer],
        s=10,
        alpha=0.4,
        label="Ridge closer",
        color="tab:red",
    )
    ax.set_xlabel("LightGBM predicted FP")
    ax.set_ylabel("Ridge predicted FP")
    ax.set_title("Ridge vs LightGBM RB predictions")
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(outdir, "rb_ridge_vs_lgbm.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    written = [p1, p2]
    if "n_prior_games" in df.columns:
        depths = list(range(0, 13))
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
        ax.set_xlabel("in-season games played before this week")
        ax.set_ylabel("MAE (fantasy points)")
        ax.set_title("RB MAE by in-season history depth")
        ax.legend()
        fig.tight_layout()
        p3 = os.path.join(outdir, "rb_mae_by_history_depth.png")
        fig.savefig(p3, dpi=120)
        plt.close(fig)
        written.append(p3)

    print("\nPlots written: " + "  |  ".join(written))


def sparse_history_deep_dive(df: pd.DataFrame, *, no_plots: bool = False) -> None:
    """Preserve the RB LGBM-vs-peers and sparse-history deep dive."""
    models = available_models(df)
    gap = peer_gap(df, models)
    df = add_history_depth(df.assign(_gap=gap))
    y = df[ACTUAL]

    print("\n" + "=" * 72)
    print("RB LightGBM disagreement analysis")
    print("=" * 72)
    print(f"RB test player-weeks: {len(df)}   mean actual FP: {y.mean():.2f}")

    _print_metric_table("GLOBAL (all RB rows)", per_model_metrics(df, models))

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

    print("\n=== CALIBRATION: mean prediction by actual-FP bin ===")
    print(calibration_table(df, models).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    _print_metric_table("early season (wk<=3)", per_model_metrics(df[df["week"] <= 3], models))
    _print_metric_table("rest of season (wk>=4)", per_model_metrics(df[df["week"] >= 4], models))
    _print_metric_table("genuinely HIGH actual (>=20 FP)", per_model_metrics(df[y >= 20], models))

    decomp = gap_decomposition(df[hi_mask])
    print(f"\n=== per-target FP gap (Ridge - LGBM) on gap>={DISAGREEMENT_THRESHOLD} ===")
    for t, v in sorted(decomp.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {t:18} {v:+7.3f} FP")
    print(f"  {'TOTAL':18} {sum(decomp.values()):+7.3f} FP")

    g = per_model_metrics(df, models)
    cls = per_model_metrics(df[hi_mask], models)
    lgbm_best_global = min(g, key=lambda k: g[k]["mae"]) == LGBM
    lgbm_best_class = bool(cls) and min(cls, key=lambda k: cls[k]["mae"]) == LGBM
    print("\n" + "-" * 72)
    if lgbm_best_global and lgbm_best_class:
        print(
            "VERDICT: EXPECTED behaviour, not a bug. LightGBM has the lowest MAE "
            "overall and on the exact disagreement class."
        )
    else:
        print("VERDICT: NEEDS REVIEW - LightGBM is not most accurate on the disagreement class.")
    print("-" * 72)

    print("\n" + "=" * 72)
    print("PART 2 - NN & attention NN by in-season history depth")
    print("=" * 72)
    hdt = history_depth_table(df, models)
    mae_cols = ["npg", "n", "avg_actual"] + [f"{n} MAE" for n in models]
    bias_cols = ["npg"] + [f"{n} bias" for n in models]
    print("\nMAE by history depth (n_prior_games == attention seq length):")
    print(hdt[mae_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nbias (pred - actual) by history depth:")
    print(hdt[bias_cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

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
        f"{len(HISTORY_DEPTH_BUCKETS)} history-depth buckets. Season-opener bias: "
        f"Attn {attn_opener_bias:+.2f} vs LGBM {lgbm_opener_bias:+.2f}."
    )
    print("-" * 72)

    if not no_plots:
        outdir = os.path.join(os.path.dirname(__file__), "outputs", "figures")
        _make_plots(df, gap, outdir)


# --------------------------------------------------------------------------- #
# New and consolidated label predicates
# --------------------------------------------------------------------------- #
def label_sparse_history_rows(df: pd.DataFrame) -> pd.Series:
    prior_counts = _prior_game_counts(df)
    if prior_counts is None:
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)
    values = []
    for n in prior_counts.to_numpy():
        label = UNKNOWN
        for lo, hi, bucket in HISTORY_DEPTH_BUCKETS:
            if lo <= n <= hi:
                label = bucket
                break
        values.append(label)
    return pd.Series(values, index=df.index, dtype=object)


def _prior_game_counts(df: pd.DataFrame) -> pd.Series | None:
    needed = ["player_id", "season", "week"]
    if any(c not in df.columns for c in needed):
        return None
    tmp = df[needed].copy()
    tmp["_pos"] = np.arange(len(tmp))
    tmp = tmp.sort_values(needed)
    tmp["n_prior_games"] = tmp.groupby(["player_id", "season"]).cumcount()
    tmp = tmp.sort_values("_pos")
    return pd.Series(tmp["n_prior_games"].to_numpy(), index=df.index)


def label_late_week_rows(df: pd.DataFrame) -> pd.Series:
    if any(c not in df.columns for c in ("season", "week")):
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)
    return assign_final_week_buckets(df)["week_bucket"].astype(object)


def label_injury_return_rows(df: pd.DataFrame) -> pd.Series:
    if RETURN_FLAG in df.columns:
        returning = pd.to_numeric(df[RETURN_FLAG], errors="coerce").fillna(0).eq(1)
    elif DAYS_REST in df.columns:
        returning = pd.to_numeric(df[DAYS_REST], errors="coerce").fillna(7).gt(7)
    else:
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)
    return pd.Series(np.where(returning, "returning", "settled"), index=df.index, dtype=object)


def label_committee_rows(
    df: pd.DataFrame,
    *,
    min_share: float = 0.25,
    max_share: float = 0.65,
) -> pd.Series:
    """RB committee label: at least two same-team RBs in a mid carry-share band."""
    keys = ["recent_team", "season", "week"]
    if any(c not in df.columns for c in keys):
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)
    if "game_carry_share" in df.columns:
        share = pd.to_numeric(df["game_carry_share"], errors="coerce").fillna(0)
    elif "carries" in df.columns:
        carries = pd.to_numeric(df["carries"], errors="coerce").fillna(0)
        team_carries = carries.groupby([df[c] for c in keys]).transform("sum")
        share = safe_divide(carries, team_carries).fillna(0)
    else:
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)

    is_mid = share.between(min_share, max_share, inclusive="both")
    if "position" in df.columns:
        is_mid &= df["position"].eq("RB")
    mid_count = is_mid.groupby([df[c] for c in keys]).transform("sum")
    committee = is_mid & (mid_count >= 2)
    return pd.Series(np.where(committee, COMMITTEE, NON_COMMITTEE), index=df.index, dtype=object)


def label_trade_rows(df: pd.DataFrame) -> pd.Series:
    """Label rows after a player changes ``recent_team`` within the same season."""
    needed = ["player_id", "season", "week", "recent_team"]
    if any(c not in df.columns for c in needed):
        return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)
    tmp = df[needed].copy()
    tmp["_pos"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["player_id", "season", "week"])
    first_team = tmp.groupby(["player_id", "season"])["recent_team"].transform("first")
    traded = tmp["recent_team"].ne(first_team) & tmp["recent_team"].notna() & first_team.notna()
    tmp["_label"] = np.where(traded, TRADED, STABLE_TEAM)
    return pd.Series(tmp.sort_values("_pos")["_label"].to_numpy(), index=df.index, dtype=object)


def label_suspension_return_rows(df: pd.DataFrame) -> pd.Series:
    return pd.Series([UNKNOWN] * len(df), index=df.index, dtype=object)


def _ctx_ascension(df: pd.DataFrame, _ctx: CohortContext) -> pd.Series:
    return label_ascension_rows(df)


def _ctx_rookie(df: pd.DataFrame, ctx: CohortContext) -> pd.Series:
    return label_rookie_rows(
        df, ctx.min_season if ctx.min_season is not None else pd.Series(), ctx.early_games
    )


def _ctx_injury_return(df: pd.DataFrame, _ctx: CohortContext) -> pd.Series:
    return label_injury_return_rows(df)


def _ctx_committee(df: pd.DataFrame, _ctx: CohortContext) -> pd.Series:
    return label_committee_rows(df)


def _ctx_trade(df: pd.DataFrame, _ctx: CohortContext) -> pd.Series:
    return label_trade_rows(df)


def _ctx_sparse_history(df: pd.DataFrame, _ctx: CohortContext) -> pd.Series:
    return label_sparse_history_rows(df)


def _ctx_late_week(df: pd.DataFrame, _ctx: CohortContext) -> pd.Series:
    return label_late_week_rows(df)


def _ctx_suspension(df: pd.DataFrame, _ctx: CohortContext) -> pd.Series:
    return label_suspension_return_rows(df)


def _ctx_scoring_tier(df: pd.DataFrame, ctx: CohortContext) -> pd.Series:
    return label_scoring_tier_rows(df, ctx.prior_season_fp, top_n=ctx.tier_topn)


COHORTS: dict[str, CohortSpec] = {
    "scoring_tier": CohortSpec(
        "scoring_tier",
        "A-priori elite/top-drafted slice: prior-season mean FP rank per (season, "
        "position) vs the veteran field. Diagnoses tail under-prediction.",
        tuple(TIER_POSITIONS),
        TIER_BUCKET,
        TIER_ELITE,
        _ctx_scoring_tier,
    ),
    "ascension": CohortSpec(
        "ascension",
        "RB backup-to-workhorse jump: prior-3 opportunities <=8 and current opportunities >=18.",
        ("RB",),
        ROLE_CHANGE,
        ASCENSION,
        _ctx_ascension,
    ),
    "injury_return": CohortSpec(
        "injury_return",
        "Returning after an absence, using pipeline return/rest columns.",
        tuple(DEFAULT_POSITIONS),
        INJURY_RETURN_BUCKET,
        "returning",
        _ctx_injury_return,
    ),
    "rookie": CohortSpec(
        "rookie",
        "First season in the split data, split into early/rest rookie games.",
        tuple(DEFAULT_ROOKIE_POSITIONS),
        ROOKIE_BUCKET,
        ROOKIE_EARLY,
        _ctx_rookie,
    ),
    "committee": CohortSpec(
        "committee",
        "RB one-two-punch committee games with at least two mid-share backs.",
        ("RB",),
        "committee_bucket",
        COMMITTEE,
        _ctx_committee,
    ),
    "trade": CohortSpec(
        "trade",
        "Rows after a mid-season team change within the same test season.",
        tuple(DEFAULT_ROOKIE_POSITIONS),
        "trade_bucket",
        TRADED,
        _ctx_trade,
    ),
    "sparse_history": CohortSpec(
        "sparse_history",
        "In-season history-depth buckets; 0 means season opener.",
        tuple(DEFAULT_POSITIONS),
        HISTORY_BUCKET,
        "0",
        _ctx_sparse_history,
    ),
    "late_week": CohortSpec(
        "late_week",
        "Season-aware early/penultimate/final week buckets.",
        tuple(DEFAULT_POSITIONS),
        "week_bucket",
        FINAL,
        _ctx_late_week,
    ),
    "suspension_return": CohortSpec(
        "suspension_return",
        "Infeasible: no nflverse suspension data source is available.",
        tuple(),
        "suspension_bucket",
        SUSPENSION_RETURN,
        _ctx_suspension,
        feasible=False,
    ),
}


def _print_bucket_sizes(df: pd.DataFrame, spec: CohortSpec) -> None:
    counts = df[spec.label_col].value_counts(dropna=False).sort_index()
    print(f"\n  {spec.name}: {spec.description}")
    for bucket, n in counts.items():
        pct = n / len(df) if len(df) else float("nan")
        print(f"    {bucket:<18} {int(n):>6} ({pct:>5.1%})")


def _print_uniform_model_report(pos: str, spec: CohortSpec, df: pd.DataFrame, top_k: int) -> None:
    models = _prediction_columns(df)
    if not models:
        print(f"  ! {pos}/{spec.name}: no prediction columns; skipping.")
        return
    if ACTUAL not in df.columns:
        print(f"  ! {pos}/{spec.name}: no {ACTUAL!r} column; skipping.")
        return

    _hr(f"{pos}: {spec.name} per-model error")
    _print_bucket_sizes(df, spec)

    tbl = bucket_model_table(df, spec.label_col, models)
    print(f"\n  {'model':<14}{'bucket':<18}{'N':>6}{'MAE':>9}{'dMAE':>9}{'RMSE':>9}{'bias':>9}")
    for _, r in tbl.sort_values(["model", "bucket"]).iterrows():
        print(
            f"  {r['model']:<14}{r['bucket']:<18}{int(r['n']):>6}"
            f"{r['mae']:>9.3f}{r['dmae']:>+9.3f}{r['rmse']:>9.3f}{r['bias']:>+9.3f}"
        )

    if {"week", "player_id", spec.label_col} <= set(df.columns):
        print(f"\n  Ranking by bucket (top{top_k}; weeks with fewer than top_k rows skipped):")
        print(f"  {'model':<14}{'bucket':<18}{'hit':>8}{'spear':>8}")
        for name, col in models.items():
            for bucket, sub in df.groupby(spec.label_col, observed=True, sort=True):
                hit, spear = _avg_weekly_ranking(sub, col, top_k)
                print(f"  {name:<14}{str(bucket):<18}{hit:>8.2f}{spear:>8.2f}")


def _discover_targets(df: pd.DataFrame) -> list[str]:
    """Raw-stat target names from the ``pred_ridge_<target>`` per-target columns."""
    prefix = "pred_ridge_"
    return sorted(
        c[len(prefix) :] for c in df.columns if c.startswith(prefix) and c != "pred_ridge_total"
    )


def _print_tier_target_bias(pos: str, spec: CohortSpec, df: pd.DataFrame) -> None:
    """Localize the tail gap: signed bias per (model, raw target) x {elite, field}.

    Total-FP bias says *whether* the elite slice is mis-predicted; this pins it to
    the specific raw-stat head (e.g. passing_yards) and model that drive it, which
    is what scopes a Phase-2 loss/feature intervention.
    """
    models = _prediction_columns(df)
    targets = _discover_targets(df)
    if not models or not targets or spec.label_col not in df.columns:
        return

    _hr(f"{pos}: {spec.name} per-target signed bias (deep dive)")
    print("  bias = mean(pred - actual); negative on elite = under-prediction.")
    print(f"  {'model':<14}{'target':<26}{'elite_bias':>12}{'field_bias':>12}")
    stem = {name: col[len("pred_") : -len("_total")] for name, col in models.items()}
    for name in models:
        for target in targets:
            pred_col = f"pred_{stem[name]}_{target}"
            if pred_col not in df.columns or target not in df.columns:
                continue
            metrics = compute_stratum_metrics(df, target, pred_col, spec.label_col)
            by_bucket = metrics.set_index(spec.label_col)["bias"]
            elite = by_bucket.get(TIER_ELITE, float("nan"))
            field = by_bucket.get(TIER_FIELD, float("nan"))
            print(f"  {name:<14}{target:<26}{elite:>+12.3f}{field:>+12.3f}")


def _print_injury_detailed(pos: str, df: pd.DataFrame) -> dict:
    models = _prediction_columns(df)
    n_total = len(df)
    print(f"\n{'=' * 80}")
    print(
        f"{pos}: injury / return subgroup error   "
        f"(test player-weeks: {n_total}, mean actual FP: {df[ACTUAL].mean():.2f})"
    )
    print(f"models present: {', '.join(models)}")
    print("=" * 80)

    record: dict[str, dict] = {}
    for key, label, needed, mask_fn in SUBGROUP_SPECS:
        if needed is not None and needed not in df.columns:
            continue
        sub = df[mask_fn(df)]
        n = len(sub)
        pct = 100.0 * n / n_total if n_total else 0.0
        metrics = per_model_metrics(sub, models)
        record[key] = {"label": label.strip(), "n": n, "pct": round(pct, 1), "models": metrics}

        print(f"\n--- {label}   (n={n}, {pct:.1f}% of test){_flag(n)} ---")
        if n == 0:
            continue
        print(f"    {'model':14}{'MAE':>9}{'bias':>9}{'RMSE':>9}")
        for name, m in metrics.items():
            print(f"    {name:14}{m['mae']:9.3f}{m['bias']:9.3f}{m['rmse']:9.3f}")

    _print_injury_verdict(pos, record)
    return record


def _print_injury_verdict(pos: str, record: dict) -> None:
    g = record.get("global", {}).get("models", {})
    if not g:
        return
    best = min(g, key=lambda k: g[k]["mae"])
    g_mae = g[best]["mae"]
    print(f"\n  -> best model (lowest GLOBAL MAE): {best} = {g_mae:.3f}")
    for key, name in [("returning", "returning"), ("questionable", "Questionable")]:
        sub = record.get(key)
        if not sub or sub["n"] == 0:
            continue
        bm = sub["models"][best]["mae"]
        tag = _flag(sub["n"]).strip()
        tag = f"  {tag}" if tag else ""
        print(
            f"    {best} MAE on {name:12}: {bm:7.3f}  "
            f"(Delta vs global {bm - g_mae:+.3f}, n={sub['n']}, {sub['pct']:.1f}% of test){tag}"
        )


def analyze_position(pos: str) -> dict:
    """Compatibility entry point for the old injury subgroup CLI."""
    result = _run_position(pos, seed=42)
    df = result["test_df"].copy()
    record = _print_injury_detailed(pos, df)
    return {
        "position": pos,
        "n_test": len(df),
        "models_present": list(_prediction_columns(df)),
        "subgroups": record,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _run_position(
    pos: str,
    *,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    seed: int = 42,
) -> dict:
    from src.shared.registry import get_runner

    run = get_runner(pos)
    if pos in {"K", "DST"}:
        return run(seed=seed)
    if train_df is not None and val_df is not None and test_df is not None:
        return run(train_df, val_df, test_df, seed)
    return run(seed=seed)


def _applicable_positions(spec: CohortSpec, requested: list[str]) -> list[str]:
    return [p for p in requested if p in spec.positions]


def _validate_positions(parser: argparse.ArgumentParser, positions: list[str]) -> list[str]:
    out = [p.upper() for p in positions]
    unknown = [p for p in out if p not in POSITIONS]
    if unknown:
        parser.error(f"unknown position(s) {unknown}; choose from {POSITIONS}")
    return out


def _selected_cohorts(parser: argparse.ArgumentParser, names: list[str]) -> list[CohortSpec]:
    requested = names or [n for n, c in COHORTS.items() if c.feasible]
    unknown = [n for n in requested if n not in COHORTS]
    if unknown:
        parser.error(f"unknown cohort(s) {unknown}; choose from {sorted(COHORTS)}")
    return [COHORTS[n] for n in requested]


def _load_context(
    splits_dir: str | Path, early_games: int, tier_topn: int = DEFAULT_TIER_TOPN
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CohortContext]:
    train_df, val_df, test_df = _load_splits(splits_dir)
    return (
        train_df,
        val_df,
        test_df,
        CohortContext(
            min_season=player_min_season([train_df, val_df, test_df]),
            early_games=early_games,
            prior_season_fp=player_prior_season_fp([train_df, val_df, test_df]),
            tier_topn=tier_topn,
        ),
    )


def _run_data_only(args: argparse.Namespace, specs: list[CohortSpec], positions: list[str]) -> None:
    train_df = val_df = test_df = None
    ctx = CohortContext(early_games=args.early_games, tier_topn=args.tier_topn)
    needs_splits = any(s.name != "ascension" for s in specs)
    if needs_splits:
        train_df, val_df, test_df, ctx = _load_context(
            args.splits_dir, args.early_games, args.tier_topn
        )

    for spec in specs:
        if not spec.feasible:
            _hr(f"{spec.name}: infeasible")
            print(
                "  No nflverse suspension data source is available; cohort is intentionally unknown."
            )
            continue
        if spec.name == "ascension":
            ascension_data_diagnostic(deep_dive=args.deep_dive or len(specs) == 1)
            continue
        if spec.name == "late_week" and args.deep_dive:
            stage1_label_anomaly(args.splits_dir, args.established_games)
            continue
        if spec.name == "rookie":
            _print_cohort_sizes(
                test_df, ctx.min_season, _applicable_positions(spec, positions), args.early_games
            )
            continue

        _hr(f"{spec.name}: data-only cohort sizes")
        data = test_df.copy()
        if "position" in data.columns:
            data = data[data["position"].isin(_applicable_positions(spec, positions))].copy()
        data[spec.label_col] = spec.label_fn(data, ctx)
        _print_bucket_sizes(data, spec)


def _run_model_reports(
    args: argparse.Namespace, specs: list[CohortSpec], positions: list[str]
) -> None:
    train_df, val_df, test_df, ctx = _load_context(
        args.splits_dir, args.early_games, args.tier_topn
    )
    needed_positions: list[str] = []
    for spec in specs:
        for pos in _applicable_positions(spec, positions):
            if pos not in needed_positions:
                needed_positions.append(pos)

    results: dict[str, dict] = {}
    for pos in needed_positions:
        print(f"\nRunning {pos} pipeline for cohort model error ...", flush=True)
        results[pos] = _run_position(
            pos, train_df=train_df, val_df=val_df, test_df=test_df, seed=args.seed
        )

    for spec in specs:
        if not spec.feasible:
            _hr(f"{spec.name}: infeasible")
            print(
                "  No nflverse suspension data source is available; cohort is intentionally unknown."
            )
            continue
        if spec.name == "ascension" and args.deep_dive:
            ascension_data_diagnostic(deep_dive=True)
        if spec.name == "late_week" and args.deep_dive:
            stage1_label_anomaly(args.splits_dir, args.established_games)

        for pos in _applicable_positions(spec, positions):
            df = results[pos]["test_df"].copy()
            df[spec.label_col] = spec.label_fn(df, ctx)
            if spec.name == "rookie":
                _print_rookie_model_error(pos, df, ctx.min_season, args.early_games)
            elif spec.name == "injury_return" and args.deep_dive:
                _print_injury_detailed(pos, df)
            else:
                _print_uniform_model_report(pos, spec, df, args.top_k)

            if spec.name == "scoring_tier" and args.deep_dive:
                _print_tier_target_bias(pos, spec, df)

            if spec.name == "sparse_history" and pos == "RB" and args.deep_dive:
                sparse_history_deep_dive(df, no_plots=args.no_plots)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cohorts",
        nargs="*",
        help=f"cohort names (default: all feasible). Choices: {', '.join(sorted(COHORTS))}",
    )
    parser.add_argument(
        "--with-model-error",
        action="store_true",
        help="run production pipelines and emit uniform per-model cohort error tables",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="force data-only mode (default unless --with-model-error is set)",
    )
    parser.add_argument(
        "--deep-dive", action="store_true", help="also print preserved bespoke sections"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="late_week only: train KEEP vs CUT(final week) and compare",
    )
    parser.add_argument("--positions", nargs="*", default=DEFAULT_POSITIONS)
    parser.add_argument("--splits-dir", default=SPLITS_DIR)
    parser.add_argument("--established-games", type=int, default=DEFAULT_ESTABLISHED_GAMES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--tier-topn",
        type=int,
        default=DEFAULT_TIER_TOPN,
        help="scoring_tier only: top-N players per (season, position) counted as elite",
    )
    parser.add_argument("--eval-max-week", type=int, default=17)
    parser.add_argument("--early-games", type=int, default=EARLY_GAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true", help="skip sparse-history PNG output")
    args = parser.parse_args(argv)

    positions = _validate_positions(parser, args.positions)
    specs = _selected_cohorts(parser, args.cohorts)

    if args.ablation:
        if any(s.name != "late_week" for s in specs):
            parser.error("--ablation is only defined for the late_week cohort")
        run_ablation(positions, args.splits_dir, args.top_k, args.eval_max_week)
        return

    if args.with_model_error and not args.no_model:
        _run_model_reports(args, specs, positions)
    else:
        _run_data_only(args, specs, positions)


if __name__ == "__main__":
    main()
