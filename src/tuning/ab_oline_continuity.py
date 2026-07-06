"""A/B screen: E2 offensive-line continuity features (expert-edge Phase 2b).

B2 team PROE/pace (src/tuning/ab_proe_pace.py) was TESTED, REJECTED as the RB
ordering lever (benchmark-flat on the served heads). E2 is the runner-up from the
new-sources research (§E2, todo/new-sources-research-2026-06.md:109-121): O-line
*continuity* — starting-five stability — is orthogonal to team pass-volume and
the research's second-best in-house RB signal. Built from the PFR snap_counts the
repo already ingests (loader.py keeps only skill-player offense_pct and discards
the OL rows); this rebuilds from the raw table.

Three team-week features, all realized-through-W-1 (leakage-safe; describe the
line ENTERING week W, computable pre-kickoff), keyed to (recent_team, season,
week):
  - oline_distinct_starters_std : # distinct top-5-by-offense_pct OL who started
    any completed game this season BEFORE week W (5 = perfectly stable).
  - oline_prev_overlap_std      : overlap (0-5) of the top-5 in the two most
    recent completed games before W.
  - oline_stable_streak_std     : consecutive most-recent completed games before
    W with an unchanged top-5.

Judge on RANK metrics, not MAE (the edge chased is ordering). Same harness /
fleet / metric machinery as ab_proe_pace; only the source (snap_counts) and the
aggregation body differ. Team codes are normalized STL→LA / OAK→LV / SD→LAC via
schedule_team_code_normalization() (snap_counts carries legacy codes; the splits'
recent_team is modern — verified 100% join across all splits after the map).

Local (grid smoke; snap network not required for the baseline identity cell):
    python -m src.tuning.ab_oline_continuity --positions RB --only baseline \
        --seeds 42 --no-stacked-seeds -j 1

Fleet (eager; ADR-0020 branch flow):
    python -m src.tuning.launch_ab --spec src.tuning.ab_oline_continuity --dry-run
    # smoke ONE arm cell before fanning out (#1187→#1212 lesson):
    python -m src.tuning.launch_ab --spec src.tuning.ab_oline_continuity \
        --positions RB --only oline_continuity --seeds 42
Variant names are bare identifiers so the ab-batch.yml `only` input accepts them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# NB: keep module-level imports to os/numpy/pandas/ab_harness only. launch_ab
# imports this spec on the lightweight GitHub launcher runner (to size the grid)
# where the training deps are absent — a top-level `from src.data import ...`
# pulls nflreadpy and crashes the launcher (ModuleNotFoundError). The snap/cache
# helpers are imported inside _build_team_week_table, which runs only inside the
# full-deps Batch container. Pinned by test_ab_oline_continuity.
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE"]  # RB primary + QB secondary (§E2); others measure spillover
SEEDS = [42, 123, 7]

# OL position tokens (snap_counts `position` splits on "/": e.g. "G/OT", "C/G").
# Token-intersection so combos count; no single non-OL code collides (verified
# against the 2013-2025 distinct-position set).
_OL_TOKENS = frozenset({"T", "G", "C", "OL", "OT", "OG"})

# Raw snap_counts columns the builder needs (nfl_source.snap_counts drops absent
# columns silently, so _build_team_week_table gates on them loudly).
_REQUIRED_SNAP = frozenset({"position", "team", "offense_pct", "pfr_player_id", "season", "week"})

_STARTERS = 5  # a starting offensive line

FEATURE_COLS = (
    "oline_distinct_starters_std",
    "oline_prev_overlap_std",
    "oline_stable_streak_std",
)

# Investigation lineup sizes (ab_proe_pace / ab_rolling_origin_rotowire) for the
# position-appropriate regret metric.
_LINEUP_N = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}

_CACHE_VERSION = "v1"


# --------------------------------------------------------------------------- #
# Team-week feature table (screen-local; ships to src/data/ only if this wins)
# --------------------------------------------------------------------------- #
def _aggregate_oline_games(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per (recent_team, season, week) frozenset of the top-5 OL by offense_pct.

    Legacy team codes (OAK/SD/STL) are normalized to the modern schedule-universe
    codes the splits use (LV/LAC/LA) before keying.
    """
    from src.data.nflcom_loader import schedule_team_code_normalization

    tmap = schedule_team_code_normalization()
    df = snaps.copy()
    df["recent_team"] = df["team"].map(lambda c: tmap.get(c, c))
    is_ol = (
        df["position"].fillna("").str.split("/").apply(lambda toks: bool(set(toks) & _OL_TOKENS))
    )
    df = df[is_ol & df["offense_pct"].notna()].copy()
    # Top-5 by snap share within each team-game (one game per team-week).
    df["_rk"] = df.groupby(["recent_team", "season", "week"])["offense_pct"].rank(
        method="first", ascending=False
    )
    top = df[df["_rk"] <= _STARTERS]
    lines = (
        top.groupby(["recent_team", "season", "week"])["pfr_player_id"]
        .apply(frozenset)
        .reset_index(name="top5")
    )
    return lines


def _season_to_date_continuity(games: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe continuity features computed on games strictly BEFORE week W.

    Walks each (team, season) in week order: distinct-starters = |union of all
    prior lines|; prev-overlap = |line[W-1] ∩ line[W-2]|; stable-streak = run of
    identical lines ending at W-1. Openers (no prior game) are NaN → fill_nans.
    """
    g = games.sort_values(["recent_team", "season", "week"]).reset_index(drop=True)
    rows = []
    for (team, season), grp in g.groupby(["recent_team", "season"], sort=False):
        lines = list(grp["top5"])
        weeks = list(grp["week"])
        union: set = set()
        for i in range(len(lines)):
            if i == 0:
                distinct = overlap = streak = float("nan")
            else:
                distinct = float(len(union))  # union of lines[0..i-1]
                overlap = float(len(lines[i - 1] & lines[i - 2])) if i >= 2 else float("nan")
                s, j = 1, i - 1
                while j - 1 >= 0 and lines[j - 1] == lines[j]:
                    s += 1
                    j -= 1
                streak = float(s)
            rows.append((team, season, weeks[i], distinct, overlap, streak))
            union |= lines[i]
    return pd.DataFrame(rows, columns=["recent_team", "season", "week", *FEATURE_COLS])


def _build_team_week_table(seasons: list[int], cache_dir: str | None = None) -> pd.DataFrame:
    """Cached (recent_team, season, week) → season-to-date O-line continuity."""
    # Deferred (see the module-header note): these pull training deps absent on
    # the launcher runner, and only this function runs in the Batch container.
    from src.config import CACHE_DIR
    from src.data import nfl_source
    from src.data.cache_io import atomic_write_parquet
    from src.data.external_sources import _seasons_cache_signature

    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    sig = _seasons_cache_signature(sorted(set(int(s) for s in seasons)))
    cache_path = f"{cache_dir}/oline_continuity_{_CACHE_VERSION}_{sig}.parquet"
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if set(FEATURE_COLS) <= set(cached.columns):
            return cached

    snaps = nfl_source.snap_counts(sorted(set(int(s) for s in seasons)))
    missing = _REQUIRED_SNAP - set(snaps.columns)
    if missing:
        raise RuntimeError(
            f"snap_counts is missing required columns {sorted(missing)} — the loader drops "
            "absent columns silently, so failing loud instead of zeroing the feature."
        )
    tbl = _season_to_date_continuity(_aggregate_oline_games(snaps))
    atomic_write_parquet(tbl, cache_path)
    return tbl


# --------------------------------------------------------------------------- #
# Variant plumbing (identical shape to ab_proe_pace)
# --------------------------------------------------------------------------- #
def _inject_oline(train, val, test):
    """Left-join the team-week table onto all three general frames.

    Row order and count preserved (m:1 left merge on a unique team-week key);
    frames arrive RangeIndexed from parquet. Missing keys / openers stay NaN.
    """
    seasons = sorted(
        set(train["season"].unique()) | set(val["season"].unique()) | set(test["season"].unique())
    )
    tbl = _build_team_week_table([int(s) for s in seasons])
    key = ["recent_team", "season", "week"]
    out = []
    for df in (train, val, test):
        merged = df.merge(tbl, on=key, how="left", validate="m:1")
        if len(merged) != len(df):
            raise RuntimeError("oline injector changed row count — key not unique?")
        out.append(merged)
    return tuple(out)


def _mut_whitelist(cfg: dict) -> dict:
    """Feed the new columns to Ridge/LightGBM/NN-static AND the attention static
    branch (season-to-date team continuity is non-windowed → static-eligible)."""
    base = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda base=base: [*base(), *FEATURE_COLS]
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [
            *cfg["attn_static_features"],
            *[c for c in FEATURE_COLS if c not in cfg["attn_static_features"]],
        ]
    return cfg


VARIANTS = [
    Variant("baseline"),
    Variant(
        "oline_continuity",
        cfg_mutator=_mut_whitelist,
        frame_injector=_inject_oline,
        expect_ridge_identical=False,  # a real feature must move Ridge (else it didn't take)
        label="E2 bundle: distinct starters + prev-game overlap + stable streak (season-to-date)",
    ),
]


# --------------------------------------------------------------------------- #
# Metric — ordering first, MAE as the guard (identical to ab_proe_pace)
# --------------------------------------------------------------------------- #
def _regret(df: pd.DataFrame, col: str, n: int) -> float:
    """Per-week optimal-lineup shortfall, mean over weeks."""
    regrets = []
    for _, g in df.groupby("week"):
        gg = g[g[col].notna()]
        if len(gg) < n:
            continue
        opt = gg.nlargest(n, "fantasy_points")["fantasy_points"].sum()
        got = gg.nlargest(n, col)["fantasy_points"].sum()
        regrets.append(opt - got)
    return float(np.mean(regrets)) if regrets else float("nan")


def metric_fn(result: dict, position: str) -> dict[str, dict[str, float]]:
    from src.analysis.cohort_analysis import MODELS, available_models, per_model_metrics
    from src.shared.evaluation import compute_ranking_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)
    lineup_n = _LINEUP_N.get(position, 12)

    elite_mask = None
    if "prior_season_mean_fantasy_points" in df.columns:
        from src.benchmarking.benchmark import _elite_top24_mask

        elite_mask = _elite_top24_mask(df)

    out: dict[str, dict[str, float]] = {}
    for name in models:
        col = MODELS[name]
        row: dict[str, float] = {"mae": overall[name]["mae"]}
        rank = compute_ranking_metrics(df, pred_col=col, top_k=12)
        row["hit12"] = rank["season_avg_hit_rate"]
        row["spearman"] = rank["season_avg_spearman"]
        row["regret12"] = _regret(df, col, 12)
        row["regret_lineup"] = _regret(df, col, lineup_n)
        if elite_mask is not None:
            elite = df[elite_mask]
            err = elite[col] - elite["fantasy_points"]
            row["elite24_mae"] = float(err.abs().mean())
            row["elite24_bias"] = float(err.mean())
        out[name] = row
    return out


if __name__ == "__main__":
    ab_main(__spec__.name)
