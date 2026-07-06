"""A/B screen: B2 team PROE / pace season-to-date features (expert-edge Phase 2).

The expert-gap investigation (todo/expert-gap-investigation-2026-06.md) left RB as
the one ordering lever plausibly closable from data we already carry; the
new-sources research (todo/new-sources-research-2026-06.md §B2) ranked team
Pass-Rate-Over-Expected + neutral pace as the highest-confidence ordering
re-sorter. This spec screens that bundle: four season-to-date team-week rates
(PROE, neutral pass rate, neutral seconds/play, plays/game) built from nflfastR
play-by-play, lagged through W-1, broadcast onto player rows, and whitelisted
into every model plus the attention static branch (season-to-date rates are
static-eligible — they are not windowed player stats).

Judge on RANK metrics, not MAE (the edge being chased is ordering): per-model
hit@12 + Spearman (src.shared.evaluation.compute_ranking_metrics), per-week
optimal-lineup regret at the position's lineup size and at 12, and the a-priori
elite_top24 slice (benchmark._elite_top24_mask). Custom metrics land in
summary.json (print_report echoes only "mae").

Local (grid smoke; PBP network not required for the baseline identity cell):
    python -m src.tuning.ab_proe_pace --positions RB --only baseline --seeds 42 \
        --no-stacked-seeds -j 1

Fleet (eager — the metric needs all four pred columns; ADR-0020 branch flow):
    python -m src.tuning.launch_ab --spec src.tuning.ab_proe_pace --dry-run
    # smoke ONE arm cell before fanning out (#1187→#1212 lesson):
    python -m src.tuning.launch_ab --spec src.tuning.ab_proe_pace \
        --positions RB --only proe_pace --seeds 42
Variant names are bare identifiers so the ab-batch.yml `only` input accepts them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data import nfl_source
from src.data.cache_io import atomic_write_parquet
from src.data.external_sources import _seasons_cache_signature
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE"]  # frame injection is QB/RB/WR/TE-only
SEEDS = [42, 123, 7]

# nflfastR columns the builder needs; pbp_data() drops absent columns silently,
# so _build_team_week_table gates on them loudly instead.
_PBP_COLS = (
    "game_id",
    "season",
    "week",
    "posteam",
    "drive",
    "pass",
    "pass_oe",
    "wp",
    "game_seconds_remaining",
    "qb_kneel",
    "qb_spike",
)
_REQUIRED_PBP = frozenset(_PBP_COLS) - {"qb_kneel", "qb_spike"}  # kneel/spike optional

_NEUTRAL_WP = (0.2, 0.8)  # win-probability band for "neutral game script"
_MAX_SNAP_GAP_S = 45.0  # within-drive snap-to-snap gaps above this are breaks, not pace

FEATURE_COLS = (
    "team_proe_std",
    "team_neutral_pass_rate_std",
    "team_neutral_sec_per_play_std",
    "team_plays_pg_std",
)

# Investigation lineup sizes (ab_rolling_origin_rotowire._LINEUP_N) for the
# position-appropriate regret metric.
_LINEUP_N = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}

_CACHE_VERSION = "v1"


# --------------------------------------------------------------------------- #
# Team-week feature table (screen-local; ships to src/data/ only if this wins)
# --------------------------------------------------------------------------- #
def _aggregate_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, season, week, posteam) raw pace/tendency aggregates.

    Kneels/spikes are dropped when the columns exist (they are clock plays, not
    tendency); pass_oe is NaN off dropback-eligible plays and mean() skips it.
    """
    df = pbp[pbp["posteam"].notna()].copy()
    for col in ("qb_kneel", "qb_spike"):
        if col in df.columns:
            df = df[df[col].fillna(0) == 0]

    neutral = df[(df["wp"] >= _NEUTRAL_WP[0]) & (df["wp"] <= _NEUTRAL_WP[1])]

    keys = ["game_id", "season", "week", "posteam"]
    out = df.groupby(keys, as_index=False).agg(
        game_proe=("pass_oe", "mean"),
        game_plays=("pass", "size"),
    )
    npass = neutral.groupby(keys, as_index=False).agg(game_neutral_pass_rate=("pass", "mean"))

    # Neutral seconds/play: within-(game, offense, drive) snap-to-snap clock
    # deltas; gaps > _MAX_SNAP_GAP_S are possession/quarter breaks, not pace.
    n = neutral.sort_values(
        keys + ["drive", "game_seconds_remaining"], ascending=[True] * 5 + [False]
    )
    gaps = -n.groupby(keys + ["drive"])["game_seconds_remaining"].diff()
    n = n.assign(_gap=gaps)
    n = n[(n["_gap"] > 0) & (n["_gap"] <= _MAX_SNAP_GAP_S)]
    npace = n.groupby(keys, as_index=False).agg(game_neutral_sec_per_play=("_gap", "mean"))

    out = out.merge(npass, on=keys, how="left").merge(npace, on=keys, how="left")
    return out


def _season_to_date_shift(games: pd.DataFrame) -> pd.DataFrame:
    """Expanding means through W-1 within (posteam, season) — leakage-safe.

    Shift is by GAME row (a team's previous completed game), so bye weeks are
    handled naturally; season groups reset, so openers are NaN and production
    fill_nans imputes the train mean (0.0 would be out-of-distribution — the
    ab_opp_coverage_wr convention).
    """
    g = games.sort_values(["posteam", "season", "week"]).reset_index(drop=True)
    grp = g.groupby(["posteam", "season"], sort=False)
    pairs = {
        "team_proe_std": "game_proe",
        "team_neutral_pass_rate_std": "game_neutral_pass_rate",
        "team_neutral_sec_per_play_std": "game_neutral_sec_per_play",
        "team_plays_pg_std": "game_plays",
    }
    for out_col, src_col in pairs.items():
        g[out_col] = grp[src_col].transform(lambda s: s.expanding().mean().shift(1))
    tbl = g[["posteam", "season", "week", *FEATURE_COLS]].rename(columns={"posteam": "recent_team"})
    # PBP posteam and the weekly frame's recent_team are both nflverse-native
    # (LA/LV/WAS...), so no team-code normalization — the redzone_pbp precedent.
    return tbl


def _build_team_week_table(seasons: list[int], cache_dir: str | None = None) -> pd.DataFrame:
    """Cached (recent_team, season, week) → season-to-date PROE/pace features."""
    from src.config import CACHE_DIR

    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    sig = _seasons_cache_signature(sorted(set(int(s) for s in seasons)))
    cache_path = f"{cache_dir}/proe_pace_teamweek_{_CACHE_VERSION}_{sig}.parquet"
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if set(FEATURE_COLS) <= set(cached.columns):
            return cached

    pbp = nfl_source.pbp_data(sorted(set(int(s) for s in seasons)), list(_PBP_COLS))
    missing = _REQUIRED_PBP - set(pbp.columns)
    if missing:
        raise RuntimeError(
            f"PBP feed is missing required columns {sorted(missing)} — pbp_data drops "
            "absent columns silently, so failing loud instead of zeroing the feature."
        )
    tbl = _season_to_date_shift(_aggregate_team_games(pbp))
    atomic_write_parquet(tbl, cache_path)
    return tbl


# --------------------------------------------------------------------------- #
# Variant plumbing
# --------------------------------------------------------------------------- #
def _inject_proe(train, val, test):
    """Left-join the team-week table onto all three general frames.

    Row order and count are preserved (m:1 left merge on a unique team-week
    key); frames arrive RangeIndexed from parquet, so merge's index reset is a
    no-op. Missing keys / openers stay NaN for fill_nans.
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
            raise RuntimeError("proe_pace injector changed row count — key not unique?")
        out.append(merged)
    return tuple(out)


def _mut_whitelist(cfg: dict) -> dict:
    """Feed the new columns to Ridge/LightGBM/NN-static AND the attention static
    branch (season-to-date team rates are non-windowed → static-eligible)."""
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
        "proe_pace",
        cfg_mutator=_mut_whitelist,
        frame_injector=_inject_proe,
        expect_ridge_identical=False,  # a real feature must move Ridge (else it didn't take)
        label="B2 bundle: PROE + neutral pass rate + neutral pace + plays/game (season-to-date)",
    ),
]


# --------------------------------------------------------------------------- #
# Metric — ordering first, MAE as the guard
# --------------------------------------------------------------------------- #
def _regret(df: pd.DataFrame, col: str, n: int) -> float:
    """Per-week optimal-lineup shortfall, mean over weeks (mirrors
    ab_rolling_origin_rotowire._regret; inlined to avoid its module imports)."""
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
