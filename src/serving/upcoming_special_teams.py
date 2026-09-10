"""K/DST upcoming rows using the training builders and pregame-only history.

This runs in the CI artifact job, never in the serving download poller. Current
season inputs are fetched independently of the fixed training/holdout years.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, SEASONS
from src.data import nfl_source
from src.data.loader import load_team_week_stats
from src.dst import data as dst_data
from src.dst import features as dst_features
from src.dst import targets as dst_targets
from src.dst.config import POSITION_CONFIG as DST_CONFIG
from src.k import data as k_data
from src.k import features as k_features
from src.k import targets as k_targets
from src.k.config import POSITION_CONFIG as K_CONFIG
from src.serving import forecast_weather
from src.shared.weather_features import TEAM_CODE_NORMALIZATION

GAME_KEYS = ["season", "week", "home_team", "away_team"]


@dataclass
class LiveInputs:
    schedules: pd.DataFrame
    weekly: pd.DataFrame
    team_stats: pd.DataFrame
    pbp: pd.DataFrame


@dataclass
class SpecialTeamsFrames:
    kicker: pd.DataFrame
    defense: pd.DataFrame
    kicks: pd.DataFrame
    opponent_weekly: pd.DataFrame
    schedules: pd.DataFrame
    source_status: dict
    digest: str


def before_week(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """No target-week or future outcomes may reach either history branch."""
    return frame[
        (frame["season"] < season) | ((frame["season"] == season) & (frame["week"] < week))
    ].copy()


def normalize_schedules(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ("home_team", "away_team"):
        frame[col] = frame[col].replace(TEAM_CODE_NORMALIZATION)
    return frame[frame["game_type"] == "REG"].copy()


def fetch_live_inputs(season: int, week: int, historical_weekly, historical_team) -> LiveInputs:
    schedules = normalize_schedules(nfl_source.schedules([season]))
    prior = before_week(schedules, season, week)
    if prior[["home_score", "away_score"]].isna().any().any():
        raise RuntimeError("previous-week games are incomplete; refusing partial K/DST history")
    if prior.empty:
        # No games are required at a season opener. A not-yet-created season
        # PBP release is legitimate here, whereas the same 404 in week 2 is not.
        return LiveInputs(
            schedules,
            historical_weekly.iloc[:0].copy(),
            historical_team.iloc[:0].copy(),
            pd.DataFrame(columns=nfl_source.PBP_KICKER_COLS),
        )
    with ThreadPoolExecutor(max_workers=3) as pool:
        weekly_job = pool.submit(nfl_source.weekly_data, [season])
        team_job = pool.submit(nfl_source.team_week_stats_release, season)
        pbp_job = pool.submit(nfl_source.pbp_data, [season], nfl_source.PBP_KICKER_COLS)
        weekly, teams, pbp = weekly_job.result(), team_job.result(), pbp_job.result()
    values = []
    for name, frame, team_col in (
        ("weekly", weekly, "recent_team"),
        ("team", teams, "team"),
        ("PBP", pbp, "posteam"),
    ):
        frame = before_week(frame[frame["season_type"] == "REG"], season, week)
        frame[team_col] = frame[team_col].replace(TEAM_CODE_NORMALIZATION)
        expected = {
            (int(r.season), int(r.week), team)
            for r in prior.itertuples()
            for team in (r.home_team, r.away_team)
        }
        actual = set(frame[["season", "week", team_col]].itertuples(index=False, name=None))
        missing = sorted(expected - actual)
        if missing:
            raise RuntimeError(f"{name} missing completed team-weeks: {missing}")
        values.append(frame)
    missing_pbp = set(nfl_source.PBP_KICKER_COLS) - set(values[2].columns)
    if missing_pbp:
        raise RuntimeError(f"K PBP schema incomplete: {sorted(missing_pbp)}")
    return LiveInputs(schedules, *values)


def merge_live_schedule(current: pd.DataFrame, espn: pd.DataFrame) -> pd.DataFrame:
    """ESPN supplies the slate/odds/actual venue; nflverse supplies rest/context.

    ESPN and nflverse have different game IDs. Join on the matchup, never the
    ID, and preserve the canonical nflverse ID for existing schedule consumers.
    """
    current = normalize_schedules(current).set_index(GAME_KEYS)
    upcoming = normalize_schedules(espn).set_index(GAME_KEYS)
    missing = upcoming.index.difference(current.index)
    if len(missing):
        raise RuntimeError(f"current schedule missing upcoming matchups: {list(missing)}")
    result = current.loc[upcoming.index].copy()
    for col in upcoming.columns:
        if col not in ("game_id", "home_score", "away_score"):
            # Missing ESPN odds retain the live nflverse line, never an old
            # season's carry-forward. Missing in both remains explicitly NaN.
            if col in result:
                result[col] = upcoming[col].combine_first(result[col])
            else:
                result[col] = upcoming[col]
    result[["home_score", "away_score"]] = np.nan
    return result.reset_index()


def team_schedule(schedules: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
        part = schedules[
            [
                "season",
                "week",
                f"{side}_team",
                f"{other}_team",
                "roof",
                "temp",
                "wind",
                "total_line",
                "spread_line",
                f"{side}_score",
            ]
        ].copy()
        part = part.rename(
            columns={
                f"{side}_team": "recent_team",
                f"{other}_team": "opponent_team",
                f"{side}_score": "team_points_scored",
            }
        )
        part["is_home"] = int(side == "home")
        part["spread_line"] *= sign
        part["implied_team_total"] = (part["total_line"] + part["spread_line"]) / 2
        part["is_dome"] = part["roof"].map(
            {"dome": 1.0, "closed": 1.0, "outdoors": 0.0, "open": 0.0}
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_kicker_frame(history, current, roster, schedules, season, week) -> pd.DataFrame:
    history = before_week(history, season, week)
    current = before_week(current, season, week)
    # Replace the refreshed season, including stat corrections, not just append.
    history = pd.concat([history[history["season"] != season], current], ignore_index=True)
    context = team_schedule(schedules)
    upcoming = context[(context["season"] == season) & (context["week"] == week)]
    skeleton = roster[roster["position"] == "K"].merge(
        upcoming, on="recent_team", validate="many_to_one"
    )
    if skeleton.empty:
        raise RuntimeError("no mapped active kickers for the upcoming slate")
    skeleton["season_type"] = "REG"
    skeleton["game_wind"] = skeleton["wind"]
    skeleton["game_temp"] = skeleton["temp"]
    combined = pd.concat([history, skeleton], ignore_index=True)
    combined = k_targets.compute_targets(combined)
    k_features.compute_features(combined)
    # This is the same score source used by merge_team_box_score_features, but
    # includes fresh completed games; its default historical cache ends in 2025.
    points = context.set_index(["season", "week", "recent_team"])["team_points_scored"]
    keys = pd.MultiIndex.from_frame(combined[["season", "week", "recent_team"]])
    combined["team_points_scored"] = points.reindex(keys).fillna(0).to_numpy()
    combined["_team_box_score_merged"] = True
    combined["_schedule_merged"] = True
    target = (combined["season"] == season) & (combined["week"] == week)
    combined.loc[target, [*K_CONFIG.targets, "fantasy_points"]] = np.nan
    return combined


def build_defense_frame(weekly, team_stats, schedules, season, week) -> pd.DataFrame:
    weekly = before_week(weekly, season, week)
    team_stats = before_week(team_stats, season, week)
    mask = (schedules["season"] == season) & (schedules["week"] == week)
    upcoming = schedules[mask].copy()
    upcoming[["home_score", "away_score"]] = np.nan
    schedules = pd.concat([before_week(schedules, season, week), upcoming], ignore_index=True)
    # The existing opponent rolling builders group weekly QB rows. Add one
    # outcome-free QB row per target team so shift(1) produces week-W opponent
    # features from completed games, rather than falling into league defaults.
    context = team_schedule(schedules)
    target = context[(context["season"] == season) & (context["week"] == week)]
    skeleton = target[["season", "week", "recent_team", "opponent_team"]].copy()
    skeleton["position"] = "QB"
    skeleton["player_id"] = "upcoming:" + skeleton["recent_team"]
    skeleton = skeleton.reindex(columns=weekly.columns)
    with_placeholders = pd.concat([weekly, skeleton], ignore_index=True)
    frame = dst_data.build_data(
        weekly=with_placeholders, schedules=schedules, team_stats=team_stats
    )
    frame = dst_targets.compute_targets(frame)
    dst_features.compute_features(frame)
    # Preserve the fully built live context at inference; training splits keep
    # their existing shared schedule-merge behavior.
    frame["_schedule_merged"] = True
    mask = (frame["season"] == season) & (frame["week"] == week)
    frame.loc[mask, [*DST_CONFIG.targets, "fantasy_points"]] = np.nan
    return frame


def prepare_special_teams(
    season, week, roster, espn_schedule, splits, historical_kicks
) -> SpecialTeamsFrames:
    root = Path(CACHE_DIR)
    suffix = f"{SEASONS[0]}_{SEASONS[-1]}"
    weekly = pd.read_parquet(root / f"weekly_{suffix}.parquet")
    team_stats = load_team_week_stats(SEASONS, cache_dir=CACHE_DIR)
    old_schedule = normalize_schedules(pd.read_parquet(root / f"schedules_{suffix}.parquet"))
    live = fetch_live_inputs(season, week, weekly, team_stats)
    upcoming = merge_live_schedule(live.schedules, espn_schedule)
    upcoming, weather_status = forecast_weather.enrich_forecasts(upcoming)
    # Keep only completed history and the target slate. Later scheduled games
    # must not become zero-outcome rows in the defense builder.
    past_schedule = pd.concat(
        [old_schedule[old_schedule["season"] != season], before_week(live.schedules, season, week)],
        ignore_index=True,
    )
    past_schedule = before_week(past_schedule, season, week)
    schedules = pd.concat([past_schedule, upcoming], ignore_index=True).drop_duplicates(
        GAME_KEYS, keep="last"
    )
    for col in ("temp", "wind"):
        if col not in schedules:
            schedules[col] = np.nan
    weekly = before_week(weekly, season, week)
    weekly = pd.concat([weekly[weekly["season"] != season], live.weekly], ignore_index=True)
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    team_stats = before_week(team_stats, season, week)
    team_stats = pd.concat(
        [team_stats[team_stats["season"] != season], live.team_stats], ignore_index=True
    )

    k_history = pd.concat(splits["K"], ignore_index=True)
    k_current = k_history.iloc[:0].copy()
    if not live.weekly.empty:
        k_current = k_data.load_data(
            seasons=[season], weekly=live.weekly, schedules=live.schedules, pbp=live.pbp
        )
    kicker = build_kicker_frame(k_history, k_current, roster, schedules, season, week)
    fresh_kicks = (
        k_data.reconstruct_kicker_kicks_from_pbp([season], pbp=live.pbp)
        if not live.pbp.empty
        else historical_kicks.iloc[:0].copy()
    )
    # load_kicks adds home/away from actual weekly rows. Drop the old home field
    # so its validated one-to-one merge cannot acquire suffix collisions.
    if not fresh_kicks.empty:
        fresh_kicks = k_data.load_kicks(kicker, kicks_df=fresh_kicks)
    kicks = before_week(historical_kicks, season, week)
    kicks = pd.concat([kicks[kicks["season"] != season], fresh_kicks], ignore_index=True)
    defense = build_defense_frame(weekly, team_stats, schedules, season, week)
    source_status = {
        "retrieved_at": datetime.now(UTC).isoformat(),
        "season": season,
        "history_before_week": week,
        "completed_team_weeks": len(live.team_stats),
        "weather": weather_status,
        "missing_kicker_teams": sorted(
            set(team_schedule(upcoming)["recent_team"])
            - set(roster.loc[roster["position"] == "K", "recent_team"])
        ),
    }
    # Hash actual input content, not row counts: stat corrections and forecast
    # updates must invalidate a same-week artifact even with unchanged odds.
    digest = hashlib.sha256()
    for frame in (live.weekly, live.team_stats, live.pbp, upcoming):
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return SpecialTeamsFrames(
        kicker, defense, kicks, weekly, schedules, source_status, digest.hexdigest()
    )
