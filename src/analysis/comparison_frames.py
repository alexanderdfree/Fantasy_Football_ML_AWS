"""Row preparation for offline expert comparisons; no fetching or model fitting."""

import numpy as np
import pandas as pd

from src.shared.comparison_scoring import comparison_actuals as comparison_truth
from src.shared.comparison_scoring import comparison_model_totals
from src.shared.evaluation_cohorts import KEYS, regular_season_rows


def comparison_actuals(frame: pd.DataFrame, position: str, scoring="ppr") -> pd.DataFrame:
    """Use regular-season raw component truth, never a full-fantasy fallback."""
    out = comparison_model_totals(regular_season_rows(frame), position, scoring)
    if "position" in out:
        out = out.loc[out["position"].eq(position)].copy()
    out = out.dropna(subset=KEYS)
    out["player_id"] = out["player_id"].astype(str)
    for col in ("season", "week"):
        out[col] = out[col].astype(int)
    if out.duplicated(KEYS).any():
        raise ValueError("Duplicate actual player-weeks in expert comparison")
    out["fantasy_points"] = comparison_truth(out, position, scoring)
    return out.loc[out["fantasy_points"].notna()].copy()


def common_forecast_frames(frames: list[pd.DataFrame], prediction="pred_total"):
    """Intersect finite player-weeks across every available displayed source.

    Empty sources remain unavailable without deleting otherwise valid sources.
    A source with any forecasts participates in the same intersection, including
    when its rows do not overlap another source at all.
    """
    cleaned = []
    common = None
    for frame in frames:
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate forecast player-weeks in expert comparison")
        values = frame[["fantasy_points", prediction]].apply(pd.to_numeric, errors="coerce")
        clean = frame.loc[np.isfinite(values).all(axis=1)].copy()
        cleaned.append(clean)
        if not clean.empty:
            keys = pd.MultiIndex.from_frame(clean[KEYS])
            common = keys if common is None else common.intersection(keys, sort=False)
    return [
        frame.loc[pd.MultiIndex.from_frame(frame[KEYS]).isin(common)].copy()
        if common is not None
        else frame.iloc[:0].copy()
        for frame in cleaned
    ]


def prior_component_means(frames, position: str) -> pd.Series:
    """Prior-season tier scores use the same components as the current actuals."""
    past = pd.concat([f for f in frames if f is not None], ignore_index=True)
    fumbles = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
    if "fumbles_lost" not in past and all(col in past for col in fumbles):
        past["fumbles_lost"] = past[list(fumbles)].fillna(0).sum(axis=1)
    past = comparison_actuals(past.drop_duplicates(KEYS), position)
    past["season"] += 1
    return past.groupby(["player_id", "season"])["fantasy_points"].mean()


def position_frames(position: str, offense_frames):
    """Supply native K/DST components for artifact evaluation and prior tiers."""
    if position == "K":
        from src.k.data import load_data, season_split
        from src.k.features import compute_features
        from src.k.targets import compute_targets

        frame = compute_targets(load_data())
        compute_features(frame)
        return season_split(frame)
    if position == "DST":
        from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
        from src.dst.data import build_data
        from src.dst.features import compute_features
        from src.dst.targets import compute_targets

        frame = compute_targets(build_data())
        compute_features(frame)
        return tuple(
            frame.loc[frame.season.isin(seasons)].copy()
            for seasons in (TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)
        )
    return offense_frames
