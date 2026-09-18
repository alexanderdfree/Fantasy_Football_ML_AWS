"""Per-position contracts for synthetic player-history generation and replay.

Every column group is derived from the position's ``POSITION_CONFIG`` at import
time, so a whitelist change is a configuration change rather than a drifting
copy. Schema version 2 populates QB; further positions are later slices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.qb.config import POSITION_CONFIG as _QB_CONFIG
from src.qb.features import get_feature_columns as _qb_feature_columns


@dataclass(frozen=True)
class PositionHistorySchema:
    """Columns, validity relations and provenance paths for one position."""

    position: str
    identity_columns: tuple[str, ...]
    history_columns: tuple[str, ...]
    targets: tuple[str, ...]
    feature_columns: tuple[str, ...]
    max_history_games: int
    count_columns: tuple[str, ...]
    must_observe: tuple[str, ...]
    relations: tuple[tuple[str, str], ...]
    bounded_columns: tuple[tuple[str, float, float], ...]
    checks: tuple[Callable[[pd.DataFrame], str | None], ...]
    scoring_scope: str
    code_paths: tuple[str, ...]


def _qb_interceptions_within_incompletions(frame: pd.DataFrame) -> str | None:
    if (frame["interceptions"] > frame["attempts"] - frame["completions"]).any():
        return "interceptions exceed incomplete attempts"
    return None


QB_SCHEMA = PositionHistorySchema(
    position="QB",
    identity_columns=("position", "season_type", "recent_team", "opponent_team"),
    history_columns=tuple(_QB_CONFIG.attn_history_stats),
    targets=tuple(_QB_CONFIG.targets),
    feature_columns=tuple(_qb_feature_columns()),
    max_history_games=int(_QB_CONFIG.attn_max_seq_len),
    count_columns=(
        "attempts",
        "completions",
        "carries",
        "passing_tds",
        "rushing_tds",
        "interceptions",
        "fumbles_lost",
        "sacks",
    ),
    must_observe=(*_QB_CONFIG.targets, "attempts", "completions", "carries"),
    relations=(
        ("completions", "attempts"),
        ("passing_tds", "completions"),
        ("rushing_tds", "carries"),
        ("carries", "team_rush_attempts"),
    ),
    bounded_columns=(("snap_pct_raw", 0.0, 1.0), ("qbr_total", 0.0, 100.0)),
    checks=(_qb_interceptions_within_incompletions,),
    scoring_scope="QB projected components only; excludes receiving and two-point conversions",
    code_paths=(
        "features/engineer.py",
        "qb/config.py",
        "qb/features.py",
        "config.py",
        "shared/aggregate_targets.py",
    ),
)

POSITION_HISTORY_SCHEMAS: dict[str, PositionHistorySchema] = {"QB": QB_SCHEMA}


def position_schema(position: str) -> PositionHistorySchema:
    try:
        return POSITION_HISTORY_SCHEMAS[position]
    except KeyError:
        raise ValueError(
            f"schema version 2 supports {sorted(POSITION_HISTORY_SCHEMAS)} histories only"
        ) from None
