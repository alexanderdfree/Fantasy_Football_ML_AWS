"""Per-position contracts for synthetic player-history generation and replay.

Every column group is derived from the position's ``POSITION_CONFIG`` at import
time, so a whitelist change is a configuration change rather than a drifting
copy. Schema version 2 populates QB; further positions are later slices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.features.engineer import flatten_include_features
from src.qb.config import POSITION_CONFIG as _QB_CONFIG


@dataclass(frozen=True)
class PositionHistorySchema:
    """Columns, validity relations and provenance paths for one position.

    ``sequence_coupled_context`` names the static feature columns whose value
    describes the real prior sequence (calendar position, rest, games to date);
    they are held at the forecast game's real value and disclosed as such.
    ``check_columns`` lists what the ``checks`` callables read.
    """

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
    check_columns: tuple[str, ...]
    sequence_coupled_context: tuple[str, ...]
    scoring_scope: str
    code_paths: tuple[str, ...]

    def __post_init__(self):
        known = set(self.history_columns) | set(self.targets)
        referenced = (
            set(self.count_columns)
            | set(self.must_observe)
            | {column for relation in self.relations for column in relation}
            | {column for column, _, _ in self.bounded_columns}
            | set(self.check_columns)
        )
        if not referenced <= known:
            raise ValueError(
                f"{self.position} schema references columns outside its history and "
                f"targets: {sorted(referenced - known)}"
            )
        if not set(self.sequence_coupled_context) <= set(self.feature_columns):
            raise ValueError(f"{self.position} sequence-coupled context must be feature columns")

    @property
    def validated_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.history_columns, *self.targets]))


def _qb_interceptions_within_incompletions(frame: pd.DataFrame) -> str | None:
    if (frame["interceptions"] > frame["attempts"] - frame["completions"]).any():
        return "interceptions exceed incomplete attempts"
    return None


QB_SCHEMA = PositionHistorySchema(
    position="QB",
    identity_columns=("position", "season_type", "recent_team", "opponent_team"),
    history_columns=tuple(_QB_CONFIG.attn_history_stats),
    targets=tuple(_QB_CONFIG.targets),
    feature_columns=tuple(flatten_include_features(_QB_CONFIG.include_features)),
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
    check_columns=("interceptions", "attempts", "completions"),
    sequence_coupled_context=(
        "week",
        "days_rest",
        "season_starts_to_date",
        "is_returning_from_absence",
        "rookie_early",
    ),
    scoring_scope="QB projected components only; excludes receiving and two-point conversions",
    code_paths=(
        "features/engineer.py",
        "qb/config.py",
        "config.py",
        "shared/aggregate_targets.py",
    ),
)

POSITION_HISTORY_SCHEMAS: dict[str, PositionHistorySchema] = {"QB": QB_SCHEMA}


def position_schema(position: str) -> PositionHistorySchema:
    return POSITION_HISTORY_SCHEMAS[position]
