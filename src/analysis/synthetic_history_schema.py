"""Per-position contracts for synthetic player-history generation and replay.

Every column group is derived from the position's ``POSITION_CONFIG`` at import
time, so a whitelist change is a configuration change rather than a drifting
copy. Schema version 2 populates QB; further positions are later slices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from src.features.engineer import flatten_include_features
from src.qb.config import POSITION_CONFIG as _QB_CONFIG

BoundFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class PositionHistorySchema:
    """Columns, validity relations, transform declarations and provenance paths.

    ``sequence_coupled_context`` names the static feature columns whose value
    describes the real prior sequence (calendar position, rest, games to date);
    they are held at the forecast game's real value and disclosed as such.
    ``derived_checks`` are validity rules that are not a plain ``a <= b`` pair;
    ``derived_caps`` bound a count that integer rounding may push over such a
    rule. ``check_columns`` lists what those callables read.

    Transform declarations partition the history columns: ``transformable``
    production/usage stats an op may rewrite, ``opaque`` externally modeled
    signals governed only by the recipe's opaque-signal policy, team totals
    that move with a player stat through ``team_accounting``, and held game
    context. ``transform_support`` maps an op to ``None`` (supported) or the
    reason it is declined.
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
    derived_checks: tuple[tuple[str, BoundFn], ...]
    derived_caps: tuple[tuple[str, BoundFn], ...]
    check_columns: tuple[str, ...]
    sequence_coupled_context: tuple[str, ...]
    scoring_scope: str
    code_paths: tuple[str, ...]
    transformable_columns: tuple[str, ...] = ()
    opaque_columns: tuple[str, ...] = ()
    team_accounting: dict[str, tuple[str, float]] = field(default_factory=dict)
    held_context_columns: tuple[str, ...] = ()
    transform_support: dict[str, str | None] = field(default_factory=dict)

    def __post_init__(self):
        known = set(self.history_columns) | set(self.targets)
        referenced = (
            set(self.count_columns)
            | set(self.must_observe)
            | {column for relation in self.relations for column in relation}
            | {column for column, _, _ in self.bounded_columns}
            | {column for column, _ in self.derived_caps}
            | set(self.check_columns)
            | set(self.transformable_columns)
            | set(self.opaque_columns)
            | set(self.team_accounting)
            | {column for column, _ in self.team_accounting.values()}
            | set(self.held_context_columns)
        )
        if not referenced <= known:
            raise ValueError(
                f"{self.position} schema references columns outside its history and "
                f"targets: {sorted(referenced - known)}"
            )
        if not set(self.sequence_coupled_context) <= set(self.feature_columns):
            raise ValueError(f"{self.position} sequence-coupled context must be feature columns")
        groups = (
            set(self.transformable_columns),
            set(self.opaque_columns),
            self.team_accounting_columns,
            set(self.held_context_columns),
        )
        if sum(len(group) for group in groups) != len(set().union(*groups)):
            raise ValueError(f"{self.position} transform column groups overlap")

    @property
    def validated_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.history_columns, *self.targets]))

    @property
    def team_accounting_columns(self) -> set[str]:
        return {column for column, _ in self.team_accounting.values()}


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
    derived_checks=(
        (
            "interceptions exceed incomplete attempts",
            lambda frame: frame["interceptions"] > frame["attempts"] - frame["completions"],
        ),
        (
            "touchdowns exceed team points (six per touchdown)",
            lambda frame: (
                6.0 * (frame["passing_tds"] + frame["rushing_tds"]) > frame["team_points_scored"]
            ),
        ),
    ),
    derived_caps=(("interceptions", lambda frame: frame["attempts"] - frame["completions"]),),
    check_columns=(
        "interceptions",
        "attempts",
        "completions",
        "passing_tds",
        "rushing_tds",
        "team_points_scored",
    ),
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
    transformable_columns=(
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "attempts",
        "completions",
        "carries",
        "interceptions",
        "fumbles_lost",
        "snap_pct_raw",
        "sacks",
        "sack_yards",
    ),
    opaque_columns=(
        "pass_yards_gained_exp",
        "pass_touchdown_exp",
        "pass_interception_exp",
        "rush_yards_gained_exp",
        "rush_touchdown_exp",
        "qbr_total",
        "pts_added",
    ),
    team_accounting={
        "rushing_yards": ("team_rushing_yards", 1.0),
        "carries": ("team_rush_attempts", 1.0),
        "passing_tds": ("team_points_scored", 6.0),
        "rushing_tds": ("team_points_scored", 6.0),
    },
    held_context_columns=(
        "implied_team_total",
        "implied_opp_total",
        "is_home",
        "days_rest",
        "opp_team_points_scored",
    ),
    transform_support={"scale": None, "set_history_ppg": None},
)

POSITION_HISTORY_SCHEMAS: dict[str, PositionHistorySchema] = {"QB": QB_SCHEMA}


def position_schema(position: str) -> PositionHistorySchema:
    return POSITION_HISTORY_SCHEMAS[position]
