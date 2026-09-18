"""Per-position contracts for synthetic player-history generation and replay.

Every column group is derived from the position's ``POSITION_CONFIG`` at import
time, so a whitelist change is a configuration change rather than a drifting
copy. The skill positions (QB, RB, WR, TE) share one flat history structure
and one loading path; DST and K are later slices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module

import pandas as pd

from src.features.engineer import flatten_include_features

BoundFn = Callable[[pd.DataFrame], pd.Series]

SKILL_IDENTITY_COLUMNS = ("position", "season_type", "recent_team", "opponent_team")
SKILL_HELD_CONTEXT = (
    "implied_team_total",
    "implied_opp_total",
    "is_home",
    "days_rest",
    "opp_team_points_scored",
)
SKILL_TRANSFORM_SUPPORT = {"scale": None, "set_history_ppg": None}
SEQUENCE_COUPLED_CANDIDATES = (
    "week",
    "days_rest",
    "season_starts_to_date",
    "is_returning_from_absence",
    "rookie_early",
)


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
    signals governed only by the recipe's opaque-signal policy (including
    per-game shares whose position-group denominators the history does not
    carry), team totals that move with a player stat through
    ``team_accounting``, and held game context. ``transform_support`` maps an
    op to ``None`` (supported) or the reason it is declined.
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
    default_ppg_band: tuple[float, float] = (0.0, 0.0)

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


def _touchdowns_within_team_points(*td_columns: str) -> tuple[str, BoundFn]:
    def violated(frame: pd.DataFrame) -> pd.Series:
        return 6.0 * sum(frame[c] for c in td_columns) > frame["team_points_scored"]

    return "touchdowns exceed team points (six per touchdown)", violated


def _skill_schema(position: str, **declarations) -> PositionHistorySchema:
    """Bind the production whitelists of a skill position to its declarations."""
    config = import_module(f"src.{position.lower()}.config").POSITION_CONFIG
    feature_columns = tuple(flatten_include_features(config.include_features))
    return PositionHistorySchema(
        position=position,
        identity_columns=SKILL_IDENTITY_COLUMNS,
        history_columns=tuple(config.attn_history_stats),
        targets=tuple(config.targets),
        feature_columns=feature_columns,
        max_history_games=int(config.attn_max_seq_len),
        sequence_coupled_context=tuple(
            c for c in SEQUENCE_COUPLED_CANDIDATES if c in feature_columns
        ),
        code_paths=(
            "features/engineer.py",
            f"{position.lower()}/config.py",
            "config.py",
            "shared/aggregate_targets.py",
        ),
        held_context_columns=SKILL_HELD_CONTEXT,
        transform_support=dict(SKILL_TRANSFORM_SUPPORT),
        **declarations,
    )


def _unit_interval(*columns: str) -> tuple[tuple[str, float, float], ...]:
    return tuple((column, 0.0, 1.0) for column in columns)


QB_SCHEMA = _skill_schema(
    "QB",
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
    must_observe=(
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "interceptions",
        "fumbles_lost",
        "attempts",
        "completions",
        "carries",
    ),
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
        _touchdowns_within_team_points("passing_tds", "rushing_tds"),
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
    scoring_scope="QB projected components only; excludes receiving and two-point conversions",
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
    default_ppg_band=(15.0, 30.0),
)

RB_SCHEMA = _skill_schema(
    "RB",
    count_columns=(
        "carries",
        "targets",
        "receptions",
        "rushing_tds",
        "receiving_tds",
        "fumbles_lost",
        "rushing_first_downs",
        "receiving_first_downs",
        "redzone_carries",
        "redzone_targets",
        "inside10_carries",
        "inside5_carries",
    ),
    must_observe=(
        "rushing_tds",
        "receiving_tds",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "fumbles_lost",
        "carries",
        "targets",
    ),
    relations=(
        ("receptions", "targets"),
        ("receiving_tds", "receptions"),
        ("rushing_tds", "carries"),
        ("rushing_first_downs", "carries"),
        ("receiving_first_downs", "receptions"),
        ("inside5_carries", "inside10_carries"),
        ("inside10_carries", "redzone_carries"),
        ("redzone_carries", "carries"),
        ("redzone_targets", "targets"),
        ("carries", "team_rush_attempts"),
        ("targets", "team_pass_attempts"),
        ("receptions", "team_completions"),
    ),
    bounded_columns=_unit_interval(
        "snap_pct_raw",
        "game_carry_share",
        "game_target_share",
        "game_carry_hhi",
        "game_target_hhi",
        "redzone_target_share",
    ),
    derived_checks=(_touchdowns_within_team_points("rushing_tds", "receiving_tds"),),
    derived_caps=(),
    check_columns=("rushing_tds", "receiving_tds", "team_points_scored"),
    scoring_scope="RB projected components only; excludes passing and two-point conversions",
    transformable_columns=(
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
        "carries",
        "targets",
        "receptions",
        "fumbles_lost",
        "snap_pct_raw",
        "rushing_first_downs",
        "receiving_first_downs",
        "redzone_carries",
        "redzone_targets",
        "inside10_carries",
        "inside5_carries",
    ),
    opaque_columns=(
        "rush_yards_gained_exp",
        "rush_touchdown_exp",
        "rec_yards_gained_exp",
        "rec_touchdown_exp",
        "receptions_exp",
        "game_carry_share",
        "game_target_share",
        "game_carry_hhi",
        "game_target_hhi",
        "redzone_target_share",
    ),
    team_accounting={
        "rushing_yards": ("team_rushing_yards", 1.0),
        "receiving_yards": ("team_passing_yards", 1.0),
        "carries": ("team_rush_attempts", 1.0),
        "targets": ("team_pass_attempts", 1.0),
        "receptions": ("team_completions", 1.0),
        "rushing_tds": ("team_points_scored", 6.0),
        "receiving_tds": ("team_points_scored", 6.0),
        "fumbles_lost": ("team_turnovers", 1.0),
    },
    default_ppg_band=(8.0, 20.0),
)

_RECEIVER_DECLARATIONS = dict(
    count_columns=(
        "targets",
        "receptions",
        "carries",
        "receiving_tds",
        "rushing_tds",
        "fumbles_lost",
        "redzone_targets",
    ),
    must_observe=(
        "receiving_tds",
        "receiving_yards",
        "receptions",
        "fumbles_lost",
        "targets",
        "carries",
    ),
    relations=(
        ("receptions", "targets"),
        ("receiving_tds", "receptions"),
        ("rushing_tds", "carries"),
        ("redzone_targets", "targets"),
        ("targets", "team_pass_attempts"),
        ("carries", "team_rush_attempts"),
    ),
    bounded_columns=_unit_interval(
        "snap_pct_raw",
        "redzone_target_share",
        "game_target_share",
        "game_target_hhi",
        "game_opportunity_index",
    ),
    derived_checks=(_touchdowns_within_team_points("receiving_tds", "rushing_tds"),),
    derived_caps=(),
    check_columns=("receiving_tds", "rushing_tds", "team_points_scored"),
    transformable_columns=(
        "receiving_yards",
        "rushing_yards",
        "receiving_tds",
        "rushing_tds",
        "targets",
        "receptions",
        "fumbles_lost",
        "carries",
        "snap_pct_raw",
        "redzone_targets",
    ),
    opaque_columns=(
        "rec_yards_gained_exp",
        "rec_touchdown_exp",
        "receptions_exp",
        "rec_first_down_exp",
        "redzone_target_share",
        "game_target_share",
        "game_target_hhi",
        "game_opportunity_index",
    ),
    team_accounting={
        "receiving_yards": ("team_passing_yards", 1.0),
        "targets": ("team_pass_attempts", 1.0),
        "carries": ("team_rush_attempts", 1.0),
        "receiving_tds": ("team_points_scored", 6.0),
        "rushing_tds": ("team_points_scored", 6.0),
    },
)

WR_SCHEMA = _skill_schema(
    "WR",
    scoring_scope="WR projected components only; excludes rushing, passing and two-point conversions",
    default_ppg_band=(8.0, 20.0),
    **_RECEIVER_DECLARATIONS,
)

TE_SCHEMA = _skill_schema(
    "TE",
    scoring_scope="TE projected components only; excludes rushing, passing and two-point conversions",
    default_ppg_band=(5.0, 15.0),
    **_RECEIVER_DECLARATIONS,
)

POSITION_HISTORY_SCHEMAS: dict[str, PositionHistorySchema] = {
    "QB": QB_SCHEMA,
    "RB": RB_SCHEMA,
    "WR": WR_SCHEMA,
    "TE": TE_SCHEMA,
}


def position_schema(position: str) -> PositionHistorySchema:
    return POSITION_HISTORY_SCHEMAS[position]
