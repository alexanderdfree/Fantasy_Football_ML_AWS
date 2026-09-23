"""The position schema registry is a projection of each production configuration."""

import dataclasses
from importlib import import_module

import pytest

from src.analysis.synthetic_history_schema import (
    POSITION_HISTORY_SCHEMAS,
    TE_SCHEMA,
    WR_SCHEMA,
    position_schema,
)

pytestmark = pytest.mark.unit

# Static columns whose value describes the real prior sequence; pinned per
# position so a whitelist change is a visible registry change.
SEQUENCE_COUPLED = {
    "QB": (
        "week",
        "days_rest",
        "season_starts_to_date",
        "is_returning_from_absence",
        "rookie_early",
    ),
    "RB": ("week", "days_rest", "rest_advantage", "career_carries"),
    "WR": ("week", "days_rest", "is_returning_from_absence"),
    "TE": ("week", "days_rest", "is_returning_from_absence"),
    "DST": ("week", "rest_days"),
}


@pytest.mark.parametrize("position", sorted(POSITION_HISTORY_SCHEMAS))
def test_registry_matches_the_production_configuration(position):
    schema = position_schema(position)
    config = import_module(f"src.{position.lower()}.config").POSITION_CONFIG
    features = import_module(f"src.{position.lower()}.features").get_feature_columns()
    assert schema.history_columns == tuple(config.attn_history_stats)
    assert schema.targets == tuple(config.targets)
    assert schema.feature_columns == tuple(features)
    assert schema.max_history_games == config.attn_max_seq_len
    assert schema.sequence_coupled_context == SEQUENCE_COUPLED[position]
    assert set(schema.sequence_coupled_context) <= set(features)
    known = set(schema.validated_columns)
    assert set(schema.targets) <= set(schema.must_observe) <= known
    assert set(schema.transformable_columns) | set(schema.opaque_columns) <= known
    assert set(schema.count_columns) <= known
    assert schema.team_accounting_columns <= set(schema.history_columns)
    assert not set(schema.transformable_columns) & set(schema.opaque_columns)
    assert set(schema.transform_support) == {"scale", "set_history_ppg"}
    assert schema.transform_support["scale"] is None
    assert (schema.transform_support["set_history_ppg"] is None) == (position != "DST")
    assert set(schema.opponent_history_columns).isdisjoint(schema.history_columns)
    if position == "DST":
        assert schema.donor_identity == "team" and schema.fantasy_points_policy == "assert_equal"
        assert schema.opponent_history_columns == tuple(config.opp_attn_history_stats)
        assert schema.opponent_max_history_games == config.opp_attn_max_seq_len == 17
    else:
        assert schema.donor_identity == "player" and not schema.opponent_history_columns


def test_registry_rejects_foreign_overlapping_or_unobserved_declarations():
    schema = position_schema("RB")
    with pytest.raises(ValueError, match="outside its history and targets"):
        dataclasses.replace(schema, relations=(("carries", "not_a_column"),))
    with pytest.raises(ValueError, match="sequence-coupled context"):
        dataclasses.replace(schema, sequence_coupled_context=("carries",))
    with pytest.raises(ValueError, match="column groups overlap"):
        dataclasses.replace(schema, opaque_columns=(*schema.opaque_columns, "carries"))
    with pytest.raises(ValueError, match="must observe every target"):
        dataclasses.replace(schema, must_observe=("carries",))
    with pytest.raises(ValueError, match="donor_identity"):
        dataclasses.replace(schema, donor_identity="franchise")
    with pytest.raises(ValueError, match="fantasy_points_policy"):
        dataclasses.replace(schema, fantasy_points_policy="ignore")
    with pytest.raises(ValueError, match="opponent stream needs columns and a length"):
        dataclasses.replace(schema, opponent_history_columns=("off_pass_yards",))
    with pytest.raises(ValueError, match="overlap the history"):
        dataclasses.replace(
            schema, opponent_history_columns=("carries",), opponent_max_history_games=17
        )


def test_shared_receiver_declarations_are_not_aliased():
    assert WR_SCHEMA.team_accounting == TE_SCHEMA.team_accounting
    assert WR_SCHEMA.team_accounting is not TE_SCHEMA.team_accounting
    assert WR_SCHEMA.transform_support is not TE_SCHEMA.transform_support


def test_position_group_shares_are_opaque_not_transformable():
    for position, columns in {
        "RB": ("game_carry_share", "game_target_share", "game_carry_hhi", "game_target_hhi"),
        "WR": ("game_target_share", "game_target_hhi", "game_opportunity_index"),
        "TE": ("game_target_share", "game_target_hhi", "game_opportunity_index"),
    }.items():
        schema = position_schema(position)
        assert set(columns) <= set(schema.opaque_columns)
        assert not set(columns) & set(schema.transformable_columns)
