"""The position schema registry is a projection of each production configuration."""

import dataclasses
from importlib import import_module

import pytest

from src.analysis.synthetic_history_schema import (
    POSITION_HISTORY_SCHEMAS,
    SEQUENCE_COUPLED_CANDIDATES,
    position_schema,
)
from src.features.engineer import flatten_include_features

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("position", sorted(POSITION_HISTORY_SCHEMAS))
def test_registry_matches_the_production_configuration(position):
    schema = position_schema(position)
    config = import_module(f"src.{position.lower()}.config").POSITION_CONFIG
    features = import_module(f"src.{position.lower()}.features").get_feature_columns()
    assert schema.history_columns == tuple(config.attn_history_stats)
    assert schema.targets == tuple(config.targets)
    assert schema.feature_columns == tuple(features)
    assert schema.feature_columns == tuple(flatten_include_features(config.include_features))
    assert schema.max_history_games == config.attn_max_seq_len
    assert schema.sequence_coupled_context == tuple(
        c for c in SEQUENCE_COUPLED_CANDIDATES if c in features
    )
    known = set(schema.validated_columns)
    assert set(schema.transformable_columns) | set(schema.opaque_columns) <= known
    assert set(schema.must_observe) <= known and set(schema.count_columns) <= known
    assert schema.team_accounting_columns <= set(schema.history_columns)
    assert not set(schema.transformable_columns) & set(schema.opaque_columns)
    assert schema.transform_support == {"scale": None, "set_history_ppg": None}
    assert schema.default_ppg_band[0] < schema.default_ppg_band[1]


def test_registry_rejects_foreign_or_overlapping_declarations():
    schema = position_schema("RB")
    with pytest.raises(ValueError, match="outside its history and targets"):
        dataclasses.replace(schema, relations=(("carries", "not_a_column"),))
    with pytest.raises(ValueError, match="sequence-coupled context"):
        dataclasses.replace(schema, sequence_coupled_context=("carries",))
    with pytest.raises(ValueError, match="column groups overlap"):
        dataclasses.replace(schema, opaque_columns=(*schema.opaque_columns, "carries"))


def test_position_group_shares_are_opaque_not_transformable():
    for position, columns in {
        "RB": ("game_carry_share", "game_target_share", "game_carry_hhi", "game_target_hhi"),
        "WR": ("game_target_share", "game_target_hhi", "game_opportunity_index"),
        "TE": ("game_target_share", "game_target_hhi", "game_opportunity_index"),
    }.items():
        schema = position_schema(position)
        assert set(columns) <= set(schema.opaque_columns)
        assert not set(columns) & set(schema.transformable_columns)
