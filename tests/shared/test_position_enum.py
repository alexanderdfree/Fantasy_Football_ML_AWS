"""Coverage tests for ``src/shared/position.py``.

The ``Position`` enum is the canonical source of truth for position codes.
String-valued so existing serialisation (URL params, S3 keys, JSON
payloads) keeps working unchanged. PR 6 wired it into the registry's
``ALL_POSITIONS``, the aggregator's ``POSITION_TARGET_MAP`` keys, and
the ``PositionConfig.name`` validator.
"""

from __future__ import annotations

import pytest

from src.shared.position import Position
from src.shared.position_config import PositionConfig

EXPECTED_VALUES = ["QB", "RB", "WR", "TE", "K", "DST"]


# --------------------------------------------------------------------------
# Enum shape
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_position_values_match_canonical_set():
    assert [p.value for p in Position] == EXPECTED_VALUES


@pytest.mark.unit
def test_position_values_helper_returns_strings():
    assert Position.values() == EXPECTED_VALUES
    assert all(isinstance(v, str) for v in Position.values())


@pytest.mark.unit
@pytest.mark.parametrize("value", EXPECTED_VALUES)
def test_position_lookup_by_value(value: str):
    assert Position(value).value == value


@pytest.mark.unit
def test_position_raises_on_unknown_value():
    with pytest.raises(ValueError):
        Position("FOO")


# --------------------------------------------------------------------------
# Backward-compatibility with bare strings — same hash, same equality, same str()
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("value", EXPECTED_VALUES)
def test_position_compares_equal_to_string(value: str):
    """``Position(str, Enum)`` lets bare strings continue to flow through
    code that has been refactored to take a ``Position`` parameter."""
    assert Position(value) == value
    assert value == Position(value)


@pytest.mark.unit
@pytest.mark.parametrize("value", EXPECTED_VALUES)
def test_position_hashes_same_as_string(value: str):
    """Required so dicts keyed on either form work interchangeably —
    ``my_dict[Position.QB]`` and ``my_dict["QB"]`` look up the same slot.
    """
    assert hash(Position(value)) == hash(value)
    d = {value: "x"}
    assert d[Position(value)] == "x"


@pytest.mark.unit
@pytest.mark.parametrize("value", EXPECTED_VALUES)
def test_position_str_returns_value_not_qualified_name(value: str):
    """``str(Position.QB)`` must produce ``"QB"`` (not ``"Position.QB"``)
    so f-strings, logging, and URL/S3 key building keep working unchanged.
    """
    assert str(Position(value)) == value
    assert f"{Position(value)}" == value


# --------------------------------------------------------------------------
# PositionConfig name validation (added in PR 6)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_position_config_rejects_unknown_position_name():
    """A typo in a per-position ``POSITION_CONFIG`` call should fail at
    import time, not at downstream-dict-lookup time."""
    with pytest.raises(ValueError):
        PositionConfig(
            name="QBB",
            targets=["x"],
            specific_features=[],
            ridge_alpha_grids={},
            loss_weights={"x": 1.0},
            head_losses={"x": "huber"},
            huber_deltas={"x": 1.0},
        )


@pytest.mark.unit
@pytest.mark.parametrize("name", EXPECTED_VALUES)
def test_position_config_accepts_canonical_position_names(name: str):
    """Each of the six canonical position codes constructs cleanly."""
    pc = PositionConfig(
        name=name,
        targets=["x"],
        specific_features=[],
        ridge_alpha_grids={},
        loss_weights={"x": 1.0},
        head_losses={"x": "huber"},
        huber_deltas={"x": 1.0},
    )
    assert pc.name == name


# --------------------------------------------------------------------------
# aggregate_targets.POSITION_TARGET_MAP — typo guard
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_position_target_map_keys_are_valid_positions():
    from src.shared.aggregate_targets import POSITION_TARGET_MAP

    for key in POSITION_TARGET_MAP:
        # Will raise ValueError if not a real Position.
        Position(key)


# --------------------------------------------------------------------------
# Drift guards: hard-coded position copies must track Position.values()
# (same pattern as tests/scripts/test_scope_positions.py — the copies stay
# plain strings to avoid retrain-triggering src edits; the tests pin them)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_src_config_positions_matches_position_enum():
    """GH #1428: ``src.config.POSITIONS`` is the global raw-row allowlist
    (``src/data/preprocessing.py`` filters on it pre-split), kept as an
    independent hard-coded copy of the canonical ``Position`` enum. A
    position added to the enum but not there would be silently dropped
    before any model-specific loading — pin the copy to the enum,
    list-order-exact."""
    from src.config import POSITIONS

    assert Position.values() == POSITIONS


@pytest.mark.unit
def test_run_pipeline_factory_tiny_allowlist_matches_position_enum():
    """GH #1429: ``run_pipeline_factory._ALL_POSITIONS`` gates the
    ``python -m src.<pos>.run_pipeline --tiny`` CLI path. A position added
    to the enum/registry would dispatch fine through normal registry lookup
    while ``--tiny`` raised "Unknown position" — pin the private allowlist
    to the enum."""
    from src.shared import run_pipeline_factory

    assert tuple(Position.values()) == run_pipeline_factory._ALL_POSITIONS
