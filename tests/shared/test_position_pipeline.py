"""Coverage tests for ``src/shared/position_pipeline.py``.

The factory takes a position's ``POSITION_CONFIG`` and produces the runtime
CONFIG dict that ``src/shared/pipeline.py::run_pipeline`` consumes. PR 5
added a missing-key validator that runs at construction time so a
misconfigured position fails at *build* time rather than partway through
training.
"""

from __future__ import annotations

import importlib

import pytest

from src.shared.position_config import PositionConfig
from src.shared.position_pipeline import (
    REQUIRED_PIPELINE_CFG_KEYS,
    PipelineConfigError,
    build_pipeline_config,
    validate_pipeline_config,
)

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]


# --------------------------------------------------------------------------
# build_pipeline_config — all six positions produce validated dicts
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("position", POSITIONS)
def test_build_pipeline_config_includes_all_required_keys(position: str) -> None:
    pc_mod = importlib.import_module(f"src.{position.lower()}.config")
    cfg = build_pipeline_config(position, pc_mod.POSITION_CONFIG)
    missing = REQUIRED_PIPELINE_CFG_KEYS - cfg.keys()
    assert not missing, f"{position}: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize("position", POSITIONS)
def test_build_pipeline_config_overrides_merge_last(position: str) -> None:
    """Caller-supplied overrides win against the factory's defaults — this is
    the mechanism K uses to inject its runtime ``attn_history_builder_fn``.
    """
    pc_mod = importlib.import_module(f"src.{position.lower()}.config")

    sentinel = object()
    cfg = build_pipeline_config(position, pc_mod.POSITION_CONFIG, attn_history_builder_fn=sentinel)
    assert cfg["attn_history_builder_fn"] is sentinel


# --------------------------------------------------------------------------
# validate_pipeline_config — single-shot missing-key report
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_pipeline_config_accepts_complete_dict() -> None:
    complete = {key: object() for key in REQUIRED_PIPELINE_CFG_KEYS}
    # Should not raise.
    validate_pipeline_config(complete)


@pytest.mark.unit
def test_validate_pipeline_config_lists_all_missing_keys() -> None:
    with pytest.raises(PipelineConfigError) as exc_info:
        validate_pipeline_config({"targets": ["x"], "loss_weights": {}})
    msg = str(exc_info.value)
    # Every required key we didn't supply must show up in the error.
    for missing in REQUIRED_PIPELINE_CFG_KEYS - {"targets", "loss_weights"}:
        assert f"'{missing}'" in msg
    # The keys we *did* supply must not be flagged. Quote-wrap to avoid the
    # substring false positive where "targets" matches inside e.g.
    # "compute_targets_fn".
    listing = msg.split("missing required keys:")[1].split("]")[0]
    assert "'targets'" not in listing
    assert "'loss_weights'" not in listing


@pytest.mark.unit
def test_validate_pipeline_config_includes_context_in_error() -> None:
    with pytest.raises(PipelineConfigError, match="for QB"):
        validate_pipeline_config({}, context="QB")


@pytest.mark.unit
def test_required_keys_set_is_non_empty() -> None:
    """Guard against an accidental empty-set bypass of the validator."""
    assert len(REQUIRED_PIPELINE_CFG_KEYS) > 0
    # Spot-check a few entries that pipeline.py reads unconditionally
    for key in ("targets", "loss_weights", "huber_deltas", "nn_lr"):
        assert key in REQUIRED_PIPELINE_CFG_KEYS


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing_key",
    ["specific_features", "add_features_fn", "fill_nans_fn"],
)
def test_validate_pipeline_config_catches_feature_build_required_keys(
    missing_key: str,
) -> None:
    """M9: ``src/shared/feature_build.py`` reads these three keys via ``cfg["..."]``
    (not ``.get``); they MUST be in :data:`REQUIRED_PIPELINE_CFG_KEYS` so the
    validator surfaces a missing one at *build* time instead of crashing inside
    ``build_position_features``.
    """
    cfg = {key: object() for key in REQUIRED_PIPELINE_CFG_KEYS}
    cfg.pop(missing_key)
    with pytest.raises(PipelineConfigError, match=missing_key):
        validate_pipeline_config(cfg)


@pytest.mark.unit
def test_build_pipeline_config_with_minimal_position_config() -> None:
    """A hand-built PositionConfig should produce a buildable cfg too —
    but only if it sets enough fields. This pins the contract for anyone
    adding a 7th position later."""
    minimal_pc = PositionConfig(
        name="QB",  # piggyback on QB's data/features/targets modules
        targets=["passing_yards"],
        specific_features=[],
        ridge_alpha_grids={"passing_yards": [1.0]},
        nn_backbone_layers=[8],
        loss_weights={"passing_yards": 1.0},
        head_losses={"passing_yards": "huber"},
        huber_deltas={"passing_yards": 1.0},
    )
    cfg = build_pipeline_config("QB", minimal_pc)
    # All required keys present
    assert REQUIRED_PIPELINE_CFG_KEYS.issubset(cfg.keys())
