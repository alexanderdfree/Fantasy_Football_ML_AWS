"""Contract tests for the WR ``CONFIG_TINY`` e2e fixture (#1566 drift, #1612).

``CONFIG_TINY`` is consumed two ways. The WR e2e/CV fixtures
(``tests/wr/test_pipeline_e2e.py``, ``tests/wr/test_run_cv_pipeline.py``) resolve
it standalone, where it must satisfy the recipe validator on its own; the
``--tiny`` paths (``src.shared.run_pipeline_factory._build_tiny_config`` and
``tests._pipeline_e2e_utils.build_tiny_config``) merge it LAST over
``POSITION_CONFIG``, where it must not override production's head losses.
Both contracts are cheap to pin without data, so CI's WR shard runs them even
though it skips the split-dependent e2e tests (``require_splits``).
"""

from __future__ import annotations

import pytest

from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.run_pipeline_factory import _build_tiny_config
from src.training.contracts import resolve_recipe
from src.wr.config import CONFIG_TINY, POSITION_CONFIG
from src.wr.data import filter_to_position
from src.wr.features import add_specific_features, fill_nans, get_feature_columns
from src.wr.targets import compute_targets
from tests._pipeline_e2e_utils import build_tiny_config

pytestmark = pytest.mark.unit


def _standalone_tiny_cfg() -> dict:
    """The exact shape the WR e2e/CV fixtures build: CONFIG_TINY plus callables."""
    return {
        **CONFIG_TINY,
        "filter_fn": filter_to_position,
        "compute_targets_fn": compute_targets,
        "add_features_fn": add_specific_features,
        "fill_nans_fn": fill_nans,
        "get_feature_columns_fn": get_feature_columns,
        "aggregate_fn": aggregate_fn_for("WR"),
    }


def test_config_tiny_resolves_standalone():
    """No head_losses → every head is Huber, so a delta must exist per target."""
    assert "head_losses" not in CONFIG_TINY
    recipe = resolve_recipe("WR", _standalone_tiny_cfg(), require_runtime=False)
    assert set(recipe["huber_deltas"]) == set(POSITION_CONFIG.targets)
    assert (
        recipe["huber_deltas"]["receiving_yards"] == POSITION_CONFIG.huber_deltas["receiving_yards"]
    )


@pytest.mark.parametrize("build", [_build_tiny_config, build_tiny_config])
def test_config_tiny_keeps_production_heads_when_merged_last(build):
    """Merged last over POSITION_CONFIG, CONFIG_TINY must leave the head losses alone."""
    merged = build("WR")
    assert merged["head_losses"] == POSITION_CONFIG.head_losses
    assert merged["gated_targets"] == POSITION_CONFIG.gated_targets
    resolve_recipe("WR", merged, require_runtime=False)
