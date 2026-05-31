"""Import-smoke for the Ridge-PCR ablation CLI.

Operator diagnostic (no full run in CI — it needs current splits + minutes of
Ridge tuning). This guards against import/signature drift in the shared APIs it
calls (``run_pipeline``, ``build_pipeline_config``) so a rename fails the unit
shard rather than only surfacing when someone runs the ablation by hand.
"""

import pytest

from src.tuning import ablate_ridge_pca as abl

pytestmark = pytest.mark.unit


def test_module_and_helpers_importable():
    for name in (
        "_ridge_only_cfg",
        "_shrink_for_smoke",
        "_ridge_total_mae",
        "sweep_position",
        "main",
    ):
        assert hasattr(abl, name), f"missing {name}"
    assert abl.DEFAULT_PCA_GRID[0] is None  # baseline always first


def test_ridge_only_cfg_disables_nonridge_models():
    """The A/B is exact only if every non-Ridge model is off (PCA feeds Ridge only)."""
    cfg = abl._ridge_only_cfg("QB")
    assert cfg["train_base_nn"] is False
    assert cfg["train_attention_nn"] is False
    assert cfg["train_lightgbm"] is False
    assert cfg["train_elasticnet"] is False
    # Ridge stays on (default True).
    assert cfg.get("train_ridge", True) is True
    # build_pipeline_config omits ridge_pca_components when the position's
    # default is None (QB), so it's absent here -> reads as None. The sweep
    # sets it explicitly per iteration; pipeline reads it via cfg.get(...).
    assert cfg.get("ridge_pca_components") is None


def test_shrink_for_smoke_collapses_alpha_search():
    cfg = abl._ridge_only_cfg("QB")
    abl._shrink_for_smoke(cfg)
    assert cfg["ridge_cv_folds"] == 2
    assert cfg["ridge_refine_points"] == 0
    assert all(len(g) == 1 for g in cfg["ridge_alpha_grids"].values())
