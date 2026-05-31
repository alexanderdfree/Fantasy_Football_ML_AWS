"""Smoke + math tests for src/tuning/ablate_min_games.py.

Covers the loadability surface and the pure helpers (bucketing, subgroup MAE,
seed aggregation) so a future config/pipeline refactor doesn't break the ablation
CLI silently. The actual pipeline ``run`` is not invoked — that is a multi-minute
training run, out of scope for unit tests.

The most important guard here is ``test_feature_cache_disabled_on_import``: the
min-games threshold is NOT in the feature-cache key, so without the env flag every
threshold variant would reuse the first variant's filtered features and the whole
ablation would be silently invalid.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.tuning import ablate_min_games as amg

pytestmark = pytest.mark.unit


def test_module_imports_cleanly():
    for attr in (
        "main",
        "parse_args",
        "run_one",
        "print_position_summary",
        "_subgroup_mae",
        "_bucket_masks",
        "_aggregate_seeds",
        "_pred_total_cols",
    ):
        assert hasattr(amg, attr), f"missing {attr}"


def test_feature_cache_disabled_on_import():
    """Importing the module must force the feature cache off — the threshold is not
    in the cache key, so a live cache would serve stale filtered features."""
    assert os.environ.get("FF_FEATURE_CACHE_DISABLE") == "1"


def test_parse_args_defaults():
    args = amg.parse_args([])
    assert args.positions == ["rb", "wr"]
    assert args.thresholds == [1, 3, 6, 8]
    assert args.seeds == 1
    assert args.baseline_threshold == 6


def test_bucket_masks_partition_and_cover():
    g = pd.Series([1, 2, 3, 4, 5, 6, 7, 9, 10, 12])
    masks = amg._bucket_masks(g)
    # would_filter (<6) and kept (>=6) are complementary and cover ALL.
    assert (masks["would_filter(<6)"] | masks["kept(>=6)"]).all()
    assert not (masks["would_filter(<6)"] & masks["kept(>=6)"]).any()
    assert masks["ALL"].all()
    # Fine buckets under 6 sum to would_filter.
    fine_low = masks["g=1"] | masks["g=2-3"] | masks["g=4-5"]
    assert (fine_low == masks["would_filter(<6)"]).all()
    # Fine buckets at/above 6 sum to kept.
    fine_hi = masks["g=6-9"] | masks["g>=10"]
    assert (fine_hi == masks["kept(>=6)"]).all()


def test_pred_total_cols_discovery():
    df = pd.DataFrame(
        columns=["player_id", "pred_ridge_total", "pred_baseline", "pred_nn_total", "pred_ridge_a"]
    )
    cols = amg._pred_total_cols(df)
    assert cols == ["pred_nn_total", "pred_ridge_total"]  # sorted; baseline + per-target excluded


def _synthetic_test_df() -> pd.DataFrame:
    """Player A: 3 games (would_filter); Player B: 7 games (kept). Total FP = pts.
    pred_ridge_total errs by +2 on A rows, +1 on B rows -> would_filter MAE 2.0,
    kept MAE 1.0, ALL = (3*2 + 7*1)/10 = 1.3. pred_nn_total is exact (MAE 0)."""
    rows = []
    for wk in range(1, 4):  # player A, 3 games
        rows.append({"player_id": "A", "season": 2025, "week": wk, "pts": 10.0})
    for wk in range(1, 8):  # player B, 7 games
        rows.append({"player_id": "B", "season": 2025, "week": wk, "pts": 20.0})
    df = pd.DataFrame(rows)
    df["pred_ridge_total"] = df["pts"] + np.where(df["player_id"] == "A", 2.0, 1.0)
    df["pred_nn_total"] = df["pts"]
    df["pred_baseline"] = df["pts"] + 5.0  # must be ignored (no _total suffix)
    return df


def test_subgroup_mae_known_values():
    df = _synthetic_test_df()
    agg = lambda preds: preds["pts"]  # noqa: E731 — fantasy total == pts here
    out = amg._subgroup_mae(df, targets=["pts"], agg=agg)

    assert out["ALL"]["n"] == 10
    assert out["would_filter(<6)"]["n"] == 3
    assert out["kept(>=6)"]["n"] == 7

    assert out["would_filter(<6)"]["pred_ridge_total"] == pytest.approx(2.0)
    assert out["kept(>=6)"]["pred_ridge_total"] == pytest.approx(1.0)
    assert out["ALL"]["pred_ridge_total"] == pytest.approx(1.3)
    # Exact model contributes 0 error everywhere; baseline column is excluded.
    assert out["ALL"]["pred_nn_total"] == pytest.approx(0.0)
    assert "pred_baseline" not in out["ALL"]


def test_subgroup_mae_falls_back_to_sum_without_agg():
    """No aggregator -> actual total = sum(heads). Two targets summed."""
    df = _synthetic_test_df().rename(columns={"pts": "a"})
    df["b"] = 0.0
    df["pred_ridge_total"] = df["a"] + np.where(df["player_id"] == "A", 2.0, 1.0)
    out = amg._subgroup_mae(df, targets=["a", "b"], agg=None)
    assert out["would_filter(<6)"]["pred_ridge_total"] == pytest.approx(2.0)


def test_aggregate_seeds_mean_std():
    seed_runs = [
        {"buckets": {"ALL": {"n": 5, "pred_nn_total": 1.0}}, "pred_cols": ["pred_nn_total"]},
        {"buckets": {"ALL": {"n": 5, "pred_nn_total": 3.0}}, "pred_cols": ["pred_nn_total"]},
    ]
    agg, pred_cols = amg._aggregate_seeds(seed_runs)
    assert pred_cols == ["pred_nn_total"]
    cell = agg["ALL"]["pred_nn_total"]
    assert cell["mean"] == pytest.approx(2.0)
    assert cell["std"] == pytest.approx(1.0)
    assert cell["n_seeds"] == 2
    assert agg["ALL"]["n"] == 5


def test_print_position_summary_runs(capsys):
    """Two thresholds, one seed each -> format path executes and prints a verdict."""

    def _stub(ridge_mae):
        return {
            "buckets": {
                "ALL": {"n": 10, "pred_ridge_total": ridge_mae},
                "would_filter(<6)": {"n": 3, "pred_ridge_total": ridge_mae + 1.0},
                "kept(>=6)": {"n": 7, "pred_ridge_total": ridge_mae},
            },
            "pred_cols": ["pred_ridge_total"],
        }

    per_threshold = {1: [_stub(6.0)], 6: [_stub(6.5)]}
    train_rows = {1: 5000, 6: 4200}
    amg.print_position_summary("rb", train_rows, per_threshold, baseline_threshold=6)
    out = capsys.readouterr().out
    assert "RB" in out
    assert "VERDICT" in out
    assert "would_filter(<6)" in out
