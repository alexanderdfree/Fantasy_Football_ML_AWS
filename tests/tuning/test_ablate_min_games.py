"""Smoke + math tests for src/tuning/ablate_min_games.py.

Covers the loadability surface and the pure helpers (bucketing, subgroup MAE,
seed aggregation, cfg-mutator, decision table) so a future config/pipeline
refactor doesn't break the ablation CLI silently. The actual pipeline ``run``
is not invoked — that is a multi-minute training run, out of scope for unit tests.

The most important guard here is ``test_min_games_in_cache_fingerprint``: the swept
threshold flows through ``cfg["min_games_per_season"]``, which must be in the
feature-cache fingerprint so threshold variants get distinct cache entries (else a
live cache serves the prior threshold's filtered features and the ablation is
silently invalid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tuning import ablate_min_games as amg

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    for attr in (
        "main",
        "VARIANTS",
        "DEFAULT_VARIANTS",
        "DEFAULT_POSITIONS",
        "ABLATION_NAME",
        "_execute_min_games_job",
        "_build_jobs",
        "print_position_summary",
        "_subgroup_mae",
        "_bucket_masks",
        "_aggregate_seeds",
        "_pred_total_cols",
        "_make_cfg",
        "_results_to_seed_runs",
    ):
        assert hasattr(amg, attr), f"missing {attr}"


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


def test_variants_keys_and_thresholds():
    """VARIANTS maps thr<N> keys to the expected integer thresholds."""
    assert amg.VARIANTS == {"thr1": 1, "thr3": 3, "thr6": 6, "thr8": 8}


def test_default_variants_are_subset_of_variants():
    assert set(amg.DEFAULT_VARIANTS) <= set(amg.VARIANTS)


def test_default_positions_are_uppercase():
    for pos in amg.DEFAULT_POSITIONS:
        assert pos == pos.upper(), f"expected uppercase position, got {pos!r}"


# ---------------------------------------------------------------------------
# Cache fingerprint sentinel (most important guard)
# ---------------------------------------------------------------------------


def test_min_games_in_cache_fingerprint():
    """The swept threshold flows through cfg["min_games_per_season"], which must be in
    the feature-cache fingerprint so threshold variants get distinct cache entries
    (else a live cache serves the prior threshold's filtered features)."""
    from src.shared.feature_cache import _config_fingerprint

    base = {
        "filter_fn": lambda d: d,
        "compute_targets_fn": lambda d: d,
        "get_feature_columns_fn": lambda: [],
        "add_features_fn": lambda *a, **k: None,
        "fill_nans_fn": lambda *a, **k: None,
    }
    fp1 = _config_fingerprint({**base, "min_games_per_season": 1})
    fp6 = _config_fingerprint({**base, "min_games_per_season": 6})
    assert fp1["min_games_per_season"] == 1
    assert fp1 != fp6  # distinct thresholds → distinct fingerprint → distinct cache key


# ---------------------------------------------------------------------------
# _make_cfg: cfg-mutator sets min_games_per_season correctly
# ---------------------------------------------------------------------------


def test_make_cfg_sets_threshold():
    """_make_cfg sets cfg['min_games_per_season'] = threshold without side-effects."""
    base = {"min_games_per_season": 6, "nn_epochs": 50, "train_attention_nn": True}
    cfg, meta = amg._make_cfg(base, threshold=1, skip_nn=False)
    assert cfg["min_games_per_season"] == 1
    assert meta["threshold"] == 1
    assert meta["skip_nn"] is False
    # Original must not be mutated.
    assert base["min_games_per_season"] == 6


def test_make_cfg_skip_nn_sets_minimal_epochs():
    """skip_nn=True degrades the NN branch while preserving Ridge/LightGBM at full config."""
    base = {"min_games_per_season": 6, "nn_epochs": 50, "train_attention_nn": True}
    cfg, meta = amg._make_cfg(base, threshold=3, skip_nn=True)
    assert cfg["min_games_per_season"] == 3
    assert cfg["nn_epochs"] == 2
    assert cfg["train_attention_nn"] is False
    assert meta["skip_nn"] is True
    # Original must not be mutated.
    assert base["nn_epochs"] == 50
    assert base["train_attention_nn"] is True


def test_make_cfg_no_skip_nn_leaves_nn_intact():
    """skip_nn=False must not alter nn_epochs or train_attention_nn."""
    base = {"min_games_per_season": 6, "nn_epochs": 50, "train_attention_nn": True}
    cfg, _ = amg._make_cfg(base, threshold=8, skip_nn=False)
    assert cfg["nn_epochs"] == 50
    assert cfg["train_attention_nn"] is True


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _subgroup_mae
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _aggregate_seeds
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _results_to_seed_runs: AblationResult → per-seed dict re-packaging
# ---------------------------------------------------------------------------


def test_results_to_seed_runs_basic():
    """_results_to_seed_runs extracts the per-seed bucket dicts for one (pos, variant)."""
    from src.tuning.ablation_runner import AblationResult

    buckets = {"ALL": {"n": 5, "pred_ridge_total": 2.0}, "would_filter(<6)": {"n": 2}}
    pred_cols = ["pred_ridge_total"]
    results = [
        AblationResult(
            position="RB",
            seed=42,
            variant="thr1",
            metrics={"buckets": buckets, "pred_cols": pred_cols},
            timings={},
            metadata={"run_kind": "experiment"},
        ),
        AblationResult(
            position="RB",
            seed=43,
            variant="thr6",  # different variant — must not appear
            metrics={"buckets": buckets, "pred_cols": pred_cols},
            timings={},
            metadata={"run_kind": "experiment"},
        ),
        AblationResult(
            position="WR",
            seed=42,
            variant="thr1",  # different position — must not appear
            metrics={"buckets": buckets, "pred_cols": pred_cols},
            timings={},
            metadata={"run_kind": "experiment"},
        ),
    ]
    runs = amg._results_to_seed_runs(results, "RB", "thr1")
    assert len(runs) == 1
    assert runs[0]["pred_cols"] == pred_cols
    assert runs[0]["buckets"] == buckets


def test_results_to_seed_runs_skips_errors():
    """Results with error set must not appear in seed_runs."""
    from src.tuning.ablation_runner import AblationResult

    results = [
        AblationResult(
            position="RB",
            seed=42,
            variant="thr1",
            metrics={},
            timings={},
            metadata={},
            error="something went wrong",
        ),
    ]
    runs = amg._results_to_seed_runs(results, "RB", "thr1")
    assert runs == []


# ---------------------------------------------------------------------------
# Decision / summary table
# ---------------------------------------------------------------------------


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


def test_print_position_summary_unreliable_rows_and_empty_bucket(capsys):
    """K/DST shape: unreliable train-row count + an empty would_filter bucket (n=0)
    must not crash and must annotate both."""

    def _stub():
        return {
            "buckets": {
                "ALL": {"n": 5, "pred_lgbm_total": 5.0},
                "would_filter(<6)": {"n": 0},  # no pred cols when the bucket is empty
                "kept(>=6)": {"n": 5, "pred_lgbm_total": 5.0},
            },
            "pred_cols": ["pred_lgbm_total"],
        }

    per_threshold = {1: [_stub()], 6: [_stub()]}
    amg.print_position_summary(
        "dst", {1: 60645, 6: 50789}, per_threshold, baseline_threshold=6, train_rows_reliable=False
    )
    out = capsys.readouterr().out
    assert "UNRELIABLE" in out  # train-row caveat for identity-filter positions
    assert "n=0 (no rows" in out  # empty-bucket verdict guard


def test_print_position_summary_multi_seed(capsys):
    """Multi-seed path: mean±std columns must appear in header."""

    def _stub(mae):
        return {
            "buckets": {
                "ALL": {"n": 10, "pred_ridge_total": mae},
                "would_filter(<6)": {"n": 3, "pred_ridge_total": mae + 0.5},
                "kept(>=6)": {"n": 7, "pred_ridge_total": mae},
            },
            "pred_cols": ["pred_ridge_total"],
        }

    per_threshold = {1: [_stub(5.0), _stub(5.2)], 6: [_stub(5.5), _stub(5.7)]}
    amg.print_position_summary("wr", {1: 4000, 6: 3500}, per_threshold, baseline_threshold=6)
    out = capsys.readouterr().out
    assert "mean±std over seeds" in out
    assert "WR" in out
