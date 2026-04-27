"""Unit tests for src.analysis.analysis_nflcom_baseline.

All tests construct synthetic frames; nothing hits the network or trains a real
model. The analysis script's ``pipeline_runner`` and ``nflcom_loader`` injection
points are used to substitute fakes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_nflcom_baseline as ab
from src.config import SCORING_PPR, SCORING_STANDARD

pytestmark = pytest.mark.unit


# ---------- Fixtures --------------------------------------------------------


def _make_qb_test_df(n_weeks: int = 4) -> pd.DataFrame:
    """Two QBs × n_weeks weeks. Targets and fantasy_points are consistent."""
    rng = np.random.default_rng(42)
    rows = []
    for player_id in ("00-Q1", "00-Q2"):
        for week in range(1, n_weeks + 1):
            py = rng.uniform(180, 320)
            ptd = rng.uniform(1.0, 3.0)
            ints = rng.uniform(0.0, 1.5)
            ry = rng.uniform(0.0, 50.0)
            rtd = rng.uniform(0.0, 0.6)
            fum_lost = rng.uniform(0.0, 0.4)
            fp = (
                py * SCORING_PPR["passing_yards"]
                + ptd * SCORING_PPR["passing_tds"]
                + ints * SCORING_PPR["interceptions"]
                + ry * SCORING_PPR["rushing_yards"]
                + rtd * SCORING_PPR["rushing_tds"]
                + fum_lost * SCORING_PPR["fumbles_lost"]
            )
            rows.append(
                {
                    "player_id": player_id,
                    "season": 2025,
                    "week": week,
                    "position": "QB",
                    "passing_yards": py,
                    "passing_tds": ptd,
                    "interceptions": ints,
                    "rushing_yards": ry,
                    "rushing_tds": rtd,
                    "fumbles_lost": fum_lost,
                    "fantasy_points": fp,
                }
            )
    return pd.DataFrame(rows)


def _make_pipeline_result(test_df: pd.DataFrame, *, perfect: bool = False) -> dict:
    """Return a pipeline_result dict matching the real shape.

    With ``perfect=True``, model preds equal actuals → MAE should be 0.
    Otherwise add small noise.
    """
    targets = (
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "fumbles_lost",
    )
    rng = np.random.default_rng(0)
    preds = {}
    for t in targets:
        truth = test_df[t].to_numpy()
        preds[t] = truth.copy() if perfect else truth + rng.normal(0, 0.05, len(truth))
    per_target = {"ridge": preds, "nn": preds, "attn_nn": preds}
    return {
        "test_df": test_df,
        "per_target_preds": per_target,
        "ridge_metrics": {"total": {"mae": 0.0}},
        "nn_metrics": {"total": {"mae": 0.0}},
        "attn_nn_metrics": {"total": {"mae": 0.0}},
    }


def _make_nflcom_df_for_qb(test_df: pd.DataFrame, *, perfect: bool = False) -> pd.DataFrame:
    """Build a NFL.com-shaped frame keyed by (player_id, season, week, position) so
    that ``_project_nflcom_to_ppr`` can run."""
    rng = np.random.default_rng(7)
    out_rows = []
    for _, row in test_df.iterrows():
        if perfect:
            scaled = {
                t: row[t]
                for t in (
                    "passing_yards",
                    "passing_tds",
                    "interceptions",
                    "rushing_yards",
                    "rushing_tds",
                    "fumbles_lost",
                )
            }
        else:
            scaled = {
                "passing_yards": row["passing_yards"] + rng.normal(0, 5),
                "passing_tds": row["passing_tds"] + rng.normal(0, 0.1),
                "interceptions": row["interceptions"] + rng.normal(0, 0.1),
                "rushing_yards": row["rushing_yards"] + rng.normal(0, 1),
                "rushing_tds": row["rushing_tds"] + rng.normal(0, 0.05),
                "fumbles_lost": row["fumbles_lost"] + rng.normal(0, 0.05),
            }
        out_rows.append(
            {
                "player_id": row["player_id"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "position": "QB",
                "nflcom_projected_pts": 18.0,  # arbitrary native-scoring reference
                **scaled,
                # Fill the offensive-stat columns the loader writes for all positions.
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
                "receptions": 0.0,
            }
        )
    return pd.DataFrame(out_rows)


# ---------- Pure-function tests ---------------------------------------------


def test_select_model_predictions_prefers_attn_nn():
    pipeline_result = {
        "per_target_preds": {
            "ridge": {"x": np.zeros(2)},
            "nn": {"x": np.ones(2)},
            "attn_nn": {"x": np.full(2, 7.0)},
        },
        "attn_nn_metrics": {"total": {"mae": 1.0}},
    }
    preds, label = ab._select_model_predictions(pipeline_result, "QB")
    assert label == "Attention NN"
    np.testing.assert_array_equal(preds["x"], np.full(2, 7.0))


def test_select_model_predictions_falls_back_to_nn():
    pipeline_result = {
        "per_target_preds": {"ridge": {"x": np.zeros(2)}, "nn": {"x": np.ones(2)}},
        # No attn_nn_metrics — falls through.
    }
    preds, label = ab._select_model_predictions(pipeline_result, "K")
    assert label == "Multi-Head NN"


def test_select_model_predictions_falls_back_to_ridge():
    pipeline_result = {"per_target_preds": {"ridge": {"x": np.zeros(2)}}}
    preds, label = ab._select_model_predictions(pipeline_result, "QB")
    assert label == "Ridge Multi-Target"


def test_select_model_predictions_raises_when_empty():
    with pytest.raises(RuntimeError, match="No model predictions"):
        ab._select_model_predictions({"per_target_preds": {}}, "QB")


def test_decide_winner():
    assert ab._decide_winner(5.0, 5.0) == "tie"
    assert ab._decide_winner(5.0, 5.005) == "tie"  # within tolerance
    assert ab._decide_winner(4.9, 5.1) == "model"
    assert ab._decide_winner(5.2, 5.0) == "nflcom"


# ---------- Aggregation tests ------------------------------------------------


def test_project_nflcom_to_ppr_qb_uses_scoring_dict():
    """NFL.com row with known stats should yield exactly SCORING_PPR-weighted total."""
    df = pd.DataFrame(
        [
            {
                "player_id": "00-001",
                "season": 2025,
                "week": 1,
                "position": "QB",
                "passing_yards": 300.0,
                "passing_tds": 2.0,
                "interceptions": 1.0,
                "rushing_yards": 50.0,
                "rushing_tds": 0.5,
                "fumbles_lost": 0.2,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
                "receptions": 0.0,
                "nflcom_projected_pts": 22.0,
            }
        ]
    )
    out = ab._project_nflcom_to_ppr(df, "QB", scoring_format="ppr")
    expected = (
        300.0 * SCORING_PPR["passing_yards"]
        + 2.0 * SCORING_PPR["passing_tds"]
        + 1.0 * SCORING_PPR["interceptions"]
        + 50.0 * SCORING_PPR["rushing_yards"]
        + 0.5 * SCORING_PPR["rushing_tds"]
        + 0.2 * SCORING_PPR["fumbles_lost"]
    )
    assert out["nflcom_pred_total"].iloc[0] == pytest.approx(expected)
    # Native projection passed through.
    assert out["nflcom_projected_pts"].iloc[0] == pytest.approx(22.0)


def test_project_nflcom_to_ppr_rb_ppr_vs_standard_propagates():
    """Verify scoring_format reaches the aggregator: 5 receptions worth +5 in PPR, 0 in standard."""
    df = pd.DataFrame(
        [
            {
                "player_id": "00-RB1",
                "season": 2025,
                "week": 1,
                "position": "RB",
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "rushing_yards": 0.0,
                "rushing_tds": 0.0,
                "receiving_yards": 80.0,
                "receiving_tds": 0.0,
                "receptions": 5.0,
                "fumbles_lost": 0.0,
                "nflcom_projected_pts": 13.0,
            }
        ]
    )
    ppr = ab._project_nflcom_to_ppr(df, "RB", scoring_format="ppr")
    std = ab._project_nflcom_to_ppr(df, "RB", scoring_format="standard")
    # PPR: 80*0.1 + 5*1 = 13; Standard: 80*0.1 + 5*0 = 8
    assert ppr["nflcom_pred_total"].iloc[0] == pytest.approx(
        80 * SCORING_PPR["receiving_yards"] + 5 * SCORING_PPR["receptions"]
    )
    assert std["nflcom_pred_total"].iloc[0] == pytest.approx(
        80 * SCORING_STANDARD["receiving_yards"] + 5 * SCORING_STANDARD["receptions"]
    )


def test_project_nflcom_to_ppr_drops_unmatched():
    """Rows with player_id NaN are dropped — they can't be joined to test_df."""
    df = pd.DataFrame(
        [
            {
                "player_id": np.nan,
                "season": 2025,
                "week": 1,
                "position": "QB",
                "passing_yards": 200.0,
                "passing_tds": 1.0,
                "interceptions": 0.0,
                "rushing_yards": 0.0,
                "rushing_tds": 0.0,
                "fumbles_lost": 0.0,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
                "receptions": 0.0,
                "nflcom_projected_pts": 12.0,
            }
        ]
    )
    out = ab._project_nflcom_to_ppr(df, "QB")
    assert len(out) == 0


def test_project_nflcom_to_ppr_k_uses_native_pts():
    """K can't decompose into raw stats — uses NFL.com's native projection directly."""
    df = pd.DataFrame(
        [
            {
                "player_id": "00-K1",
                "season": 2025,
                "week": 1,
                "position": "K",
                "nflcom_projected_pts": 8.5,
                # K rows from the loader have all offensive cols filled with 0.
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "rushing_yards": 0.0,
                "rushing_tds": 0.0,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
                "receptions": 0.0,
                "fumbles_lost": 0.0,
            }
        ]
    )
    out = ab._project_nflcom_to_ppr(df, "K")
    assert out["nflcom_pred_total"].iloc[0] == pytest.approx(8.5)
    # No per-target columns for K.
    assert not any(c.startswith("nflcom_pred_") and c != "nflcom_pred_total" for c in out.columns)


# ---------- Position-comparison tests ---------------------------------------


def test_compute_position_comparison_perfect_model_zero_mae():
    test_df = _make_qb_test_df(n_weeks=3)
    pipeline_result = _make_pipeline_result(test_df, perfect=True)
    nflcom = _make_nflcom_df_for_qb(test_df, perfect=False)

    result = ab._compute_position_comparison(
        "QB", pipeline_result, nflcom, eval_season=2025, scoring_format="ppr"
    )
    assert result["position"] == "QB"
    assert result["model_label"] == "Attention NN"
    assert result["match_rate"] == pytest.approx(1.0)
    # Model is perfect → MAE = 0.
    assert result["metrics"]["model"]["mae"] == pytest.approx(0.0, abs=1e-9)
    # NFL.com is noisy → MAE > 0.
    assert result["metrics"]["nflcom_ppr"]["mae"] > 0
    assert result["who_won"] == "model"
    # Per-target: all 6 QB targets present.
    assert set(result["per_target"].keys()) == {
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "fumbles_lost",
    }
    # Weekly: 3 entries.
    assert len(result["weekly"]) == 3


def test_compute_position_comparison_match_rate_partial():
    """Test_df has 2 players × 3 weeks; NFL.com only has player 1."""
    test_df = _make_qb_test_df(n_weeks=3)
    pipeline_result = _make_pipeline_result(test_df, perfect=True)
    full_nflcom = _make_nflcom_df_for_qb(test_df, perfect=True)
    partial_nflcom = full_nflcom[full_nflcom["player_id"] == "00-Q1"].copy()

    result = ab._compute_position_comparison(
        "QB", pipeline_result, partial_nflcom, eval_season=2025
    )
    # Only 3/6 rows match.
    assert result["n_test_total"] == 6
    assert result["n_matched"] == 3
    assert result["match_rate"] == pytest.approx(0.5)


def test_compute_position_comparison_per_target_only_skill_positions():
    test_df = _make_qb_test_df(n_weeks=2)
    pipeline_result = _make_pipeline_result(test_df, perfect=True)
    nflcom = _make_nflcom_df_for_qb(test_df, perfect=True)

    qb_result = ab._compute_position_comparison("QB", pipeline_result, nflcom, eval_season=2025)
    assert qb_result["per_target"]  # non-empty


def test_compute_position_comparison_raises_on_empty_join():
    test_df = _make_qb_test_df(n_weeks=2)
    pipeline_result = _make_pipeline_result(test_df, perfect=True)
    # NFL.com frame for a different season — no overlap.
    nflcom = _make_nflcom_df_for_qb(test_df, perfect=True)
    nflcom["season"] = 2024

    with pytest.raises(RuntimeError, match="No .* overlap"):
        ab._compute_position_comparison("QB", pipeline_result, nflcom, eval_season=2025)


def test_compute_position_comparison_skipped_when_test_df_empty():
    test_df = _make_qb_test_df(n_weeks=2)
    test_df["season"] = 2024  # no rows for 2025 eval
    pipeline_result = _make_pipeline_result(test_df, perfect=True)
    nflcom = _make_nflcom_df_for_qb(test_df, perfect=True)
    nflcom["season"] = 2025

    result = ab._compute_position_comparison("QB", pipeline_result, nflcom, eval_season=2025)
    assert result.get("skipped") is True


# ---------- main() smoke test -----------------------------------------------


def test_main_writes_json_and_handles_dst_skip(tmp_path):
    test_df = _make_qb_test_df(n_weeks=3)
    pipeline_result = _make_pipeline_result(test_df, perfect=True)
    nflcom_full = _make_nflcom_df_for_qb(test_df, perfect=False)

    def fake_runner(pos):
        # Same fake pipeline for every position; only QB is exercised here.
        def _run():
            return pipeline_result

        return _run

    def fake_loader(seasons, force_refresh=False):
        return nflcom_full

    output_dir = tmp_path / "analysis_output"
    result = ab.main(
        eval_season=2025,
        scoring_format="ppr",
        positions=("QB", "DST"),
        output_dir=str(output_dir),
        pipeline_runner=fake_runner,
        nflcom_loader=fake_loader,
    )
    assert "QB" in result["positions"]
    assert "DST" in result["positions"]
    assert result["positions"]["DST"]["skipped"] is True
    assert "summary" in result and "headlines" in result["summary"]
    # JSON written and re-readable.
    json_path = output_dir / "nflcom_baseline_comparison.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk["eval_window"] == "2025_test"
    assert on_disk["scoring"] == "ppr"


def test_main_passes_force_refresh_through(tmp_path):
    test_df = _make_qb_test_df(n_weeks=2)
    pipeline_result = _make_pipeline_result(test_df, perfect=True)
    nflcom_full = _make_nflcom_df_for_qb(test_df, perfect=False)
    seen = {}

    def fake_runner(pos):
        return lambda: pipeline_result

    def fake_loader(seasons, force_refresh=False):
        seen["force_refresh"] = force_refresh
        return nflcom_full

    ab.main(
        eval_season=2025,
        positions=("QB",),
        output_dir=str(tmp_path),
        pipeline_runner=fake_runner,
        nflcom_loader=fake_loader,
        force_refresh_nflcom=True,
    )
    assert seen["force_refresh"] is True
