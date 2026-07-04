"""Unit tests for the top-N expert-gap diagnostic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import topn_expert_gap as mod
from src.analysis.analysis_expert_comparison import ExpertSource

pytestmark = pytest.mark.unit


def _source(name: str = "model") -> mod.SourceMeta:
    return mod.SourceMeta(name=name, label=name.title(), kind="model", native_col="pred_total")


def test_cohort_error_uses_actual_season_total_topn_players():
    df = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4", "P1", "P2", "P3", "P4"],
            "season": [2025] * 8,
            "week": [1, 1, 1, 1, 2, 2, 2, 2],
            "position": ["WR"] * 8,
            "fantasy_points": [30.0, 20.0, 10.0, 5.0, 20.0, 22.0, 10.0, 5.0],
            "pred_total": [28.0, 17.0, 12.0, 4.0, 18.0, 23.0, 9.0, 7.0],
        }
    )

    rows = mod.cohort_error_rows("WR", _source(), df, df, top_ns=(2,))
    global_row = next(
        r
        for r in rows
        if r["metric_group"] == "cohort_error" and r["slice_family"] == "global" and r["top_n"] == 2
    )

    # Top-2 by season total is P1 (50) and P2 (42), so all four of their weekly
    # rows are scored: abs errors 2, 3, 2, 1.
    assert global_row["n_rows"] == 4
    assert global_row["n_players"] == 2
    assert global_row["mae"] == pytest.approx(2.0)
    assert global_row["bias"] == pytest.approx((-2.0 - 3.0 - 2.0 + 1.0) / 4.0)


def test_weekly_selection_precision_regret_and_undersized_week_skip():
    df = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4", "Q1", "Q2"],
            "season": [2025] * 6,
            "week": [1, 1, 1, 1, 2, 2],
            "fantasy_points": [10.0, 9.0, 8.0, 1.0, 7.0, 6.0],
            "pred_total": [10.0, 9.0, 0.0, 8.0, 6.0, 7.0],
        }
    )

    rows, misses = mod.weekly_selection_rows("QB", _source(), df, top_ns=(3,))

    assert len(rows) == 1
    row = rows[0]
    assert row["week"] == 1
    assert row["precision"] == pytest.approx(2.0 / 3.0)
    assert row["hit_rate"] == pytest.approx(2.0 / 3.0)
    # Ideal top-3 actual total 27, selected actual total P1+P2+P4 = 20.
    assert row["regret"] == pytest.approx(7.0)
    by_type = {(m["player_id"], m["miss_type"]) for m in misses}
    assert ("P3", "miss") in by_type
    assert ("P4", "false_positive") in by_type
    assert all(m["week"] == 1 for m in misses)


def test_weekly_selection_counts_missing_actual_topn_from_base_universe():
    base = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4"],
            "season": [2025] * 4,
            "week": [1] * 4,
            "fantasy_points": [10.0, 9.0, 8.0, 1.0],
        }
    )
    expert = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P4"],
            "season": [2025] * 3,
            "week": [1] * 3,
            "fantasy_points": [10.0, 9.0, 1.0],
            "pred_total": [10.0, 9.0, 8.0],
        }
    )

    rows, misses = mod.weekly_selection_rows("QB", _source(), expert, base, top_ns=(3,))

    row = rows[0]
    assert row["precision"] == pytest.approx(2.0 / 3.0)
    assert row["regret"] == pytest.approx(8.0 - 1.0)
    assert row["actual_universe_rows"] == 4
    assert {(m["player_id"], m["miss_type"]) for m in misses} == {
        ("P1", "hit"),
        ("P2", "hit"),
        ("P3", "miss"),
        ("P4", "false_positive"),
    }
    missed = next(m for m in misses if m["player_id"] == "P3")
    assert missed["actual_fp"] == pytest.approx(8.0)
    assert missed["pred_fp"] is None


def test_season_selection_overlap_lists_hits_misses_and_false_positives():
    df = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4"],
            "season": [2025] * 4,
            "week": [1] * 4,
            "fantasy_points": [30.0, 25.0, 20.0, 10.0],
            "pred_total": [40.0, 5.0, 35.0, 1.0],
        }
    )

    rows, misses = mod.season_selection_rows("RB", _source(), df, df, top_ns=(2,))

    row = rows[0]
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["f1"] == pytest.approx(0.5)
    assert row["hits"] == 1
    assert row["misses"] == 1
    assert row["false_positives"] == 1
    assert {(m["player_id"], m["miss_type"]) for m in misses} == {
        ("P1", "hit"),
        ("P2", "miss"),
        ("P3", "false_positive"),
    }


def test_expert_source_frame_uses_exact_keys_and_skip_rules():
    experts = {src.name: src for src in mod._build_experts(None, None)}
    assert "DST" in experts["nflcom"].skipped
    assert "K" in experts["sleeper"].skipped

    def _project(raw, pos, scoring_format):
        del pos, scoring_format
        return raw.rename(columns={"proj": mod._EXPERT_PRED_COL})

    source = ExpertSource(
        name="fake",
        label="Fake",
        load=lambda seasons: pd.DataFrame(),
        project=_project,
    )
    base = pd.DataFrame(
        {
            "player_id": ["P1", "P2"],
            "season": [2025, 2025],
            "week": [1, 1],
            "fantasy_points": [10.0, 20.0],
        }
    )
    raw = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 1],
            "proj": [11.0, 21.0, 31.0],
        }
    )

    meta, frame, status = mod.expert_source_frame(source, raw, "QB", base, "ppr")

    assert meta.name == "fake"
    assert status["skipped"] is False
    assert len(frame) == 1
    assert frame.iloc[0]["player_id"] == "P1"
    assert frame.iloc[0]["pred_total"] == pytest.approx(11.0)


def test_local_expert_source_reads_approved_snapshot_file(tmp_path):
    path = tmp_path / "fantasypros.csv"
    pd.DataFrame(
        {
            "player_id": ["P1", "P2", "K1"],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "position": ["WR", "WR", "K"],
            "fp_projection": [12.5, 9.0, 7.0],
        }
    ).to_csv(path, index=False)

    spec = mod.parse_local_expert_spec(f"fantasypros:fp_projection={path}")
    source = mod.local_expert_source(spec)
    raw = source.load([2025])
    projected = source.project(raw, "WR", "ppr")

    assert source.name == "fantasypros"
    assert source.label == "Fantasypros"
    assert list(projected["player_id"]) == ["P1", "P2"]
    assert projected[mod._EXPERT_PRED_COL].tolist() == [12.5, 9.0]


def test_local_expert_source_drops_placeholder_zeros_and_preserves_raw_td_cols(tmp_path):
    path = tmp_path / "fantasypros.csv"
    pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4"],
            "season": [2025] * 4,
            "week": [1] * 4,
            "position": ["WR"] * 4,
            "fp_projection": [12.5, 0.0, -1.0, None],
            "projected_receiving_tds": [0.8, 0.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)

    source = mod.local_expert_source(
        mod.parse_local_expert_spec(f"fantasypros:fp_projection={path}")
    )
    projected = source.project(source.load([2025]), "WR", "ppr")

    assert projected["player_id"].tolist() == ["P1"]
    assert projected[mod._EXPERT_PRED_COL].tolist() == [12.5]
    assert projected["expert_pred_receiving_tds"].tolist() == [0.8]


def test_build_expert_sources_rejects_reserved_and_duplicate_local_names(tmp_path):
    path = tmp_path / "expert.csv"
    pd.DataFrame(
        {
            "player_id": ["P1"],
            "season": [2025],
            "week": [1],
            "projection": [1.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="reserved or duplicated"):
        mod.build_expert_sources(local_experts=[mod.parse_local_expert_spec(f"sleeper={path}")])
    with pytest.raises(ValueError, match="reserved or duplicated"):
        mod.build_expert_sources(
            local_experts=[
                mod.parse_local_expert_spec(f"fantasypros={path}"),
                mod.parse_local_expert_spec(f"fantasypros={path}"),
            ]
        )


def test_context_slice_masks_add_market_and_favorite_buckets():
    df = pd.DataFrame(
        {
            "player_id": [f"P{i}" for i in range(8)],
            "season": [2025] * 8,
            "week": [1] * 8,
            "fantasy_points": list(range(8)),
            "pred_total": list(range(8)),
            "implied_team_total": [14, 16, 18, 20, 22, 24, 26, 28],
            "implied_opp_total": [20, 20, 20, 20, 20, 20, 20, 20],
            "total_line": [35, 37, 39, 41, 43, 45, 47, 49],
        }
    )

    masks = {(family, name): mask for family, name, mask in mod.context_slice_masks(df)}

    assert ("market", "implied_team_total_low_q1") in masks
    assert ("market", "implied_team_total_high_q4") in masks
    assert masks[("market", "implied_team_total_low_q1")].sum() == 2
    assert masks[("market", "implied_team_total_high_q4")].sum() == 2
    assert masks[("market", "underdog_by_3plus")].sum() == 2
    assert masks[("market", "favorite_by_3plus")].sum() == 3
    assert masks[("market", "pickem_within_3")].sum() == 3


def test_context_slice_masks_keep_tied_market_values_together():
    df = pd.DataFrame(
        {
            "player_id": [f"P{i}" for i in range(8)],
            "season": [2025] * 8,
            "week": [1] * 8,
            "fantasy_points": list(range(8)),
            "pred_total": list(range(8)),
            "implied_team_total": [10, 10, 10, 10, 20, 30, 40, 50],
        }
    )

    masks = {(family, name): mask for family, name, mask in mod.context_slice_masks(df)}

    assert masks[("market", "implied_team_total_low_q1")].sum() == 4
    assert masks[("market", "implied_team_total_high_q4")].sum() == 2


def test_context_slice_masks_derives_opponent_total_from_total_line():
    df = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3"],
            "season": [2025] * 3,
            "week": [1] * 3,
            "fantasy_points": [1.0, 2.0, 3.0],
            "pred_total": [1.0, 2.0, 3.0],
            "implied_team_total": [24.0, 20.0, 21.0],
            "total_line": [42.0, 43.0, 42.0],
        }
    )

    masks = {(family, name): mask for family, name, mask in mod.context_slice_masks(df)}

    assert masks[("market", "favorite_by_3plus")].tolist() == [True, False, False]
    assert masks[("market", "underdog_by_3plus")].tolist() == [False, True, False]
    assert masks[("market", "pickem_within_3")].tolist() == [False, False, True]


def test_slice_masks_treat_explicit_absence_flag_as_authoritative():
    df = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4"],
            "season": [2025] * 4,
            "week": [2] * 4,
            "fantasy_points": [1.0, 2.0, 3.0, 4.0],
            "pred_total": [1.0, 2.0, 3.0, 4.0],
            "days_rest": [10, 14, 21, 14],
            "is_returning_from_absence": [0, 0, 1, 1],
        }
    )

    masks = {(family, name): mask for family, name, mask in mod.slice_masks("WR", df)}

    assert masks[("availability", "returning")].tolist() == [False, False, True, True]
    assert masks[("availability", "returning_1wk")].tolist() == [False, False, False, True]
    assert masks[("availability", "returning_2plus")].tolist() == [False, False, True, False]


def test_slice_masks_falls_back_to_absence_length_rest_without_flag():
    df = pd.DataFrame(
        {
            "player_id": ["P1", "P2"],
            "season": [2025, 2025],
            "week": [2, 2],
            "fantasy_points": [1.0, 2.0],
            "pred_total": [1.0, 2.0],
            "days_rest": [10, 14],
        }
    )

    masks = {(family, name): mask for family, name, mask in mod.slice_masks("WR", df)}

    assert masks[("availability", "returning")].tolist() == [False, True]


def test_td_calibration_rows_score_positive_probability():
    df = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4"],
            "receiving_tds": [0.0, 1.0, 0.0, 2.0],
            "pred_attn_nn_total": [5.0, 6.0, 7.0, 8.0],
            "pred_attn_nn_receiving_tds": [0.1, 0.8, 0.2, 1.5],
        }
    )
    meta = mod.SourceMeta(
        name="attn_nn",
        label="Attention NN",
        kind="model",
        native_col="pred_attn_nn_total",
    )

    rows = mod.td_calibration_rows("WR", meta, df)

    assert len(rows) == 1
    row = rows[0]
    assert row["metric_group"] == "td_calibration"
    assert row["target"] == "receiving_tds"
    assert row["n_rows"] == 4
    assert row["actual_positive_rate"] == pytest.approx(0.5)
    assert row["pred_mean"] == pytest.approx((0.1 + 0.8 + 0.2 + 1.5) / 4.0)
    assert row["brier_td_positive"] >= 0.0
    assert row["auc_td_positive"] == pytest.approx(1.0)


def test_td_calibration_rows_scores_local_expert_raw_td_probability():
    df = pd.DataFrame(
        {
            "player_id": ["P1", "P2"],
            "receiving_tds": [0.0, 1.0],
            "expert_pred_total": [5.0, 12.0],
            "expert_pred_receiving_tds": [0.1, 0.9],
        }
    )
    meta = mod.SourceMeta(
        name="fantasypros",
        label="FantasyPros",
        kind="expert",
        native_col=mod._EXPERT_PRED_COL,
    )

    rows = mod.td_calibration_rows("WR", meta, df)

    assert len(rows) == 1
    assert rows[0]["source"] == "fantasypros"
    assert rows[0]["target"] == "receiving_tds"
    assert rows[0]["auc_td_positive"] == pytest.approx(1.0)


def test_parse_args_defaults_to_artifacts_with_validation():
    args = mod._parse_args([])
    assert args.from_artifacts is True
    assert args.validate is True

    fresh = mod._parse_args(["--fresh", "--no-validate"])
    assert fresh.from_artifacts is False
    assert fresh.validate is False


def test_load_position_predictions_validates_artifacts_before_trusting_them():
    artifact_df = pd.DataFrame(
        {"player_id": ["P1"], "season": [2025], "week": [1], "fantasy_points": [10.0]}
    )
    fresh_df = pd.DataFrame(
        {"player_id": ["P2"], "season": [2025], "week": [1], "fantasy_points": [20.0]}
    )
    calls = {"fresh": 0}

    def _builder(*args, **kwargs):
        return artifact_df

    def _fresh(*args, **kwargs):
        calls["fresh"] += 1
        return fresh_df

    ok = mod.load_position_predictions(
        "WR",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        eval_seasons=(2025,),
        scoring_format="ppr",
        from_artifacts=True,
        validate=True,
        artifact_builder=_builder,
        artifact_validator=lambda *args, **kwargs: {"status": "ok"},
        fresh_loader=_fresh,
    )
    assert ok.mode == "artifacts"
    assert ok.df.iloc[0]["player_id"] == "P1"
    assert calls["fresh"] == 0

    stale = mod.load_position_predictions(
        "WR",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        eval_seasons=(2025,),
        scoring_format="ppr",
        from_artifacts=True,
        validate=True,
        artifact_builder=_builder,
        artifact_validator=lambda *args, **kwargs: {"status": "fail"},
        fresh_loader=_fresh,
    )
    assert stale.mode == "fresh_fallback"
    assert stale.df.iloc[0]["player_id"] == "P2"
    assert "artifact validation fail" in stale.artifact_error
    assert calls["fresh"] == 1
