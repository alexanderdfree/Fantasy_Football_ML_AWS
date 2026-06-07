"""Unit tests for the consolidated cohort analysis registry and new predicates."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import cohort_analysis as ca

pytestmark = pytest.mark.unit


def test_registry_covers_expected_cohorts_and_import_is_cheap():
    assert not hasattr(ca, "run")
    assert {
        "ascension",
        "injury_return",
        "rookie",
        "committee",
        "trade",
        "sparse_history",
        "late_week",
        "suspension_return",
        "scoring_tier",
    } <= set(ca.COHORTS)
    assert ca.COHORTS["suspension_return"].feasible is False


def test_prior_season_fp_attaches_S_minus_1_mean_to_season_S():
    frame = pd.DataFrame(
        {
            "player_id": ["A", "A", "B", "B"],
            "season": [2023, 2023, 2023, 2023],
            "fantasy_points": [10.0, 20.0, 4.0, 6.0],
        }
    )
    prior = ca.player_prior_season_fp([frame])
    # 2023 mean attaches to 2024.
    assert prior.loc[("A", 2024)] == pytest.approx(15.0)
    assert prior.loc[("B", 2024)] == pytest.approx(5.0)
    # No prior expectation exists for the first observed season.
    assert ("A", 2023) not in prior.index


def test_scoring_tier_labels_top_n_per_position_season_as_elite():
    # Two QBs and two RBs in 2024; top-1 per (season, position) is elite.
    df = pd.DataFrame(
        {
            "player_id": ["A", "B", "C", "D", "ROOK"],
            "season": [2024, 2024, 2024, 2024, 2024],
            "position": ["QB", "QB", "RB", "RB", "QB"],
            "week": [1, 1, 1, 1, 1],
        }
    )
    prior = pd.Series(
        {("A", 2024): 22.0, ("B", 2024): 9.0, ("C", 2024): 18.0, ("D", 2024): 7.0},
        name="prior_season_fp",
    )
    labels = ca.label_scoring_tier_rows(df, prior, top_n=1)
    assert labels.tolist() == [
        ca.TIER_ELITE,  # A: top QB
        ca.TIER_FIELD,  # B: lower QB
        ca.TIER_ELITE,  # C: top RB
        ca.TIER_FIELD,  # D: lower RB
        "unknown",  # ROOK: no prior-season expectation
    ]


def test_scoring_tier_guards_missing_proxy_and_columns():
    df = pd.DataFrame({"player_id": ["A"], "season": [2024], "position": ["QB"]})
    assert ca.label_scoring_tier_rows(df, None, top_n=1).tolist() == ["unknown"]
    empty = pd.Series(dtype=float)
    assert ca.label_scoring_tier_rows(df, empty, top_n=1).tolist() == ["unknown"]
    no_pos = df.drop(columns=["position"])
    prior = pd.Series({("A", 2024): 22.0})
    assert ca.label_scoring_tier_rows(no_pos, prior, top_n=1).tolist() == ["unknown"]


def test_committee_label_requires_two_mid_share_backs():
    df = pd.DataFrame(
        {
            "position": ["RB", "RB", "RB", "RB"],
            "recent_team": ["KC", "KC", "BUF", "BUF"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 1, 1],
            "game_carry_share": [0.55, 0.45, 0.90, 0.10],
        }
    )
    assert ca.label_committee_rows(df).tolist() == [
        "committee",
        "committee",
        "non_committee",
        "non_committee",
    ]


def test_committee_can_derive_share_from_carries_and_guards_missing_keys():
    df = pd.DataFrame(
        {
            "position": ["RB", "RB", "RB"],
            "recent_team": ["KC", "KC", "KC"],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "carries": [10, 8, 1],
        }
    )
    assert ca.label_committee_rows(df).tolist() == [
        "committee",
        "committee",
        "non_committee",
    ]
    assert ca.label_committee_rows(df.drop(columns=["recent_team"])).unique().tolist() == [
        "unknown"
    ]


def test_trade_label_marks_rows_after_in_season_team_change():
    df = pd.DataFrame(
        {
            "player_id": ["P", "P", "P", "Q"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 3, 1],
            "recent_team": ["LV", "LV", "NYG", "DAL"],
        },
        index=[10, 11, 12, 13],
    )
    labels = ca.label_trade_rows(df)
    assert labels.index.tolist() == [10, 11, 12, 13]
    assert labels.tolist() == ["stable_team", "stable_team", "midseason_trade", "stable_team"]
    assert ca.label_trade_rows(df.drop(columns=["week"])).unique().tolist() == ["unknown"]


def test_injury_return_label_uses_return_flag_then_days_rest_fallback():
    flagged = pd.DataFrame({"is_returning_from_absence": [0, 1, 0]})
    assert ca.label_injury_return_rows(flagged).tolist() == ["settled", "returning", "settled"]

    rest = pd.DataFrame({"days_rest": [7, 14, 21]})
    assert ca.label_injury_return_rows(rest).tolist() == ["settled", "returning", "returning"]
    assert ca.label_injury_return_rows(pd.DataFrame({"x": [1]})).tolist() == ["unknown"]


def test_sparse_history_labels_reset_each_season_and_preserve_order():
    df = pd.DataFrame(
        {
            "player_id": ["A", "A", "A", "B"],
            "season": [2025, 2025, 2024, 2025],
            "week": [2, 1, 1, 1],
        },
        index=[4, 3, 2, 1],
    )
    labels = ca.label_sparse_history_rows(df)
    assert labels.index.tolist() == [4, 3, 2, 1]
    assert labels.tolist() == ["1", "0", "0", "0"]


def test_ascension_label_requires_prior_games_when_keys_are_available():
    df = pd.DataFrame(
        {
            "player_id": ["A", "A", "A"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "rolling_mean_carries_L3": [0.0, 4.0, 4.0],
            "rolling_mean_targets_L3": [0.0, 1.0, 1.0],
            "carries": [20, 20, 20],
            "targets": [2, 2, 2],
        }
    )
    assert ca.label_ascension_rows(df).tolist() == [
        "established",
        "established",
        "ascension",
    ]


def test_bucket_model_table_reports_delta_vs_global():
    df = pd.DataFrame(
        {
            "bucket": ["a", "a", "b", "b"],
            "fantasy_points": [10.0, 12.0, 1.0, 1.0],
            "pred_ridge_total": [9.0, 11.0, 3.0, 3.0],
        }
    )
    tbl = ca.bucket_model_table(df, "bucket", {"Ridge": "pred_ridge_total"})
    a_row = tbl[tbl["bucket"] == "a"].iloc[0]
    b_row = tbl[tbl["bucket"] == "b"].iloc[0]
    assert a_row["mae"] == pytest.approx(1.0)
    assert b_row["mae"] == pytest.approx(2.0)
    assert a_row["dmae"] == pytest.approx(-0.5)
    assert b_row["dmae"] == pytest.approx(0.5)
