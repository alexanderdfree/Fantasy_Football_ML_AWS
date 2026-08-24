"""Expert projections (NFL.com / RotoWire / ESPN) on the upcoming-week artifact.

Covers the Phase-5 plumbing: best-effort fetch wrappers, the change-detection
digest that folds experts into the rebuild signature, the in-place projection
apply, and the serialized row contract (nflcom_pred / rotowire_pred / espn_pred keys
that the homepage feature-detects).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.serving.upcoming_week as uw
from src.serving.serialization import _pred_col

pytestmark = pytest.mark.unit


def _upcoming_results():
    frame = pd.DataFrame(
        {
            "player_id": ["QB1", "WR1"],
            "player_display_name": ["Quarter Back", "Wide Out"],
            "position": ["QB", "WR"],
            "recent_team": ["KC", "BUF"],
            "season": [2026, 2026],
            "week": [12, 12],
            "opponent_team": ["BUF", "KC"],
            "is_home": [1, 0],
            "headshot_url": ["", ""],
        }
    )
    for prefix in ("ridge", "nn", "attn_nn", "lgbm"):
        for fmt in uw._VALID_SCORING:
            frame[_pred_col(prefix, fmt)] = 10.0
    return frame


def _rotowire_raw():
    # Shape mirrors load_sleeper_with_gsis_id output: expert key cols +
    # position + raw-stat target columns (missing targets zero-fill).
    return pd.DataFrame(
        {
            "player_id": ["QB1"],
            "season": [2026],
            "week": [12],
            "position": ["QB"],
            "passing_yards": [300.0],
            "passing_tds": [2.0],
        }
    )


class TestApplyUpcomingExperts:
    def test_rotowire_rows_project_and_join(self):
        results = _upcoming_results()
        uw._apply_upcoming_experts(results, None, _rotowire_raw(), None)
        # 300 pass yds (0.04/yd) + 2 pass TDs (4 pts) = 20.0 under every format
        # (QB scoring has no reception term).
        val = results.loc[results["player_id"] == "QB1", _pred_col("rotowire", "ppr")].iloc[0]
        assert val == pytest.approx(20.0, abs=0.5)
        # The WR had no RotoWire row -> stays NaN; NFL.com feed absent -> NaN.
        assert np.isnan(
            results.loc[results["player_id"] == "WR1", _pred_col("rotowire", "ppr")].iloc[0]
        )
        assert results[_pred_col("nflcom", "ppr")].isna().all()

    def test_missing_feeds_leave_stable_nan_columns(self):
        results = _upcoming_results()
        uw._apply_upcoming_experts(results, None, None, None)
        for source in uw._UPCOMING_EXPERTS:
            for fmt in uw._VALID_SCORING:
                assert results[_pred_col(source, fmt)].isna().all()

    def test_rows_serialize_expert_keys(self):
        results = _upcoming_results()
        uw._apply_upcoming_experts(results, None, _rotowire_raw(), None)
        rows = uw._results_to_upcoming_rows(results, scoring="ppr")
        qb = next(r for r in rows if r["player_id"] == "QB1")
        wr = next(r for r in rows if r["player_id"] == "WR1")
        assert qb["rotowire_pred"] == pytest.approx(20.0, abs=0.5)
        assert qb["nflcom_pred"] is None
        assert wr["rotowire_pred"] is None


class TestFetchWrappers:
    def test_loader_failure_degrades_to_none(self):
        def boom(*a, **k):
            raise RuntimeError("feed down")

        nfl, rw, espn = uw._fetch_upcoming_expert_frames(
            2026, 12, nflcom_loader=boom, rotowire_loader=boom, espn_loader=boom
        )
        assert nfl is None and rw is None and espn is None

    def test_empty_or_malformed_frames_degrade_to_none(self):
        empty = pd.DataFrame()
        no_pos = pd.DataFrame({"player_id": ["x"]})
        nfl, rw, espn = uw._fetch_upcoming_expert_frames(
            2026,
            12,
            nflcom_loader=lambda *a, **k: empty,
            rotowire_loader=lambda *a, **k: no_pos,
            espn_loader=lambda *a, **k: empty,
        )
        assert nfl is None and rw is None and espn is None

    def test_loaders_receive_single_week(self):
        seen = {}

        def spy(seasons, *, weeks=None, force_refresh=False, **k):
            seen["seasons"] = list(seasons)
            seen["weeks"] = list(weeks)
            seen["force_refresh"] = force_refresh
            return None

        uw._fetch_upcoming_expert_frames(
            2026, 12, nflcom_loader=spy, rotowire_loader=spy, espn_loader=lambda *a, **k: None
        )
        assert seen == {"seasons": [2026], "weeks": [12], "force_refresh": True}


class TestExpertDigest:
    def test_digest_tracks_content_changes(self):
        raw = _rotowire_raw()
        d1 = uw._expert_digest(None, raw, None)
        bumped = raw.copy()
        bumped.loc[0, "passing_yards"] = 310.0
        d2 = uw._expert_digest(None, bumped, None)
        assert d1 != d2

    def test_digest_marks_missing_feeds(self):
        assert uw._expert_digest(None, None, None) == "nfl:none|rw:none|espn:none"

    def test_digest_feeds_input_signature(self):
        # Same inputs, different expert digest -> different rebuild signature.
        slate = pd.DataFrame(
            {
                "recent_team": ["KC"],
                "opponent_team": ["BUF"],
                "is_home": [1],
                "spread_line": [-3.0],
                "total_line": [47.5],
            }
        )
        roster = pd.DataFrame({"player_id": ["QB1"]})
        s1 = uw._input_signature(2026, 12, slate, roster, expert_digest="a")
        s2 = uw._input_signature(2026, 12, slate, roster, expert_digest="b")
        assert s1 != s2
