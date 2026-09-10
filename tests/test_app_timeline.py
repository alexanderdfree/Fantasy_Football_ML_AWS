"""Changelog & Timeline tab: /api/timeline + the committed release changelog.

The weekly log is computed from the same synthetic cache the other endpoint
tests use; the release changelog is the committed owner-curated JSON whose
schema this suite pins (the file is hand-edited, so a malformed entry must be
caught here, not by a serving 500).
"""

from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pandas as pd
import pytest

import src.serving.timeline as timeline
from src.data.loader import compute_all_scoring_formats
from src.serving.serialization import _actual_col, _pred_col
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.comparison_scoring import ACTUAL_BASIS, scoring_components
from src.wr.targets import compute_targets

pytestmark = pytest.mark.unit

_MODELS = ("ridge", "nn", "attn_nn", "lgbm")
_EXPERTS = ("nflcom", "rotowire")


def _wr_forecasts(errors, scoring="ppr", *, receiving_yards=50.0, receptions=5.0):
    """Serving-shaped rows produced from raw WR targets and real aggregators."""
    n = len(next(iter(errors.values())))
    raw = pd.DataFrame(
        {
            "player_id": [f"wr-{i}" for i in range(n)],
            "position": "WR",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "receiving_yards": receiving_yards,
            "receptions": receptions,
        }
    )
    for col in (
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "receiving_tds",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
    ):
        raw[col] = 0.0
    frame = compute_targets(compute_all_scoring_formats(raw))
    for target in scoring_components("WR"):
        frame[f"actual_{target}"] = frame[target]
    for source, delta in errors.items():
        pred = {target: frame[target].to_numpy().copy() for target in scoring_components("WR")}
        pred["receiving_yards"] += np.asarray(delta) * 10
        frame[_pred_col(source, scoring)] = predictions_to_fantasy_points("WR", pred, scoring)
    return frame


def _compute(monkeypatch, frame, scoring="ppr"):
    monkeypatch.setattr(timeline.core, "_get_data", lambda scoring: (frame, {}))
    return timeline.compute_timeline(scoring)


class TestTimelineEndpoint:
    def test_payload_shape(self, client_with_data):
        data = client_with_data.get("/api/timeline").get_json()
        assert data["edge_basis"] == "common_rows"
        assert data["actual_basis"] == ACTUAL_BASIS
        assert data["model_labels"]["attn_nn"] == "Attention NN"
        assert isinstance(data["releases"], list)
        weekly = data["weekly"]
        assert weekly, "synthetic cache has weeks 1-7"
        for entry in weekly:
            assert set(("week", "n", "winner", "edge")).issubset(entry)
            assert 0 <= entry["n"] <= entry["cohort_n"]
            for src in (*_MODELS, *_EXPERTS):
                assert src in entry

    def test_winner_is_argmin_of_model_maes(self, client_with_data):
        weekly = client_with_data.get("/api/timeline").get_json()["weekly"]
        for entry in weekly:
            maes = {m: entry[m] for m in _MODELS if entry[m] is not None}
            assert maes, "synthetic cache carries all four models"
            assert entry["winner"] == min(maes, key=maes.get)

    def test_summary_is_consistent_with_weekly(self, client_with_data):
        data = client_with_data.get("/api/timeline").get_json()
        weekly, summary = data["weekly"], data["summary"]
        assert summary["total_weeks"] == len(weekly)
        wins = [w for w in weekly if w["winner"] == summary["champion"]]
        assert summary["champion_weeks"] == len(wins)
        assert 0 <= summary["beat_experts"] <= summary["total_weeks"]
        best = min(w[w["winner"]] for w in weekly if w["winner"])
        assert summary["best_mae"] == pytest.approx(best)

    def test_scoring_routes_to_format_slice(self, client_with_data):
        # The synthetic cache builds each format at a different multiplier, so
        # the weekly MAEs must differ across formats (proves the scoring param
        # reaches the cache slot, mirroring the other format-aware endpoints).
        ppr = client_with_data.get("/api/timeline?scoring=ppr").get_json()["weekly"]
        std = client_with_data.get("/api/timeline?scoring=standard").get_json()["weekly"]
        assert any(
            a["ridge"] != b["ridge"]
            for a, b in zip(ppr, std, strict=True)
            if a["ridge"] and b["ridge"]
        )


@pytest.mark.parametrize("missing_expert", ["nflcom", "rotowire", None])
def test_beating_both_experts_requires_both_comparisons(monkeypatch, missing_expert):
    row = {
        "position": "WR",
        "actual_receiving_yards": 50.0,
        "actual_receptions": 5.0,
        "actual_receiving_tds": 0.0,
        "actual_fumbles_lost": 0.0,
        "week": 1,
        "season": 2025,
        "fantasy_points": 10.0,
        "ridge_pred_ppr": 9.0,
        "nn_pred_ppr": 8.0,
        "attn_nn_pred_ppr": 7.0,
        "lgbm_pred_ppr": 6.0,
        "nflcom_pred_ppr": 14.0,
        "rotowire_pred_ppr": 15.0,
    }
    if missing_expert:
        row[f"{missing_expert}_pred_ppr"] = float("nan")
    monkeypatch.setattr(timeline.core, "_get_data", lambda scoring: (pd.DataFrame([row]), {}))
    result = timeline.compute_timeline("ppr")
    assert result["weekly"][0]["edge"] == (None if missing_expert else 3.0)
    assert result["summary"]["beat_experts"] == (0 if missing_expert else 1)


def test_expert_without_winner_overlap_cannot_count_as_beaten(monkeypatch):
    rows = pd.DataFrame(
        {
            "week": [1, 1],
            "position": ["WR", "WR"],
            "actual_receiving_yards": [50.0, 50.0],
            "actual_receptions": [5.0, 5.0],
            "actual_receiving_tds": [0.0, 0.0],
            "actual_fumbles_lost": [0.0, 0.0],
            "fantasy_points": [10.0, 10.0],
            "ridge_pred_ppr": [9.0, float("nan")],
            "nn_pred_ppr": [8.0, 8.0],
            "attn_nn_pred_ppr": [7.0, 7.0],
            "lgbm_pred_ppr": [6.0, 6.0],
            "nflcom_pred_ppr": [float("nan"), 14.0],
            "rotowire_pred_ppr": [15.0, float("nan")],
        }
    )
    monkeypatch.setattr(timeline.core, "_get_data", lambda scoring: (rows, {}))
    result = timeline.compute_timeline("ppr")
    assert result["weekly"][0]["winner"] is None
    assert result["weekly"][0]["n"] == 0
    assert result["weekly"][0]["edge"] is None
    assert result["summary"]["beat_experts"] == 0


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_winner_uses_identical_player_weeks(monkeypatch, missing):
    frame = _wr_forecasts(
        {
            "ridge": [1, missing],
            "nn": [0.5, 20],
            "attn_nn": [5, 5],
            "lgbm": [6, 6],
            "nflcom": [4, 4],
            "rotowire": [5, 5],
        }
    )
    before = frame.copy(deep=True)
    result = _compute(monkeypatch, frame)
    week = result["weekly"][0]
    assert week["winner"] == "nn"
    assert week["ridge"] == 1.0
    assert week["nn"] == 0.5
    assert week["n"] == 1
    assert week["cohort_n"] == 2
    assert week["source_n"]["ridge"] == 1
    assert week["source_n"]["nn"] == 2
    assert week["edge"] == 3.5
    pd.testing.assert_frame_equal(frame, before)


def test_edge_and_displayed_maes_share_one_expert_intersection(monkeypatch):
    frame = _wr_forecasts(
        {
            "ridge": [1, 1, 4],
            "nn": [10, 10, 10],
            "attn_nn": [11, 11, 11],
            "lgbm": [12, 12, 12],
            "nflcom": [10, np.nan, 1],
            "rotowire": [np.nan, 10, 1],
        }
    )
    result = _compute(monkeypatch, frame)
    week = result["weekly"][0]
    assert week["n"] == 1
    assert week["ridge"] == 4.0
    assert week["nflcom"] == week["rotowire"] == 1.0
    assert week["winner"] == "ridge"
    assert week["edge"] == -3.0
    assert result["summary"]["beat_experts"] == 0


@pytest.mark.parametrize("scoring", ["ppr", "half_ppr", "standard"])
def test_wr_rushing_does_not_change_shared_component_truth(monkeypatch, scoring):
    # Sterling Shepard's cached 2025 Wk4 receiving line: 14 yards, 2 catches.
    # His additional 6 rushing yards are outside WR's projected components.
    frame = _wr_forecasts(
        {
            "ridge": [0],
            "nn": [0.6],
            "attn_nn": [20],
            "lgbm": [30],
            "nflcom": [10],
            "rotowire": [11],
        },
        scoring,
        receiving_yards=14.0,
        receptions=2.0,
    )
    control = _compute(monkeypatch, frame, scoring)
    frame["rushing_yards"] = 6.0
    frame = compute_all_scoring_formats(frame)
    result = _compute(monkeypatch, frame, scoring)
    assert result == control
    assert result["weekly"][0]["ridge"] == 0.0
    assert result["weekly"][0]["nn"] == 0.6
    assert result["weekly"][0]["winner"] == "ridge"


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
@pytest.mark.parametrize("scoring", ["ppr", "half_ppr", "standard"])
def test_all_positions_use_canonical_components(monkeypatch, position, scoring):
    components = {target: np.array([1.0]) for target in scoring_components(position)}
    if position == "DST":
        components.update(points_allowed=np.array([20.0]), yards_allowed=np.array([350.0]))
    truth = float(predictions_to_fantasy_points(position, components, scoring)[0])
    row = {
        "player_id": "p1",
        "position": position,
        "season": 2025,
        "week": 1,
        _actual_col(scoring): 999.0,
        **{f"actual_{target}": value[0] for target, value in components.items()},
        **{
            _pred_col(source, scoring): truth + i + 1
            for i, source in enumerate((*_MODELS, *_EXPERTS))
        },
    }
    if position == "K":
        row[_pred_col("rotowire", scoring)] = np.nan  # no RotoWire K feed
    result = _compute(monkeypatch, pd.DataFrame([row]), scoring)
    week = result["weekly"][0]
    assert week["ridge"] == 1.0
    assert week["n"] == 1
    assert result["actual_basis"] == ACTUAL_BASIS
    assert result["scoring_components"][position] == list(scoring_components(position))
    if position == "K":
        assert week["nflcom"] is None
        assert week["edge"] is None
        assert "nflcom" in result["excluded_sources"]["K"]


@pytest.mark.parametrize("missing", ["column", "value"])
def test_missing_actual_components_are_unavailable(monkeypatch, missing):
    frame = _wr_forecasts({source: [1] for source in (*_MODELS, *_EXPERTS)})
    if missing == "column":
        frame = frame.drop(columns=["actual_receptions"])
    else:
        frame["actual_receptions"] = np.nan
    result = _compute(monkeypatch, frame)
    week = result["weekly"][0]
    assert week["n"] == 0
    assert week["winner"] is None
    assert week["edge"] is None
    assert week["reason"] == "shared_actual_components_missing"
    assert all(week[source] is None for source in (*_MODELS, *_EXPERTS))


def test_absent_expert_week_does_not_change_comparison_sources(monkeypatch):
    frame = _wr_forecasts({source: [i, i] for i, source in enumerate((*_MODELS, *_EXPERTS))})
    frame["week"] = [1, 2]
    frame.loc[1, "nflcom_pred_ppr"] = np.nan
    result = _compute(monkeypatch, frame)
    available, absent = result["weekly"]
    assert available["n"] == 1
    assert available["winner"] == "ridge"
    assert available["ridge"] == 0.0  # a zero error is available
    assert absent["n"] == 0
    assert absent["cohort_n"] == 1
    assert absent["source_n"]["nflcom"] == 0
    assert absent["winner"] is None
    assert absent["edge"] is None
    assert absent["reason"] == "no_common_source_rows"


@pytest.mark.parametrize("missing", ["column", "infinite"])
def test_unavailable_expert_preserves_other_sources_without_an_edge(monkeypatch, missing):
    frame = _wr_forecasts({source: [i + 1] for i, source in enumerate((*_MODELS, *_EXPERTS))})
    if missing == "column":
        frame = frame.drop(columns=["nflcom_pred_ppr"])
    else:
        frame["nflcom_pred_ppr"] = np.inf
    # ESPN is not displayed in Timeline; its coverage must not change this set.
    frame["espn_pred_ppr"] = np.nan
    result = _compute(monkeypatch, frame)
    week = result["weekly"][0]
    assert week["n"] == 1
    assert week["winner"] == "ridge"
    assert week["ridge"] == 1.0
    assert week["nflcom"] is None
    assert week["edge"] is None
    assert "nflcom" not in result["sources"]
    assert "espn" not in result["sources"]


def test_postseason_rows_are_not_weekly_benchmarks(monkeypatch):
    frame = _wr_forecasts({source: [1, 2, 3] for source in (*_MODELS, *_EXPERTS)})
    frame["week"] = [1, 18, 19]
    frame["season"] = [2025, 2020, 2025]
    frame["season_type"] = ["REG", "REG", "POST"]
    result = _compute(monkeypatch, frame)
    assert [row["week"] for row in result["weekly"]] == [1]
    assert result["summary"]["total_weeks"] == 1


class TestReleaseChangelog:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        timeline.reset_release_cache()
        yield
        timeline.reset_release_cache()

    def test_committed_file_matches_schema(self):
        path = os.path.join(os.path.dirname(timeline.__file__), "release_changelog.json")
        with open(path) as fh:
            entries = json.load(fh)
        assert entries, "seeded changelog must not be empty"
        for e in entries:
            assert timeline._RELEASE_REQUIRED_KEYS.issubset(e), e.get("version")
            # date must parse ISO and family must key into the model hues.
            datetime.date.fromisoformat(e["date"])
            assert e["family"] in timeline.MODEL_LABELS
            assert isinstance(e["mae"], int | float)
            assert e.get("prev_mae") is None or isinstance(e["prev_mae"], int | float)

    def test_loader_sorts_newest_first_and_drops_malformed(self, tmp_path, monkeypatch):
        bad = [
            {
                "version": "v1",
                "date": "2026-01-01",
                "family": "nn",
                "model": "NN",
                "title": "a",
                "summary": "b",
                "mae": 5.0,
            },
            {"not": "a release"},
            {
                "version": "v2",
                "date": "2026-03-01",
                "family": "attn_nn",
                "model": "Attn",
                "title": "c",
                "summary": "d",
                "mae": 4.5,
            },
        ]
        p = tmp_path / "release_changelog.json"
        p.write_text(json.dumps(bad))
        monkeypatch.setattr(timeline, "_RELEASE_CHANGELOG_PATH", str(p))
        out = timeline.load_release_changelog()
        assert [e["version"] for e in out] == ["v2", "v1"]

    def test_missing_file_degrades_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(timeline, "_RELEASE_CHANGELOG_PATH", str(tmp_path / "absent.json"))
        assert timeline.load_release_changelog() == []
