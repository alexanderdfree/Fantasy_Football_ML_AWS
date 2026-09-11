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
from src.shared.comparison_scoring import ACTUAL_BASIS, scoring_components

pytestmark = pytest.mark.unit

_MODELS = ("ridge", "nn", "attn_nn", "lgbm")
_EXPERTS = ("nflcom", "rotowire")


def comparison_column(source):
    return f"{source}_pred_ppr" if source in _MODELS else f"{source}_comparison_pred_ppr"


def records(positions=("WR",), weeks=(1, 2), players=3):
    """Known component truth, deliberately different from full-fantasy actuals."""
    raw = {
        "passing_yards": 250.0,
        "passing_tds": 2.0,
        "interceptions": 1.0,
        "rushing_yards": 40.0,
        "rushing_tds": 1.0,
        "receiving_yards": 50.0,
        "receiving_tds": 1.0,
        "receptions": 5.0,
        "fumbles_lost": 1.0,
    }
    rows = []
    for position in positions:
        for week in weeks:
            for player in range(players):
                row = {
                    "player_id": f"{position}-{player}",
                    "position": position,
                    "season": 2025,
                    "season_type": "REG",
                    "week": week,
                    "fantasy_points": 9999.0,
                    **{
                        f"actual_{name}": raw.get(name, 0.0)
                        for name in scoring_components(position)
                    },
                }
                for fmt, reception_weight in (("ppr", 1.0), ("half_ppr", 0.5), ("standard", 0.0)):
                    truth = {
                        "QB": 24,
                        "RB": 19 + 5 * reception_weight,
                        "WR": 9 + 5 * reception_weight,
                        "TE": 9 + 5 * reception_weight,
                        "K": 0,
                        "DST": 5,
                    }[position]
                    row.update(
                        {f"{model}_pred_{fmt}": truth + 1 + reception_weight for model in _MODELS}
                    )
                    row.update(
                        {
                            f"{expert}_pred_{fmt}": truth + 2 + reception_weight
                            for expert in (*_EXPERTS, "espn")
                        }
                    )
                    row.update(
                        {
                            f"{expert}_comparison_pred_{fmt}": truth + 2 + reception_weight
                            for expert in (*_EXPERTS, "espn")
                        }
                    )
                if position == "DST":
                    row.update({f"{model}_pred_comparison": 7.0 for model in _MODELS})
                    row.update({f"{expert}_pred_comparison": 8.0 for expert in (*_EXPERTS, "espn")})
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate(monkeypatch, data, scoring="ppr", group="offense", season=None):
    monkeypatch.setattr(timeline.core, "_get_data", lambda _scoring: (data, {}))
    return timeline.compute_timeline(scoring, group, season)


class TestTimelineEndpoint:
    def test_payload_shape(self, client_with_data):
        data = client_with_data.get("/api/timeline").get_json()
        assert data["edge_basis"] == "common_rows_per_model"
        assert data["actual_basis"] == ACTUAL_BASIS
        assert data["schema_version"] == 2
        assert data["model_labels"]["attn_nn"] == "Attention NN"
        assert isinstance(data["releases"], list)
        weekly = data["weekly"]
        assert weekly, "synthetic cache has weeks 1-7"
        for entry in weekly:
            assert set(("week", "n", "cohort_n", "source_n", "mae", "edges")).issubset(entry)
            assert "winner" not in entry
            for src in (*_MODELS, *_EXPERTS):
                assert src in entry["mae"]

    def test_summary_is_consistent_with_weekly(self, client_with_data):
        data = client_with_data.get("/api/timeline").get_json()
        weekly, summary = data["weekly"], data["summary"]
        assert summary["total_weeks"] == len(weekly)
        assert "champion" not in summary
        for model in _MODELS:
            report = summary["models"][model]
            assert report["beat_experts"] == sum(w["edges"][model] > 0 for w in weekly)
            assert report["evaluated_weeks"] == len(weekly)

    def test_scoring_routes_to_format_slice(self, client_with_data):
        # The synthetic cache builds each format at a different multiplier, so
        # the weekly MAEs must differ across formats (proves the scoring param
        # reaches the cache slot, mirroring the other format-aware endpoints).
        ppr = client_with_data.get("/api/timeline?scoring=ppr").get_json()["weekly"]
        std = client_with_data.get("/api/timeline?scoring=standard").get_json()["weekly"]
        assert any(
            a["mae"]["ridge"] != b["mae"]["ridge"]
            for a, b in zip(ppr, std, strict=True)
            if a["n"] and b["n"]
        )

    @pytest.mark.parametrize("query", ["group=ALL", "season=invalid"])
    def test_invalid_selection(self, client_with_data, query):
        assert client_with_data.get(f"/api/timeline?{query}").status_code == 400

    def test_group_and_season_reach_the_evaluator(self, client_with_data, monkeypatch):
        data = records(positions=("K",))
        monkeypatch.setattr(timeline.core, "_get_data", lambda _scoring: (data, {}))
        payload = client_with_data.get(
            "/api/timeline?group=k&season=2025&scoring=standard"
        ).get_json()
        assert payload["group"] == "k" and payload["season"] == 2025
        assert payload["sources"] == [*_MODELS, "espn"]
        assert payload["summary"]["mae"]["ridge"] == 1


@pytest.mark.parametrize("fmt,expected", [("ppr", 2.0), ("half_ppr", 1.5), ("standard", 1.0)])
@pytest.mark.parametrize(
    "pos,group",
    [
        ("QB", "offense"),
        ("RB", "offense"),
        ("WR", "offense"),
        ("TE", "offense"),
        ("K", "k"),
        ("DST", "dst"),
    ],
)
def test_shared_components_for_every_position_and_format(monkeypatch, fmt, expected, pos, group):
    data = records(positions=(pos,))
    payload = evaluate(monkeypatch, data, fmt, group)
    assert payload["summary"]["n"] == 6
    assert all(
        payload["summary"]["mae"][model] == (2.0 if pos == "DST" else expected) for model in _MODELS
    )
    assert payload["summary"]["edges"] == dict.fromkeys(_MODELS, 1.0)


def test_same_player_ids_for_every_metric_despite_different_source_populations(monkeypatch):
    data = records(weeks=(1,))
    data.loc[0, comparison_column("nflcom")] = np.nan
    data.loc[1, comparison_column("rotowire")] = np.nan
    # Disjoint missing players, not merely equal counts. Poison excluded rows.
    data.loc[:1, "ridge_pred_ppr"] = 10000
    payload = evaluate(monkeypatch, data)
    row = payload["weekly"][0]
    assert row["n"] == 1 and row["cohort_n"] == 3
    assert row["source_n"]["ridge"] == 3
    assert row["source_n"]["nflcom"] == row["source_n"]["rotowire"] == 2
    assert all(row["mae"][model] == 2 for model in _MODELS)
    assert row["edges"] == dict.fromkeys(_MODELS, 1.0)


def test_extra_actual_components_and_full_totals_cannot_change_results_or_cache(monkeypatch):
    data = records()
    expected = evaluate(monkeypatch, data)
    data["actual_rushing_yards"] = 10000
    data["actual_rushing_tds"] = 100
    data["fantasy_points"] = -10000
    before = data.copy(deep=True)
    assert evaluate(monkeypatch, data) == expected
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("position,group", [("WR", "offense"), ("DST", "dst")])
def test_native_forecasts_cannot_change_shared_comparison_errors(monkeypatch, position, group):
    data = records(positions=(position,))
    expected = evaluate(monkeypatch, data, group=group)
    for source in (*_EXPERTS, "espn", *(_MODELS if position == "DST" else ())):
        data[f"{source}_pred_ppr"] = 99999.0
    assert evaluate(monkeypatch, data, group=group) == expected
    comparison = "rotowire_pred_comparison" if position == "DST" else comparison_column("rotowire")
    unavailable = evaluate(monkeypatch, data.drop(columns=comparison), group=group)
    assert unavailable["summary"]["n"] == 0
    assert unavailable["summary"]["status"] == "unavailable"


def test_timeline_rejects_cached_backfilled_nflcom_comparison_totals(monkeypatch):
    data = records().assign(season=2023)
    payload = evaluate(monkeypatch, data)
    assert payload["summary"]["n"] == 0
    assert "nflcom" in payload["summary"]["unavailable_sources"]


@pytest.mark.parametrize("missing", [None, np.nan, np.inf])
def test_missing_component_never_falls_back_to_full_actuals(monkeypatch, missing):
    data = records()
    if missing is None:
        data = data.drop(columns="actual_fumbles_lost")
    else:
        data["actual_fumbles_lost"] = missing
    payload = evaluate(monkeypatch, data)
    assert payload["summary"]["reason"] == "shared_actual_components_missing"
    assert all(w["n"] == 0 and all(v is None for v in w["mae"].values()) for w in payload["weekly"])
    assert payload["summary"]["models"]["ridge"]["evaluated_weeks"] == 0


def test_kicker_and_dst_comparisons_keep_their_own_compatible_sources(monkeypatch):
    data = records(positions=("WR", "K", "DST"))
    data.loc[data.position.eq("K"), "nflcom_pred_ppr"] = 99999
    data.loc[data.position.eq("K"), "rotowire_pred_ppr"] = np.nan
    data.loc[data.position.eq("DST"), "nflcom_pred_ppr"] = np.nan
    for group, position in (("k", "K"), ("dst", "DST")):
        payload = evaluate(monkeypatch, data, group=group)
        assert payload["positions"] == (position,)
        assert payload["summary"]["n"] == 6
        assert "nflcom" not in payload["sources"]
        assert "nflcom" in payload["excluded_sources"]
        assert payload["summary"]["mae"]["ridge"] == 2


@pytest.mark.parametrize("source", ["rotowire", "attn_nn"])
def test_missing_required_week_does_not_relax_source_set_or_count_as_a_win(monkeypatch, source):
    data = records()
    data.loc[data.week.eq(2), comparison_column(source)] = np.nan
    payload = evaluate(monkeypatch, data)
    assert payload["weekly"][1]["n"] == 0
    assert payload["weekly"][1]["unavailable_sources"] == [source]
    assert all(edge is None for edge in payload["weekly"][1]["edges"].values())
    assert payload["summary"]["models"]["ridge"]["beat_experts"] == 1
    assert payload["summary"]["models"]["ridge"]["evaluated_weeks"] == 1
    data = data.drop(columns=comparison_column(source))
    assert evaluate(monkeypatch, data)["summary"]["n"] == 0


def test_zero_forecasts_are_valid_but_infinity_is_not(monkeypatch):
    data = records(weeks=(1,))
    for source in (*_MODELS, *_EXPERTS):
        data[comparison_column(source)] = 0.0
    data.loc[0, comparison_column("nflcom")] = np.inf
    payload = evaluate(monkeypatch, data)
    assert payload["summary"]["n"] == 2
    assert set(payload["summary"]["mae"].values()) == {14.0}
    assert payload["summary"]["models"]["ridge"]["beat_experts"] == 0


def test_missing_position_week_remains_an_explicit_gap(monkeypatch):
    data = records(positions=("WR", "K"))
    data = data[~(data.position.eq("K") & data.week.eq(2))]
    payload = evaluate(monkeypatch, data, group="k")
    assert [w["week"] for w in payload["weekly"]] == [1, 2]
    assert payload["weekly"][1]["reason"] == "no_regular_season_rows"
    assert payload["summary"]["evaluated_weeks"] == 1


def test_alternating_winners_never_create_an_oracle_model_record(monkeypatch):
    data = records(players=1)
    data["ridge_pred_ppr"] = [14.0, 18.0]
    data["nn_pred_ppr"] = [18.0, 14.0]
    data["attn_nn_pred_ppr"] = data["lgbm_pred_ppr"] = 18.0
    data[comparison_column("nflcom")] = data[comparison_column("rotowire")] = 16.0
    summary = evaluate(monkeypatch, data)["summary"]
    assert "champion" not in summary and "best_mae" not in summary
    for model in ("ridge", "nn"):
        assert summary["models"][model] == {
            "mae": 2.0,
            "edge": 0.0,
            "beat_experts": 1,
            "evaluated_weeks": 2,
        }
    assert max(model["beat_experts"] for model in summary["models"].values()) == 1


def test_season_mae_pools_player_errors_instead_of_averaging_week_means(monkeypatch):
    data = records()
    data = data[(data.week == 2) | (data.player_id == "WR-0")].copy()
    data["ridge_pred_ppr"] = np.where(data.week == 1, 24.0, 15.0)
    assert evaluate(monkeypatch, data)["summary"]["mae"]["ridge"] == 3.25


def test_seasons_and_postseason_never_share_a_week(monkeypatch):
    data = records()
    older = data.assign(season=2024, ridge_pred_ppr=114.0)
    postseason = data.assign(week=19, season_type="POST", ridge_pred_ppr=10000.0)
    data = pd.concat([data, older, postseason], ignore_index=True)
    current = evaluate(monkeypatch, data)
    assert current["season"] == 2025 and current["seasons"] == [2024, 2025]
    assert current["summary"]["n"] == 6 and len(current["weekly"]) == 2
    assert evaluate(monkeypatch, data, season=2024)["summary"]["mae"]["ridge"] == 100.0
    assert evaluate(monkeypatch, data, season=2023)["summary"]["reason"] == "no_regular_season_rows"


def test_empty_cache_is_explicit_and_json_safe(monkeypatch):
    payload = evaluate(monkeypatch, pd.DataFrame())
    assert payload["weekly"] == [] and payload["season"] is None
    assert payload["summary"]["status"] == "unavailable"
    json.dumps(payload, allow_nan=False)


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
