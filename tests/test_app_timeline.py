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

import pytest

import src.serving.timeline as timeline

pytestmark = pytest.mark.unit

_MODELS = ("ridge", "nn", "attn_nn", "lgbm")
_EXPERTS = ("nflcom", "rotowire")


class TestTimelineEndpoint:
    def test_payload_shape(self, client_with_data):
        data = client_with_data.get("/api/timeline").get_json()
        assert data["edge_basis"] == "common_rows"
        assert data["model_labels"]["attn_nn"] == "Attention NN"
        assert isinstance(data["releases"], list)
        weekly = data["weekly"]
        assert weekly, "synthetic cache has weeks 1-7"
        for entry in weekly:
            assert set(("week", "n", "winner", "edge")).issubset(entry)
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
