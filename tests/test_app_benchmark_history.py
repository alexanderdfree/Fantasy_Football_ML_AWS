"""Flask API contract tests for /api/benchmark_history.

Mirrors the marker convention in tests/test_app.py and tests/test_wiki.py:
these cross the Flask boundary via test_client() and so are tagged
``integration``.

The endpoint reads benchmark_history/*.json off disk — we monkeypatch the
dir-resolution helper so each test exercises a self-contained fixture tree
in tmp_path, never the repo's real history.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration


def _write_run(history_dir, *, ts, sha, pr=None, results, backend="ec2"):
    entry = {
        "run_id": f"{ts}_{sha}",
        "timestamp": ts,
        "git_hash": sha,
        "note": f"EC2 auto-run ({sha})",
        "backend": backend,
        "instance_type": "g4dn.xlarge (On-Demand)",
        "positions": [r["position"] for r in results],
        "results": results,
    }
    if pr is not None:
        entry["pr_number"] = pr
    path = history_dir / f"{ts.replace(':', '-')}_{sha}.json"
    path.write_text(json.dumps(entry))
    return path


@pytest.fixture
def history_client(app_module, tmp_path, monkeypatch):
    """Flask test client wired to a tmp benchmark_history/ dir."""
    history_dir = tmp_path / "benchmark_history"
    history_dir.mkdir()
    monkeypatch.setattr(app_module, "_benchmark_history_dir", lambda: str(history_dir))
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c, history_dir


class TestEmpty:
    def test_returns_200_with_no_rows_when_dir_empty(self, history_client):
        client, _ = history_client
        r = client.get("/api/benchmark_history")
        assert r.status_code == 200
        body = r.get_json()
        assert body["rows"] == []
        assert body["repo_slug"]  # any non-empty slug

    def test_returns_200_when_dir_missing(self, app_module, monkeypatch):
        # Simulate a brand-new container that hasn't synced yet.
        monkeypatch.setattr(app_module, "_benchmark_history_dir", lambda: "/nonexistent/path")
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            r = c.get("/api/benchmark_history")
        assert r.status_code == 200
        assert r.get_json()["rows"] == []


class TestRowShape:
    def test_one_row_per_file_newest_first(self, history_client):
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-04-01T12:00:00",
            sha="aaa1111",
            pr=190,
            results=[{"position": "QB", "ridge_mae": 5.0, "elapsed_sec": 30.0}],
        )
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="dff43fb",
            pr=199,
            results=[{"position": "K", "ridge_mae": 6.668, "elapsed_sec": 84.0}],
        )
        rows = client.get("/api/benchmark_history").get_json()["rows"]
        assert len(rows) == 2
        assert rows[0]["timestamp"] == "2026-05-19T22:47:20"
        assert rows[0]["pr_number"] == 199
        assert rows[1]["timestamp"] == "2026-04-01T12:00:00"

    def test_mae_pills_carry_position_and_value(self, history_client):
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="dff43fb",
            pr=199,
            results=[
                {
                    "position": "K",
                    "ridge_mae": 6.668,
                    "nn_mae": 7.133,
                    "attn_nn_mae": 7.12,
                    "lgbm_mae": 7.188,
                    "elapsed_sec": 84.0,
                },
            ],
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        assert row["ridge"] == [{"position": "K", "mae": 6.668}]
        assert row["nn"] == [{"position": "K", "mae": 7.133}]
        assert row["attn_nn"] == [{"position": "K", "mae": 7.12}]
        assert row["lgbm"] == [{"position": "K", "mae": 7.188}]
        assert row["total_elapsed_sec"] == 84.0

    def test_multi_position_run_stacks_pills(self, history_client):
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-01T10:00:00",
            sha="abc1234",
            pr=180,
            results=[
                {"position": "QB", "ridge_mae": 4.1, "nn_mae": 3.9, "elapsed_sec": 50.0},
                {"position": "RB", "ridge_mae": 5.2, "nn_mae": 4.8, "elapsed_sec": 70.0},
            ],
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        assert row["ridge"] == [
            {"position": "QB", "mae": 4.1},
            {"position": "RB", "mae": 5.2},
        ]
        assert row["positions"] == ["QB", "RB"]
        assert row["total_elapsed_sec"] == 120.0

    def test_missing_model_mae_skipped_not_nulled(self, history_client):
        """K and DST runs historically had no attn_nn/lgbm MAE — those pills
        should simply be absent from the cell, not rendered as null entries
        that the frontend has to filter."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="abc1234",
            pr=199,
            results=[{"position": "K", "ridge_mae": 6.5, "elapsed_sec": 80.0}],
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        assert row["ridge"] == [{"position": "K", "mae": 6.5}]
        assert row["nn"] == []
        assert row["attn_nn"] == []
        assert row["lgbm"] == []


class TestPrFallback:
    def test_pr_number_null_when_field_absent(self, history_client):
        """Old benchmark files don't have pr_number — endpoint returns null
        so the JS can fall back to a commit-SHA link instead."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="dff43fb",
            results=[{"position": "K", "ridge_mae": 6.5, "elapsed_sec": 80.0}],
            # No pr= arg → no pr_number field in the file.
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        assert row["pr_number"] is None
        assert row["git_hash"] == "dff43fb"


class TestRobustness:
    def test_malformed_file_is_skipped(self, history_client):
        """A corrupt JSON shouldn't 500 the whole tab — silently skip it."""
        client, history_dir = history_client
        (history_dir / "bad.json").write_text("{ not valid json")
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="abc1234",
            pr=199,
            results=[{"position": "K", "ridge_mae": 6.5, "elapsed_sec": 80.0}],
        )
        rows = client.get("/api/benchmark_history").get_json()["rows"]
        assert len(rows) == 1
        assert rows[0]["git_hash"] == "abc1234"

    def test_non_json_files_ignored(self, history_client):
        client, history_dir = history_client
        (history_dir / "notes.txt").write_text("ignore me")
        (history_dir / "scratch.json.tmp").write_text("{}")
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="abc1234",
            pr=199,
            results=[{"position": "K", "ridge_mae": 6.5, "elapsed_sec": 80.0}],
        )
        rows = client.get("/api/benchmark_history").get_json()["rows"]
        assert len(rows) == 1

    def test_nan_mae_is_filtered_safely(self, history_client):
        """``float('nan')`` from a malformed pipeline result shouldn't end
        up in the JSON (browsers reject NaN). _safe_num drops it."""
        client, history_dir = history_client
        path = history_dir / "2026-05-01T10-00-00_abc.json"
        path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-05-01T10:00:00",
                    "git_hash": "abc1234",
                    "results": [{"position": "QB", "ridge_mae": 4.1, "nn_mae": None}],
                }
            )
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        assert row["ridge"] == [{"position": "QB", "mae": 4.1}]
        assert row["nn"] == []
