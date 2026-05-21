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
    # Clear the mtime-keyed cache so a previous test's load doesn't shadow
    # this one (cache is module-global; mtime usually differs across tmp
    # dirs anyway but resetting is explicit and cheap).
    monkeypatch.setattr(app_module, "_BENCHMARK_HISTORY_CACHE", None)
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
        """Partial run (K-only): six pills per model in canonical order; the
        untrained positions carry ``mae=None`` so the frontend can render
        them as ``--``."""
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
        assert row["ridge"] == [
            {"position": "QB", "mae": None},
            {"position": "RB", "mae": None},
            {"position": "WR", "mae": None},
            {"position": "TE", "mae": None},
            {"position": "K", "mae": 6.668},
            {"position": "DST", "mae": None},
        ]
        assert row["nn"] == [
            {"position": "QB", "mae": None},
            {"position": "RB", "mae": None},
            {"position": "WR", "mae": None},
            {"position": "TE", "mae": None},
            {"position": "K", "mae": 7.133},
            {"position": "DST", "mae": None},
        ]
        assert row["attn_nn"] == [
            {"position": "QB", "mae": None},
            {"position": "RB", "mae": None},
            {"position": "WR", "mae": None},
            {"position": "TE", "mae": None},
            {"position": "K", "mae": 7.12},
            {"position": "DST", "mae": None},
        ]
        assert row["lgbm"] == [
            {"position": "QB", "mae": None},
            {"position": "RB", "mae": None},
            {"position": "WR", "mae": None},
            {"position": "TE", "mae": None},
            {"position": "K", "mae": 7.188},
            {"position": "DST", "mae": None},
        ]
        assert row["total_elapsed_sec"] == 84.0
        assert row["training_skipped"] is False

    def test_multi_position_run_stacks_pills(self, history_client):
        """Partial QB+RB run: trained positions carry their numeric MAEs,
        the other four positions are emitted as ``mae=None`` placeholders."""
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
            {"position": "WR", "mae": None},
            {"position": "TE", "mae": None},
            {"position": "K", "mae": None},
            {"position": "DST", "mae": None},
        ]
        assert row["positions"] == ["QB", "RB"]
        assert row["total_elapsed_sec"] == 120.0
        assert row["training_skipped"] is False

    def test_missing_model_mae_rendered_as_null(self, history_client):
        """K and DST runs historically had no attn_nn/lgbm MAE; modern partial
        runs only train one position. Either way, every (position, model) pair
        absent from ``results`` surfaces as ``mae=None`` so the frontend can
        render ``--`` uniformly."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="abc1234",
            pr=199,
            results=[{"position": "K", "ridge_mae": 6.5, "elapsed_sec": 80.0}],
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        # Ridge: K has data, the other five positions are nulled.
        assert row["ridge"] == [
            {"position": "QB", "mae": None},
            {"position": "RB", "mae": None},
            {"position": "WR", "mae": None},
            {"position": "TE", "mae": None},
            {"position": "K", "mae": 6.5},
            {"position": "DST", "mae": None},
        ]
        # NN / Attn NN / LGBM had no MAE at all for K (old-format file) — every
        # pill across all six positions is None.
        for model in ("nn", "attn_nn", "lgbm"):
            assert row[model] == [
                {"position": "QB", "mae": None},
                {"position": "RB", "mae": None},
                {"position": "WR", "mae": None},
                {"position": "TE", "mae": None},
                {"position": "K", "mae": None},
                {"position": "DST", "mae": None},
            ], f"unexpected {model} pills: {row[model]!r}"


class TestSkippedTraining:
    """Sentinel JSON files written by ``.github/workflows/skip-sentinel.yml``
    for [docs-only] or no-model-relevant-path commits. The shape is empty
    ``results`` plus a top-level ``training_skipped: true`` flag, and the row
    should surface as six all-null pills across every model."""

    def test_skipped_training_entry_renders_all_null(self, history_client):
        client, history_dir = history_client
        ts = "2026-05-21T18:00:00"
        sha = "abcdef0"
        path = history_dir / f"{ts.replace(':', '-')}_{sha}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": f"{ts}_{sha}",
                    "timestamp": ts,
                    "git_hash": sha,
                    "note": "Training skipped — docs-only commit",
                    "results": [],
                    "positions": [],
                    "training_skipped": True,
                    "pr_number": 999,
                }
            )
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        assert row["training_skipped"] is True
        assert row["positions"] == []
        for model in ("ridge", "nn", "attn_nn", "lgbm"):
            assert row[model] == [
                {"position": "QB", "mae": None},
                {"position": "RB", "mae": None},
                {"position": "WR", "mae": None},
                {"position": "TE", "mae": None},
                {"position": "K", "mae": None},
                {"position": "DST", "mae": None},
            ], f"unexpected {model} pills: {row[model]!r}"
        assert row["total_elapsed_sec"] == 0.0

    def test_empty_results_without_flag_still_marked_skipped(self, history_client):
        """Older sentinel files (or any file with empty results) should be
        flagged as skipped even without the explicit ``training_skipped`` key,
        so the UI can rely on a single field rather than two."""
        client, history_dir = history_client
        ts = "2026-05-21T18:30:00"
        sha = "deadbee"
        path = history_dir / f"{ts.replace(':', '-')}_{sha}.json"
        path.write_text(
            json.dumps(
                {
                    "timestamp": ts,
                    "git_hash": sha,
                    "results": [],
                }
            )
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        assert row["training_skipped"] is True


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
        up in the JSON (browsers reject NaN). _safe_num converts to None,
        which the row payload surfaces as a ``mae=None`` pill so the
        frontend renders ``--``."""
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
        # QB ridge has a numeric MAE; QB nn is null (input had None); the
        # other five positions are null across every model.
        assert row["ridge"][0] == {"position": "QB", "mae": 4.1}
        assert row["nn"][0] == {"position": "QB", "mae": None}
        # Spot-check one of the untrained positions to confirm padding.
        assert row["ridge"][5] == {"position": "DST", "mae": None}
        assert row["nn"][5] == {"position": "DST", "mae": None}


class TestRepoSlugResolution:
    """``_resolve_repo_slug`` guards the env-overridable repo slug. The slug
    is interpolated into an ``href`` in the frontend, so we reject anything
    that doesn't look like ``owner/repo`` and fall back to the default.

    Hostile env vars are above the user-input threat model, but a typo'd
    BENCHMARK_REPO_SLUG would otherwise silently break every History row's
    link without a log line; the warning surfaces that to the operator."""

    def test_unset_falls_back_to_default(self, app_module):
        assert app_module._resolve_repo_slug(None) == app_module._BENCHMARK_REPO_SLUG_DEFAULT

    def test_empty_string_falls_back_to_default(self, app_module):
        assert app_module._resolve_repo_slug("") == app_module._BENCHMARK_REPO_SLUG_DEFAULT
        assert app_module._resolve_repo_slug("   ") == app_module._BENCHMARK_REPO_SLUG_DEFAULT

    def test_well_formed_slug_passes_through(self, app_module):
        assert app_module._resolve_repo_slug("alice/foo-bar") == "alice/foo-bar"
        assert app_module._resolve_repo_slug("Org_42/repo.name") == "Org_42/repo.name"
        # Whitespace is stripped before matching.
        assert app_module._resolve_repo_slug("  alice/foo  ") == "alice/foo"

    @pytest.mark.parametrize(
        "hostile",
        [
            'evil.com" onclick="x',  # HTML-attribute breakout attempt
            "owner/repo with space",  # whitespace mid-value
            "owner/repo>",  # closing-bracket
            "owner",  # missing /repo
            "owner/repo/extra",  # too many segments
            "/owner/repo",  # leading slash
            "owner//repo",  # double slash
            "javascript:alert(1)",  # scheme injection attempt
        ],
    )
    def test_malformed_slug_falls_back_with_warning(self, app_module, hostile, caplog):
        with caplog.at_level("WARNING", logger="src.serving.app"):
            resolved = app_module._resolve_repo_slug(hostile)
        assert resolved == app_module._BENCHMARK_REPO_SLUG_DEFAULT
        assert any("BENCHMARK_REPO_SLUG" in rec.message for rec in caplog.records)


class TestCaching:
    def test_repeat_calls_reuse_cache_until_dir_changes(self, history_client, monkeypatch):
        """Two consecutive GETs over an unchanged dir must hit the cache —
        no re-parse. We assert this by patching json.load to crash on the
        second call; if the cache works, that load never happens."""
        import src.serving.app as app_mod

        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="abc1234",
            pr=199,
            results=[{"position": "K", "ridge_mae": 6.5, "elapsed_sec": 80.0}],
        )
        # Warm the cache.
        first = client.get("/api/benchmark_history").get_json()
        assert len(first["rows"]) == 1

        # Second call: poison json.load so any re-parse would explode.
        def _boom(*args, **kwargs):
            raise AssertionError("cache miss — json.load should not be called")

        monkeypatch.setattr(app_mod.json, "load", _boom)
        second = client.get("/api/benchmark_history").get_json()
        assert second == first

    def test_cache_invalidates_when_new_file_lands(self, history_client):
        """A new file in the dir bumps the directory mtime, which is the
        cache key — the next request reparses and returns the new row."""
        import os
        import time as _time

        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-01T10:00:00",
            sha="aaa1111",
            pr=180,
            results=[{"position": "QB", "ridge_mae": 4.1, "elapsed_sec": 50.0}],
        )
        first = client.get("/api/benchmark_history").get_json()
        assert {r["git_hash"] for r in first["rows"]} == {"aaa1111"}

        # Ensure mtime granularity (HFS+/APFS can have second-level
        # resolution); bump it explicitly so the cache key changes.
        _time.sleep(0.01)
        os.utime(history_dir, None)
        _write_run(
            history_dir,
            ts="2026-05-19T22:47:20",
            sha="bbb2222",
            pr=199,
            results=[{"position": "K", "ridge_mae": 6.5, "elapsed_sec": 80.0}],
        )
        second = client.get("/api/benchmark_history").get_json()
        assert {r["git_hash"] for r in second["rows"]} == {"aaa1111", "bbb2222"}
