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

import src.serving.benchmark_history as benchmark_history

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
    monkeypatch.setattr(benchmark_history, "_benchmark_history_dir", lambda: str(history_dir))
    # Clear the mtime-keyed cache so a previous test's load doesn't shadow
    # this one (cache is module-global; mtime usually differs across tmp
    # dirs anyway but resetting is explicit and cheap).
    monkeypatch.setattr(benchmark_history, "_BENCHMARK_HISTORY_CACHE", None)
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
        monkeypatch.setattr(
            benchmark_history, "_benchmark_history_dir", lambda: "/nonexistent/path"
        )
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
        assert (
            benchmark_history._resolve_repo_slug(None)
            == benchmark_history._BENCHMARK_REPO_SLUG_DEFAULT
        )

    def test_empty_string_falls_back_to_default(self, app_module):
        assert (
            benchmark_history._resolve_repo_slug("")
            == benchmark_history._BENCHMARK_REPO_SLUG_DEFAULT
        )
        assert (
            benchmark_history._resolve_repo_slug("   ")
            == benchmark_history._BENCHMARK_REPO_SLUG_DEFAULT
        )

    def test_well_formed_slug_passes_through(self, app_module):
        assert benchmark_history._resolve_repo_slug("alice/foo-bar") == "alice/foo-bar"
        assert benchmark_history._resolve_repo_slug("Org_42/repo.name") == "Org_42/repo.name"
        # Whitespace is stripped before matching.
        assert benchmark_history._resolve_repo_slug("  alice/foo  ") == "alice/foo"

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
            resolved = benchmark_history._resolve_repo_slug(hostile)
        assert resolved == benchmark_history._BENCHMARK_REPO_SLUG_DEFAULT
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

        monkeypatch.setattr(benchmark_history.json, "load", _boom)
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


class TestPerTargetDetail:
    """Detailed mode (History tab) reads a flattened ``per_target`` map on each
    pill. It's attached only when the source result carries ``{model}_per_target``
    (a ``{target: {mae, r2}}`` block); otherwise the key is omitted so the
    at-a-glance pill stays exactly ``{position, mae}`` and the exact-equality
    assertions in TestRowShape keep passing untouched."""

    def test_per_target_attached_when_present(self, history_client):
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-20T10:00:00",
            sha="aaa1111",
            pr=242,
            results=[
                {
                    "position": "QB",
                    "ridge_mae": 6.5,
                    "ridge_per_target": {
                        "passing_yards": {"mae": 67.618, "r2": 0.352},
                        "passing_tds": {"mae": 0.86, "r2": 0.091},
                    },
                }
            ],
        )
        qb_ridge = client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0]
        # MAE-only, r2 dropped, rounded to 3dp.
        assert qb_ridge == {
            "position": "QB",
            "mae": 6.5,
            "per_target": {"passing_yards": 67.618, "passing_tds": 0.86},
        }

    def test_per_target_omitted_when_absent(self, history_client):
        """A run with the total ``ridge_mae`` but no ``ridge_per_target`` keeps the
        pill at exactly {position, mae} — the regression guard for the existing
        exact-equality tests."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-20T10:00:00",
            sha="aaa1111",
            pr=242,
            results=[{"position": "QB", "ridge_mae": 6.5}],
        )
        assert client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0] == {
            "position": "QB",
            "mae": 6.5,
        }

    def test_per_target_omitted_for_untrained_position(self, history_client):
        """Partial K-only run: the five untrained positions carry neither a
        numeric mae nor a per_target key across every model; the trained K cell
        carries detail only for the model whose source had per_target."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-20T10:00:00",
            sha="aaa1111",
            pr=242,
            results=[
                {
                    "position": "K",
                    "ridge_mae": 6.5,
                    "ridge_per_target": {"fg_yard_points": {"mae": 3.1, "r2": 0.2}},
                }
            ],
        )
        row = client.get("/api/benchmark_history").get_json()["rows"][0]
        for model in ("ridge", "nn", "attn_nn", "lgbm"):
            assert row[model][0] == {"position": "QB", "mae": None}
        k_ridge = row["ridge"][4]
        assert k_ridge["position"] == "K"
        assert k_ridge["per_target"] == {"fg_yard_points": 3.1}
        assert "per_target" not in row["nn"][4]

    def test_per_target_nan_or_none_mae_dropped(self, history_client):
        """Per-target entries whose mae is NaN/None are scrubbed by _safe_num and
        dropped from the flattened map (browsers reject NaN; the frontend would
        otherwise render ``undefined``)."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-20T10:00:00",
            sha="aaa1111",
            pr=242,
            results=[
                {
                    "position": "QB",
                    "ridge_mae": 6.5,
                    "ridge_per_target": {
                        "passing_yards": {"mae": 67.6, "r2": 0.3},
                        "passing_tds": {"mae": None, "r2": 0.1},
                        "rushing_tds": {"mae": float("nan"), "r2": 0.0},
                    },
                }
            ],
        )
        qb_ridge = client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0]
        assert qb_ridge["per_target"] == {"passing_yards": 67.6}

    def test_per_target_empty_dict_omitted(self, history_client):
        """An empty ``ridge_per_target: {}`` is treated as 'no detail' — the key
        is omitted so detailed mode falls back to the total for that cell."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-05-20T10:00:00",
            sha="aaa1111",
            pr=242,
            results=[{"position": "QB", "ridge_mae": 6.5, "ridge_per_target": {}}],
        )
        pill = client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0]
        assert "per_target" not in pill


class TestRmseMetric:
    """The History tab's MAE/RMSE toggle reads an additive ``rmse`` on each pill
    plus a parallel ``per_target_rmse`` map. Both surface only when the source run
    carries them — runs predating rmse omit them, so the at-a-glance pill stays
    exactly ``{position, mae}`` and the frontend renders the RMSE view as ``--``."""

    def test_rmse_attached_when_present(self, history_client):
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-06-01T10:00:00",
            sha="rms1111",
            pr=300,
            results=[{"position": "QB", "ridge_mae": 6.5, "ridge_rmse": 8.1}],
        )
        qb_ridge = client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0]
        assert qb_ridge == {"position": "QB", "mae": 6.5, "rmse": 8.1}

    def test_rmse_omitted_when_absent(self, history_client):
        """A run with mae but no rmse keeps the pill at exactly {position, mae} —
        the backward-compat guard for the exact-equality tests in TestRowShape."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-06-01T10:00:00",
            sha="rms2222",
            pr=300,
            results=[{"position": "QB", "ridge_mae": 6.5}],
        )
        qb_ridge = client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0]
        assert qb_ridge == {"position": "QB", "mae": 6.5}
        assert "rmse" not in qb_ridge

    def test_nan_rmse_is_omitted(self, history_client):
        """``float('nan')`` rmse is scrubbed by _safe_num and dropped (browsers
        reject NaN), same as the mae path."""
        client, history_dir = history_client
        path = history_dir / "2026-06-01T10-00-00_rms3.json"
        path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-06-01T10:00:00",
                    "git_hash": "rms3333",
                    "results": [{"position": "QB", "ridge_mae": 6.5, "ridge_rmse": float("nan")}],
                }
            )
        )
        qb_ridge = client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0]
        assert qb_ridge == {"position": "QB", "mae": 6.5}

    def test_per_target_rmse_attached_when_present(self, history_client):
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-06-01T10:00:00",
            sha="rms4444",
            pr=300,
            results=[
                {
                    "position": "QB",
                    "ridge_mae": 6.5,
                    "ridge_rmse": 8.1,
                    "ridge_per_target": {
                        "passing_yards": {"mae": 67.6, "rmse": 88.2, "r2": 0.35},
                        "passing_tds": {"mae": 0.86, "rmse": 1.1, "r2": 0.09},
                    },
                }
            ],
        )
        qb_ridge = client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0]
        # MAE map and the parallel RMSE map, both r2-stripped and rounded.
        assert qb_ridge["per_target"] == {"passing_yards": 67.6, "passing_tds": 0.86}
        assert qb_ridge["per_target_rmse"] == {"passing_yards": 88.2, "passing_tds": 1.1}

    def test_per_target_rmse_omitted_when_absent(self, history_client):
        """Old-format per_target ({mae, r2}, no rmse) yields the MAE map but no
        rmse map, so detailed mode under the RMSE toggle renders ``--``."""
        client, history_dir = history_client
        _write_run(
            history_dir,
            ts="2026-06-01T10:00:00",
            sha="rms5555",
            pr=300,
            results=[
                {
                    "position": "QB",
                    "ridge_mae": 6.5,
                    "ridge_per_target": {"passing_yards": {"mae": 67.6, "r2": 0.35}},
                }
            ],
        )
        qb_ridge = client.get("/api/benchmark_history").get_json()["rows"][0]["ridge"][0]
        assert qb_ridge["per_target"] == {"passing_yards": 67.6}
        assert "per_target_rmse" not in qb_ridge


class TestTargetMaps:
    """The endpoint serves static {target_key: label} and {target_key: unit}
    lookups so detailed mode can render 'Passing Yards … yds' without a second
    /api/position_details fetch. They're present regardless of how many runs
    exist (built from POSITION_INFO / TARGET_UNITS)."""

    def test_target_labels_and_units_in_response(self, history_client):
        client, _ = history_client
        body = client.get("/api/benchmark_history").get_json()
        assert body["target_labels"]["passing_yards"] == "Passing Yards"
        # K targets now carry a unit too (added so the breakdown drill-down AND
        # the History tab render kicker stats with a suffix instead of bare
        # numbers — see TODO.md Fixed archive). fg_yard_points is a point value;
        # fg_misses is a raw miss count.
        assert body["target_labels"]["fg_yard_points"] == "FG Yard Points"
        assert body["target_units"]["fg_yard_points"] == "pts"
        assert body["target_units"]["fg_misses"] == "misses"
        assert body["target_units"]["passing_yards"] == "yds"
        assert body["target_units"]["receptions"] == "rec"
