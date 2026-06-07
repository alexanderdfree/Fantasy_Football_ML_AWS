"""Focused unit tests for the audit-320 serving-bundle fixes (issue #346).

Covers the cleanly-testable findings that previously lacked dedicated
coverage:

- F114 — ``_wiki_rewrite_href`` must reject absolute-path targets (the
  ``os.path.join(doc_dir, target)`` join silently discards ``doc_dir`` when
  ``target`` is absolute, so the ``..``-prefix guard never sees it).
- F37  — K/DST ``fantasy_points_half_ppr`` / ``_standard`` mirror the
  format-invariant ``fantasy_points`` (NOT a fabricated value); the columns
  must be present and equal so non-PPR ``actual`` displays correctly.
- F35  — ``_render_wiki_doc`` re-renders when the source markdown's mtime
  advances (was: cached indefinitely).
- F112 — the all-positions orchestrator re-applies a position whose sentinel
  advanced after pre-warm (was: skipped, leaving a stale aggregate).
- F113 — ``_ensure_metrics`` must NOT re-hydrate from disk on a sentinel
  advance (a failed best-effort unlink could otherwise un-invalidate the
  refresh).
- F33  — all four model pred columns init to NaN (consistent failure
  semantics).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.serving.core as core
import src.serving.wiki as wiki

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# F114 — absolute-path rejection in _wiki_rewrite_href
# ---------------------------------------------------------------------------


class TestWikiRewriteHrefAbsolutePath:
    def test_absolute_unix_path_is_neutralized(self):
        from src.serving.app import _wiki_rewrite_href

        # Absolute target must NOT become a github blob URL (the join would
        # discard doc_dir and emit "<blob_base>//etc/passwd").
        out = _wiki_rewrite_href("/etc/passwd", "docs/ARCHITECTURE.md")
        assert out == "#"

    def test_absolute_path_with_anchor_is_neutralized(self):
        from src.serving.app import _wiki_rewrite_href

        out = _wiki_rewrite_href("/abs/anywhere.md#section", "docs/ARCHITECTURE.md")
        assert out == "#"

    def test_absolute_path_never_emits_double_slash_blob_url(self):
        from src.serving.app import _wiki_rewrite_href

        out = _wiki_rewrite_href("/var/log/x", "README.md")
        assert "github.com" not in out
        assert "//var" not in out

    def test_relative_repo_file_still_rewrites_to_blob_url(self):
        """Regression guard: the absolute-path rejection must not break the
        legitimate relative-path → GitHub blob URL rewrite."""
        from src.serving.app import _WIKI_GITHUB_BLOB_BASE, _wiki_rewrite_href

        out = _wiki_rewrite_href("../shared/aggregate_targets.py", "docs/ARCHITECTURE.md")
        # ../shared/... from docs/ resolves to shared/... (not a wiki doc) →
        # github blob URL.
        assert out.startswith(_WIKI_GITHUB_BLOB_BASE)
        assert "shared/aggregate_targets.py" in out

    def test_parent_traversal_still_neutralized(self):
        """The pre-existing ``..``-prefix guard must still fire for relative
        traversal that escapes the repo root."""
        from src.serving.app import _wiki_rewrite_href

        out = _wiki_rewrite_href("../../etc/passwd", "README.md")
        assert out == "#"

    def test_intra_wiki_link_still_rewrites_to_anchor(self):
        from src.serving.app import _wiki_rewrite_href

        # docs/ARCHITECTURE.md -> batch_design.md resolves to docs/batch_design.md
        # which IS a registered wiki doc.
        out = _wiki_rewrite_href("batch_design.md", "docs/ARCHITECTURE.md")
        assert out == "#wiki:batch-design"


# ---------------------------------------------------------------------------
# F37 — K/DST format-invariant scoring columns
# ---------------------------------------------------------------------------


def _stub_base_data_loaders(monkeypatch, app_mod, n_skill=4, n_k=3, n_dst=3):
    """Stub the QB/RB/WR/TE parquet reads + K/DST split loaders so
    ``_load_base_data_locked`` runs without on-disk artifacts. The skill-position
    frames carry the suffixed columns; the K/DST frames carry ONLY the unsuffixed
    ``fantasy_points`` (mirroring the real splits) so we exercise the F37 mirror
    branch.
    """

    def _skill_frame(_path):
        return pd.DataFrame(
            {
                "player_id": [f"S{i}" for i in range(n_skill)],
                "player_display_name": [f"Skill {i}" for i in range(n_skill)],
                "position": ["QB", "RB", "WR", "TE"][:n_skill],
                "recent_team": ["KC"] * n_skill,
                "season": [2025] * n_skill,
                "week": list(range(1, n_skill + 1)),
                "headshot_url": [""] * n_skill,
                "season_type": ["REG"] * n_skill,
                "fantasy_points": np.linspace(10, 20, n_skill),
                "fantasy_points_half_ppr": np.linspace(9, 18, n_skill),
                "fantasy_points_standard": np.linspace(8, 16, n_skill),
            }
        )

    monkeypatch.setattr(core.pd, "read_parquet", lambda path: _skill_frame(path))
    # _compute_scoring_formats is a no-op when the suffixed cols already exist.

    def _k_frame(n):
        return pd.DataFrame(
            {
                "player_id": [f"K{i}" for i in range(n)],
                "player_display_name": [f"Kicker {i}" for i in range(n)],
                "position": ["K"] * n,
                "recent_team": ["KC"] * n,
                "season": [2025] * n,
                "week": list(range(1, n + 1)),
                "headshot_url": [""] * n,
                # NOTE: only the unsuffixed PPR column — K splits don't carry
                # the suffixed ones.
                "fantasy_points": np.array([7.0, 11.0, 4.0])[:n],
            }
        )

    def _dst_frame(n):
        return pd.DataFrame(
            {
                "player_id": [f"D{i}" for i in range(n)],
                "player_display_name": [f"Defense {i}" for i in range(n)],
                "position": ["DST"] * n,
                "recent_team": ["SF"] * n,
                "season": [2025] * n,
                "week": list(range(1, n + 1)),
                "headshot_url": [""] * n,
                "fantasy_points": np.array([12.0, 3.0, 8.0])[:n],
            }
        )

    k_df = _k_frame(n_k)
    dst_df = _dst_frame(n_dst)
    monkeypatch.setattr(
        core, "_load_k_splits", lambda: (k_df, k_df, k_df, pd.DataFrame({"x": [1]}))
    )
    monkeypatch.setattr(core, "_load_dst_splits", lambda: (dst_df, dst_df, dst_df))


class TestKDstScoringColumns:
    def test_kdst_suffixed_columns_equal_ppr(self, monkeypatch, tmp_path):
        """K/DST scoring is format-invariant (no reception term), so the
        half_ppr/standard columns must exist and equal the PPR ``fantasy_points``
        — confirming the mirror is correct, not a fabricated bug."""
        import src.serving.app as app_mod

        monkeypatch.setattr(app_mod, "_cache", {})
        monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path / "sc"))
        _stub_base_data_loaders(monkeypatch, app_mod)

        core._load_base_data_locked()
        results = app_mod._cache["results"]

        for pos in ("K", "DST"):
            rows = results[results["position"] == pos]
            assert len(rows) > 0
            # All three formats present and identically equal for K/DST.
            assert (rows["fantasy_points_half_ppr"] == rows["fantasy_points"]).all()
            assert (rows["fantasy_points_standard"] == rows["fantasy_points"]).all()
            # And not silently NaN — these feed the "actual" display in non-PPR
            # scoring views.
            assert rows["fantasy_points_half_ppr"].notna().all()
            assert rows["fantasy_points_standard"].notna().all()

    def test_all_pred_columns_init_nan(self, monkeypatch, tmp_path):
        """F33: every model pred column inits to NaN (uniform failure sentinel)."""
        import src.serving.app as app_mod

        monkeypatch.setattr(app_mod, "_cache", {})
        monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path / "sc"))
        _stub_base_data_loaders(monkeypatch, app_mod)

        core._load_base_data_locked()
        results = app_mod._cache["results"]
        for prefix in ("ridge", "nn", "attn_nn", "lgbm"):
            assert results[f"{prefix}_pred"].isna().all()
            for fmt in ("ppr", "half_ppr", "standard"):
                assert results[f"{prefix}_pred_{fmt}"].isna().all()


# ---------------------------------------------------------------------------
# F35 — wiki render cache invalidates on source mtime change
# ---------------------------------------------------------------------------


class TestWikiRenderMtimeInvalidation:
    def test_rerenders_when_source_mtime_advances(self, monkeypatch, tmp_path):
        import src.serving.app as app_mod

        doc = tmp_path / "doc.md"
        doc.write_text("# First\n\nhello\n", encoding="utf-8")

        monkeypatch.setattr(app_mod, "_cache", {})
        monkeypatch.setattr(
            wiki,
            "WIKI_DOCS",
            {"d": {"name": "Doc", "group": "G", "path": "doc.md"}},
        )
        # Point repo-root resolution at tmp_path by faking __file__ two levels up.
        fake_app_file = tmp_path / "src" / "serving" / "app.py"
        fake_app_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(wiki, "__file__", str(fake_app_file))

        html1 = app_mod._render_wiki_doc("d")
        assert "First" in html1

        # Edit the doc + bump mtime well past the cached value.
        doc.write_text("# Second\n\nworld\n", encoding="utf-8")
        import os

        st = os.stat(doc)
        os.utime(doc, (st.st_mtime + 100, st.st_mtime + 100))

        html2 = app_mod._render_wiki_doc("d")
        assert "Second" in html2
        assert "First" not in html2

    def test_cache_hit_when_mtime_unchanged(self, monkeypatch, tmp_path):
        """No re-render when the file is untouched — the expensive markdown +
        bleach pass is skipped on the common hit."""
        import src.serving.app as app_mod

        doc = tmp_path / "doc.md"
        doc.write_text("# Stable\n", encoding="utf-8")
        monkeypatch.setattr(app_mod, "_cache", {})
        monkeypatch.setattr(
            wiki, "WIKI_DOCS", {"d": {"name": "Doc", "group": "G", "path": "doc.md"}}
        )
        fake_app_file = tmp_path / "src" / "serving" / "app.py"
        fake_app_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(wiki, "__file__", str(fake_app_file))

        app_mod._render_wiki_doc("d")
        calls = {"n": 0}
        real_markdown = wiki.markdown.markdown

        def _counting_markdown(*a, **k):
            calls["n"] += 1
            return real_markdown(*a, **k)

        monkeypatch.setattr(wiki.markdown, "markdown", _counting_markdown)
        app_mod._render_wiki_doc("d")  # mtime unchanged → cache hit
        assert calls["n"] == 0, "markdown re-rendered despite unchanged mtime"


# ---------------------------------------------------------------------------
# F112 — all-positions orchestrator re-applies post-prewarm sentinel advances
# ---------------------------------------------------------------------------


def _wire_all_positions(monkeypatch, app_mod, mtimes):
    """Set up a minimal cache + stubbed _apply/_sentinel for
    _ensure_all_positions_loaded, returning a per-position call counter."""
    counts: dict[str, int] = {}

    def _fake_apply(train, val, test, pos, results):
        counts[pos] = counts.get(pos, 0) + 1

    monkeypatch.setattr(core, "_apply_position_models", _fake_apply)
    monkeypatch.setattr(core, "refresh_sentinel_mtime", lambda pos: mtimes.get(pos, 0.0))
    monkeypatch.setattr(core, "_ensure_base_data", lambda: None)
    monkeypatch.setattr(core, "_invalidate_metrics_cache", lambda **k: None)
    app_mod._cache.clear()
    app_mod._cache["splits"] = {p: (None, None, None) for p in app_mod._ALL_POSITIONS}
    app_mod._cache["results"] = pd.DataFrame({"position": list(app_mod._ALL_POSITIONS)})
    app_mod._cache["positions_loaded"] = set()
    return counts


class TestAllPositionsSentinelRecheck:
    def test_post_prewarm_sentinel_advance_triggers_reapply(self, monkeypatch):
        import src.serving.app as app_mod

        mtimes = {p: 10.0 for p in app_mod._ALL_POSITIONS}
        counts = _wire_all_positions(monkeypatch, app_mod, mtimes)

        # First pass: every position loads once.
        core._ensure_all_positions_loaded()
        assert counts == {p: 1 for p in app_mod._ALL_POSITIONS}

        # No advance → second pass is a no-op (pending empty).
        core._ensure_all_positions_loaded()
        assert counts == {p: 1 for p in app_mod._ALL_POSITIONS}

        # WR's sentinel advances post-prewarm. The orchestrator must evict + re-
        # apply ONLY WR (the bug: pending stayed empty so the stale aggregate
        # survived).
        mtimes["WR"] = 99.0
        core._ensure_all_positions_loaded()
        assert counts["WR"] == 2
        assert all(counts[p] == 1 for p in app_mod._ALL_POSITIONS if p != "WR")

    def test_k_sentinel_advance_refreshes_k_data(self, monkeypatch):
        import src.serving.app as app_mod

        mtimes = {p: 10.0 for p in app_mod._ALL_POSITIONS}
        _wire_all_positions(monkeypatch, app_mod, mtimes)
        refreshed = {"n": 0}
        monkeypatch.setattr(
            core,
            "_refresh_k_data_locked",
            lambda: refreshed.__setitem__("n", refreshed["n"] + 1),
        )

        core._ensure_all_positions_loaded()
        assert refreshed["n"] == 0  # first load doesn't trigger the refresh path

        mtimes["K"] = 50.0
        core._ensure_all_positions_loaded()
        assert refreshed["n"] == 1, "K sentinel advance must refresh k_kicks_df/splits"


# ---------------------------------------------------------------------------
# F113 — _ensure_metrics must not re-hydrate on a sentinel advance
# ---------------------------------------------------------------------------


class TestEnsureMetricsNoHydrateOnSentinelAdvance:
    def test_sentinel_advance_skips_hydrate(self, monkeypatch):
        import src.serving.app as app_mod

        app_mod._cache.clear()
        # Pretend an aggregate is cached and a position's sentinel advanced.
        app_mod._cache["metrics_by_format"] = {"ppr": "STALE"}
        app_mod._cache["positions_loaded"] = {"QB"}
        app_mod._cache["positions_mtime"] = {"QB": 1.0}
        monkeypatch.setattr(core, "refresh_sentinel_mtime", lambda pos: 2.0)

        hydrate_calls = {"n": 0}

        def _fake_hydrate():
            hydrate_calls["n"] += 1
            return True  # simulate the stale cache surviving a failed unlink

        monkeypatch.setattr(core, "_try_hydrate_from_disk", _fake_hydrate)
        monkeypatch.setattr(core, "_invalidate_metrics_cache", lambda **k: None)

        applied = {"n": 0}
        monkeypatch.setattr(
            core, "_ensure_all_positions_loaded", lambda: applied.__setitem__("n", 1)
        )
        monkeypatch.setattr(core, "_compute_metrics_locked", lambda: None)

        core._ensure_metrics()

        # The sentinel-advance path must NOT consult the disk cache (which could
        # re-load the stale aggregate); it recomputes from per-position preds.
        assert hydrate_calls["n"] == 0
        assert applied["n"] == 1

    def test_cold_start_still_hydrates(self, monkeypatch):
        """Regression guard: a genuine cold start (no in-memory aggregate) must
        STILL try the disk cache — that's the fast-boot path."""
        import src.serving.app as app_mod

        app_mod._cache.clear()
        monkeypatch.setattr(core, "refresh_sentinel_mtime", lambda pos: 0.0)

        hydrate_calls = {"n": 0}
        monkeypatch.setattr(
            core,
            "_try_hydrate_from_disk",
            lambda: hydrate_calls.__setitem__("n", hydrate_calls["n"] + 1) or True,
        )
        monkeypatch.setattr(core, "_ensure_all_positions_loaded", lambda: None)
        monkeypatch.setattr(core, "_compute_metrics_locked", lambda: None)

        core._ensure_metrics()
        assert hydrate_calls["n"] == 1
