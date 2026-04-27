"""Flask API contract tests for /api/wiki/* routes.

Mirrors the marker convention in tests/test_app.py — these cross the Flask
boundary via test_client() and so are tagged `integration`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestWikiIndex:
    def test_returns_200_and_list(self, client):
        r = client.get("/api/wiki/index")
        assert r.status_code == 200
        assert r.is_json
        body = r.get_json()
        assert isinstance(body, list)
        assert len(body) >= 1

    def test_each_entry_has_required_fields(self, client):
        body = client.get("/api/wiki/index").get_json()
        for entry in body:
            assert set(entry) == {"slug", "name", "group"}
            assert entry["slug"]
            assert entry["name"]
            assert entry["group"]

    def test_includes_architecture_default(self, client):
        body = client.get("/api/wiki/index").get_json()
        slugs = {e["slug"] for e in body}
        assert "architecture" in slugs, "ADR-001 is the default landing page"


class TestWikiPage:
    def test_known_slug_returns_rendered_html(self, client):
        r = client.get("/api/wiki/architecture")
        assert r.status_code == 200
        body = r.get_json()
        assert set(body) == {"slug", "name", "group", "html"}
        assert body["slug"] == "architecture"
        # `toc` extension adds id attributes, so match the open tag with optional attrs.
        assert "<h1" in body["html"] and "</h1>" in body["html"]
        assert "ADR-001" in body["html"]

    def test_unknown_slug_returns_404(self, client):
        r = client.get("/api/wiki/this-doc-does-not-exist")
        assert r.status_code == 404
        assert r.is_json
        assert "error" in r.get_json()

    def test_intra_wiki_links_get_rewritten(self, client):
        # ADR-001 contains links like `[batch_design.md](batch_design.md)`
        # which the server-side rewriter should turn into `#wiki:batch-design`.
        body = client.get("/api/wiki/architecture").get_json()
        assert "#wiki:batch-design" in body["html"]


class TestWikiRegistryIntegrity:
    def test_every_registered_doc_file_exists(self):
        """Catches typos in WIKI_DOCS paths and accidental file removals.

        A Dockerfile that fails to copy a registered doc would still pass this
        test (it runs against the repo, not the deploy artifact), but at least
        a path typo or a deleted file won't slip through.
        """
        import os

        import src.serving.app as app_mod

        # WIKI_DOCS paths are repo-root-relative (e.g. "docs/ARCHITECTURE.md").
        # After the src/ migration the app module lives at src/serving/app.py,
        # so repo root is two parents up from its file location — must match
        # the resolution _render_wiki_doc() uses.
        app_dir = os.path.dirname(os.path.abspath(app_mod.__file__))
        repo_root = os.path.abspath(os.path.join(app_dir, "..", ".."))
        missing = [
            (slug, meta["path"])
            for slug, meta in app_mod.WIKI_DOCS.items()
            if not os.path.isfile(os.path.join(repo_root, meta["path"]))
        ]
        assert not missing, f"WIKI_DOCS references missing files: {missing}"
