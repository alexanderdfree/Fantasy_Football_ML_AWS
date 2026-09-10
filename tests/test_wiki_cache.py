"""A render must not acquire the version of a later document edit."""

import os

import pytest

from src.serving import app, wiki

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("during_render", [False, True])
def test_edit_does_not_pin_older_html(tmp_path, monkeypatch, during_render):
    path = tmp_path / "doc.md"
    path.write_text("# Original document")
    stamp = path.stat().st_mtime
    slug = "audit-render-edit"
    monkeypatch.setitem(wiki.WIKI_DOCS, slug, {"path": str(path)})
    monkeypatch.setattr(app, "_cache", {})
    renderer = wiki.markdown.markdown

    def edit():
        path.write_text("# Updated document")
        os.utime(path, (stamp + 2, stamp + 2))

    def render(text, **kwargs):
        if during_render and "Original" in text:
            edit()
        return renderer(text, **kwargs)

    monkeypatch.setattr(wiki.markdown, "markdown", render)
    assert "Original document" in wiki._render_wiki_doc(slug)
    if not during_render:
        edit()
    updated = wiki._render_wiki_doc(slug)
    assert "Updated document" in updated
    monkeypatch.setattr(wiki.markdown, "markdown", lambda *_a, **_k: pytest.fail("cache miss"))
    assert wiki._render_wiki_doc(slug) == updated
