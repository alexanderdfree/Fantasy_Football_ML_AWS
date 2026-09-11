"""HTTP and wiki regressions retained across the serving ownership migration."""

import os

import pytest
from flask.json.provider import DefaultJSONProvider
from werkzeug.exceptions import ServiceUnavailable

from src.serving.app import create_app

pytestmark = pytest.mark.unit


def test_http_errors_keep_headers_and_use_the_owning_application():
    class TaggedJSON(DefaultJSONProvider):
        def dumps(self, obj, **kwargs):
            return super().dumps({**obj, "owner": "independent"}, **kwargs)

    application = create_app(config={"TESTING": True})
    application.json = TaggedJSON(application)

    @application.get("/api/review-retry")
    def retry():
        raise ServiceUnavailable(retry_after=17)

    client = application.test_client()
    response = client.get("/api/review-retry")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "17"
    assert response.json["owner"] == "independent"
    method = client.post("/api/review-retry")
    assert method.status_code == 405
    assert "GET" in method.headers["Allow"]
    missing = client.get("/review-missing")
    assert missing.status_code == 404
    assert not missing.is_json


def test_edit_during_wiki_render_is_visible_on_the_next_request(monkeypatch, tmp_path):
    from src.serving import wiki

    application = create_app(config={"TESTING": True})
    document = tmp_path / "doc.md"
    document.write_text("# Original\n", encoding="utf-8")
    initial = document.stat().st_mtime
    module = tmp_path / "src" / "serving" / "wiki.py"
    module.parent.mkdir(parents=True)
    monkeypatch.setattr(wiki, "__file__", str(module))
    monkeypatch.setattr(wiki, "WIKI_DOCS", {"review": {"name": "Review", "path": "doc.md"}})
    render = wiki.markdown.markdown
    changed = False

    def edit_while_rendering(text, **kwargs):
        nonlocal changed
        if not changed:
            document.write_text("# Updated\n", encoding="utf-8")
            os.utime(document, (initial + 100, initial + 100))
            changed = True
        return render(text, **kwargs)

    monkeypatch.setattr(wiki.markdown, "markdown", edit_while_rendering)
    with application.app_context():
        assert "Original" in wiki._render_wiki_doc("review")
        assert "Updated" in wiki._render_wiki_doc("review")
