"""In-app wiki: render committed markdown docs (README, ADRs, design notes)
as sanitized HTML pages served under ``/api/wiki``.

Slug is the only public identifier; raw paths are never accepted from the client.
Rendered HTML is mtime-cached in the shared serving cache — ``_render_wiki_doc``
lazily imports ``src.serving.app`` and uses ``app._cache`` / ``app._wiki_cache_lock``
(the lazy import breaks the app<->wiki cycle). Extracted from ``app.py`` during the
serving decomposition; ``app.py`` re-exports ``WIKI_DOCS`` / ``_render_wiki_doc`` /
``_wiki_rewrite_href`` / ``_WIKI_GITHUB_BLOB_BASE``.
"""

import glob
import os
import re

import bleach
import markdown

# ---------------------------------------------------------------------------
# Wiki — render committed markdown docs as in-app HTML pages.
# Slug is the only public identifier; raw paths are never accepted from the
# client, so a path-traversal slug like "../etc/passwd" simply misses the
# registry and 404s. Order in the dict drives sidebar order.
# ---------------------------------------------------------------------------
WIKI_DOCS: dict[str, dict] = {
    "readme": {"name": "Project Overview", "group": "Overview", "path": "README.md"},
    "setup": {"name": "Setup & Local Run", "group": "Overview", "path": "SETUP.md"},
    "todo": {"name": "TODO & Bug Archive", "group": "Overview", "path": "TODO.md"},
    "architecture": {
        "name": "ADR-001: System Architecture",
        "group": "Architecture",
        "path": "docs/ARCHITECTURE.md",
    },
    "architecture-history": {
        "name": "ADR Update History (archived)",
        "group": "Architecture",
        "path": "docs/architecture-history.md",
    },
    "ec2-design": {
        "name": "EC2 Training Design",
        "group": "Architecture",
        "path": "docs/ec2_design.md",
    },
    "expert-comparison": {
        "name": "Expert Projection Comparison",
        "group": "Architecture",
        "path": "docs/expert_comparison.md",
    },
    "batch-design": {
        "name": "AWS Batch Design (standby)",
        "group": "Design History",
        "path": "docs/batch_design.md",
    },
    "design-lstm-multihead": {
        "name": "LSTM Multi-Head Proposal",
        "group": "Design History",
        "path": "docs/archive/design_lstm_multihead.md",
    },
    "design-weather-and-odds": {
        "name": "Weather & Odds Features",
        "group": "Design History",
        "path": "docs/archive/design_weather_and_odds.md",
    },
    "design-xgboost-ensemble": {
        "name": "XGBoost Ensemble (rejected)",
        "group": "Design History",
        "path": "docs/archive/design_xgboost_ensemble.md",
    },
    "method-contracts": {
        "name": "Method Contracts",
        "group": "Specification",
        "path": "docs/method_contracts.md",
    },
    "infra-ec2": {
        "name": "EC2 Infrastructure",
        "group": "Infrastructure",
        "path": "infra/ec2/README.md",
    },
    "infra-aws": {
        "name": "AWS Serving Infrastructure",
        "group": "Infrastructure",
        "path": "infra/aws/README.md",
    },
}

# Auto-register per-decision ADR files (docs/adr/*.md) so a new ADR shows up in
# the wiki without a manual WIKI_DOCS entry. Slug = filename stem (e.g.
# "0017-platform-autodetection-…"); display name = the file's first markdown
# heading; grouped under "Architecture Decisions"; sorted for stable order.
_ADR_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _adr_path in sorted(glob.glob(os.path.join(_ADR_REPO_ROOT, "docs", "adr", "*.md"))):
    _stem = os.path.splitext(os.path.basename(_adr_path))[0]
    _adr_slug = _stem if _stem[:1].isdigit() else f"adr-{_stem.lower()}"
    try:
        with open(_adr_path, encoding="utf-8") as _f:
            _adr_name = _f.readline().lstrip("# ").strip() or _stem
    except OSError:
        _adr_name = _stem
    WIKI_DOCS.setdefault(
        _adr_slug,
        {
            "name": _adr_name,
            "group": "Architecture Decisions",
            "path": os.path.relpath(_adr_path, _ADR_REPO_ROOT),
        },
    )

# Reverse map: normalized repo-relative path -> slug. Used to rewrite intra-wiki
# markdown links (e.g. "[ARCH](docs/ARCHITECTURE.md)") into in-app `#wiki:slug`
# anchors so the JS can swap content without a full reload.
_WIKI_PATH_TO_SLUG = {os.path.normpath(d["path"]): slug for slug, d in WIKI_DOCS.items()}

_WIKI_HREF_RE = re.compile(r'href="([^"]+)"')

# Repo-relative links that resolve to a real file but aren't in WIKI_DOCS
# (e.g. `[shared/aggregate_targets.py](../shared/aggregate_targets.py)` from
# inside docs/ARCHITECTURE.md) get rewritten to a GitHub blob URL so clicking
# them in-app shows the source on github.com instead of a Flask 404.
_WIKI_GITHUB_BLOB_BASE = "https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/blob/main/"

# Schemes a sanitizer would normally block; we neutralize them at the rewriter
# level too so a malicious or careless `[link](javascript:...)` in committed
# markdown can't survive into the rendered HTML even if the bleach allowlist
# regresses. data:/vbscript: are included for the same defense-in-depth reason.
_WIKI_DANGEROUS_SCHEMES = ("javascript:", "data:", "vbscript:")

# bleach allowlist for rendered wiki HTML. Permits the markup that
# python-markdown produces (headings, lists, tables, fenced code, blockquotes,
# inline emphasis) plus `id` attributes — the toc extension adds `id` to every
# heading and same-doc TOC links rely on those anchors. `class` is allowed so
# python-markdown's `codehilite` extension (and any future syntax-highlight
# wiring) renders correctly. Disallowed tags like <script>, <iframe>, <style>,
# <object>, <embed>, <form>, <input>, and <meta> are stripped.
_WIKI_ALLOWED_TAGS = frozenset(
    {
        "a",
        "abbr",
        "b",
        "blockquote",
        "br",
        "cite",
        "code",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "em",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "i",
        "img",
        "kbd",
        "li",
        "mark",
        "ol",
        "p",
        "pre",
        "q",
        "s",
        "samp",
        "span",
        "strong",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "u",
        "ul",
        "var",
    }
)
_WIKI_ALLOWED_ATTRS = {
    "*": ["id", "class"],
    "a": ["href", "title", "target", "rel"],
    "img": ["src", "alt", "title", "width", "height"],
    "th": ["colspan", "rowspan", "align", "scope"],
    "td": ["colspan", "rowspan", "align"],
    "abbr": ["title"],
}
_WIKI_ALLOWED_PROTOCOLS = frozenset({"http", "https", "mailto"})


def _wiki_rewrite_href(href: str, doc_path: str) -> str:
    """Rewrite a markdown link inside a wiki doc to a safe in-app or GitHub URL.

    Outcomes:
    - Empty / pure anchor (`#section`) → unchanged (handled client-side).
    - Absolute URL (`http(s)://...`) or `mailto:` → unchanged.
    - Dangerous scheme (`javascript:`, `data:`, `vbscript:`) → neutralized to
      `#` so the link is inert (bleach also strips these as defense in depth).
    - Relative path that resolves to a registered wiki doc → `#wiki:slug[:anchor]`.
    - Relative path that doesn't resolve to a wiki doc but points at a real
      repo file → GitHub blob URL so the link still works (opens externally
      via the JS click handler).
    - Anything else relative → unchanged.
    """
    if not href or href.startswith("#") or "://" in href or href.startswith("mailto:"):
        return href
    if href.lower().startswith(_WIKI_DANGEROUS_SCHEMES):
        return "#"
    target, _, anchor = href.partition("#")
    if not target:
        return href
    # Reject absolute targets up front. ``os.path.join(doc_dir, target)``
    # silently DISCARDS doc_dir when ``target`` is absolute (e.g. "/etc/passwd"),
    # so the ``..``-prefix guard below never sees it and we'd emit a malformed
    # ``<blob_base>//etc/passwd`` external link. Author-controlled markdown is the
    # trust boundary, but a copy/pasted fragment from external docs can slip an
    # absolute path past defense-in-depth — make it inert.
    if os.path.isabs(target):
        return "#"
    doc_dir = os.path.dirname(doc_path)
    resolved = os.path.normpath(os.path.join(doc_dir, target))
    target_slug = _WIKI_PATH_TO_SLUG.get(resolved)
    if target_slug:
        return f"#wiki:{target_slug}:{anchor}" if anchor else f"#wiki:{target_slug}"
    # Path resolved into the parent directory (e.g. ../etc/passwd) — refuse it.
    if resolved.startswith(".."):
        return "#"
    blob_url = _WIKI_GITHUB_BLOB_BASE + resolved.replace(os.sep, "/")
    return f"{blob_url}#{anchor}" if anchor else blob_url


def _render_wiki_doc(slug: str) -> str:
    """Return cached, rendered HTML for the doc at WIKI_DOCS[slug].

    Uses ``_wiki_cache_lock`` (separate from ``_cache_lock``) so a wiki GET
    can never serialize behind a slow ``_ensure_metrics`` model-load. Cache
    entries still live in the shared ``_cache`` dict, but writes go through
    a dedicated lock — safe because no other code path mutates the
    ``("wiki", slug)`` keys, and Python dict insert/get for distinct keys
    is atomic at the bytecode level.
    """
    from src.serving import app as app_pkg

    cache_key = ("wiki", slug)
    meta = WIKI_DOCS[slug]
    # WIKI_DOCS paths are relative to repo root (e.g. "docs/ARCHITECTURE.md").
    # After the src/ migration this module lives at src/serving/wiki.py, so
    # repo root is two parents up. The Dockerfile copies the same .md files
    # into /app at the same relative paths, so this resolves correctly in
    # both local dev and the deployed container.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    abs_path = os.path.join(repo_root, meta["path"])
    # mtime-keyed cache: the wiki docs are rendered from on-disk markdown that a
    # docs-only deploy or an in-place edit can change under a long-lived worker.
    # Caching the HTML indefinitely (the original behaviour) served the stale
    # render until a container restart. Stat the source and store ``(mtime, html)``
    # so an edit (newer mtime) re-renders. ``os.stat`` is cheap (~µs); the render
    # + bleach pass is the expensive part we still skip on the common hit.
    try:
        current_mtime = os.stat(abs_path).st_mtime
    except OSError:
        # File vanished (shouldn't happen for a registered doc) — fall through to
        # ``open`` below, which raises a clear error the route turns into a 500.
        current_mtime = None
    with app_pkg._wiki_cache_lock:
        cached = app_pkg._cache.get(cache_key)
        if cached is not None and current_mtime is not None and cached[0] == current_mtime:
            return cached[1]
    with open(abs_path, encoding="utf-8") as f:
        text = f.read()
    html = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
        output_format="html",
    )
    doc_path = meta["path"]
    html = _WIKI_HREF_RE.sub(
        lambda m: f'href="{_wiki_rewrite_href(m.group(1), doc_path)}"',
        html,
    )
    # Defense in depth: even though committed markdown is author-controlled,
    # strip raw <script>/<iframe>/event-handler attrs/non-http(s)-mailto URLs
    # so an inadvertent bad link in a doc can't execute in a viewer's browser.
    html = bleach.clean(
        html,
        tags=_WIKI_ALLOWED_TAGS,
        attributes=_WIKI_ALLOWED_ATTRS,
        protocols=_WIKI_ALLOWED_PROTOCOLS,
        strip=True,
    )
    # Re-stat right before caching so the stored mtime matches the bytes we just
    # rendered (the file could have been rewritten between the read above and
    # here); a subsequent edit then still produces a newer mtime and re-renders.
    with app_pkg._wiki_cache_lock:
        try:
            render_mtime = os.stat(abs_path).st_mtime
        except OSError:
            render_mtime = current_mtime
        app_pkg._cache[cache_key] = (render_mtime, html)
    return html
