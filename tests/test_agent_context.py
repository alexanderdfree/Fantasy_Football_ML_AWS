"""Keep startup guidance bounded and moved incident records discoverable."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "limit"),
    [("AGENTS.md", 8192), ("CODEX.md", 4096), ("CLAUDE.md", 4096), ("GEMINI.md", 4096)],
)
def test_startup_instruction_budget(name, limit):
    assert (ROOT / name).stat().st_size <= limit, (
        f"{name} exceeds {limit} UTF-8 bytes; move detail to a routed topic guide"
    )


@pytest.mark.parametrize("name", ["CODEX.md", "CLAUDE.md", "GEMINI.md"])
def test_provider_entrypoint_preserves_shared_import(name):
    assert (ROOT / name).read_text().startswith("@AGENTS.md\n")


@pytest.mark.parametrize(
    "name",
    ["AGENTS.md", "CODEX.md", "CLAUDE.md", "GEMINI.md", "agent-guides/README.md"],
)
def test_guidance_routes_resolve(name):
    source = ROOT / name
    for href in re.findall(r"\]\(([^\s)]+)\)", source.read_text()):
        target = urlsplit(href)
        if target.scheme or not target.path:
            continue
        assert (source.parent / unquote(target.path)).exists(), (name, href)


def test_incident_index_has_one_route_per_record():
    index = ROOT / "todo/fixed-archive.md"
    text = index.read_text()
    routes = re.findall(r"\]\((fixed-archive/[^\s)]+\.md)\)", text)
    files = {p.resolve() for p in (ROOT / "todo/fixed-archive").glob("*.md")}
    targets = [(index.parent / path).resolve() for path in routes]
    assert len(targets) == len(set(targets)), "Duplicate incident-index routes"
    assert set(targets) == files, "Incident records must all be indexed and present"

    headings = re.findall(r"^### .+$", text, re.MULTILINE)
    assert len(headings) == len(routes)
    for heading, target in zip(headings, targets, strict=True):
        assert heading in target.read_text().splitlines(), (heading, target)
