"""Unit tests for scripts/memory_index.py — the MEMORY.md generator + backfill.

The generator makes the auto-memory index a deterministic projection of the topic files (each
carrying its curated line in an ``index_line`` block scalar), so the index is no longer shared
mutable state and stops racing. These tests pin: round-trip fidelity (backfill -> generate
reproduces the index), idempotency, the YAML-hazard cases that broke the first design (brackets
in titles, embedded quotes, em-dashes), both frontmatter dialects, the description/body
fallbacks (the self-heal path), and cap enforcement.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "memory_index", PROJECT_ROOT / "scripts" / "memory_index.py"
)
memory_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(memory_index)


def _write(memdir: Path, filename: str, body: str = "body", **fm: str) -> None:
    """Write a topic file. fm keys become frontmatter; index_line is emitted as a block scalar."""
    lines = ["---"]
    for k, v in fm.items():
        if k == "index_line":
            lines += ["index_line: |-", f"  {v}"]
        else:
            lines.append(f"{k}: {v}")
    lines += ["---", "", body]
    (memdir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_generate_uses_index_line_verbatim(tmp_path: Path) -> None:
    _write(tmp_path, "alpha.md", index_line="[Alpha](alpha.md) — does a thing")
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert text == "- [Alpha](alpha.md) — does a thing\n"
    assert warnings == []


def test_generate_is_sorted_and_idempotent(tmp_path: Path) -> None:
    _write(tmp_path, "zeta.md", index_line="[Zeta](zeta.md) — z")
    _write(tmp_path, "alpha.md", index_line="[Alpha](alpha.md) — a")
    text1, _ = memory_index.generate_index(str(tmp_path))
    text2, _ = memory_index.generate_index(str(tmp_path))
    assert text1 == text2  # idempotent
    assert text1 == "- [Alpha](alpha.md) — a\n- [Zeta](zeta.md) — z\n"  # slug-sorted


def test_memory_md_is_not_an_entry(tmp_path: Path) -> None:
    _write(tmp_path, "alpha.md", index_line="[Alpha](alpha.md) — a")
    (tmp_path / "MEMORY.md").write_text("- [Alpha](alpha.md) — a\n", encoding="utf-8")
    text, _ = memory_index.generate_index(str(tmp_path))
    assert "MEMORY.md" not in text


def test_backfill_then_generate_roundtrip(tmp_path: Path) -> None:
    # Topic files exist; MEMORY.md holds the curated lines; backfill must let generate reproduce them.
    _write(tmp_path, "alpha.md", description="long verbose description", name="alpha-formal")
    _write(tmp_path, "beta.md", description="another", name="beta-formal")
    curated = "- [Alpha Label](alpha.md) — short hook\n- [Beta Label](beta.md) — other hook\n"
    (tmp_path / "MEMORY.md").write_text(curated, encoding="utf-8")

    changed, missing = memory_index.backfill(str(tmp_path))
    assert sorted(changed) == ["alpha.md", "beta.md"]
    assert missing == []
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert set(text.splitlines()) == set(curated.splitlines())  # same lines (reordered ok)
    assert warnings == []  # index_line present -> no fallback


def test_block_scalar_survives_brackets_quotes_emdash(tmp_path: Path) -> None:
    hazard = '[[docs-only] "tricky": title](slug.md) — a — b — c with "quotes"'
    _write(tmp_path, "slug.md", index_line=hazard)
    # backfill round-trips it through MEMORY.md too
    (tmp_path / "MEMORY.md").write_text(f"- {hazard}\n", encoding="utf-8")
    memory_index.backfill(str(tmp_path))
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert text == f"- {hazard}\n"
    assert warnings == []


def test_fallback_to_description_emits_warning(tmp_path: Path) -> None:
    _write(tmp_path, "gamma.md", description="fallback desc", name="Gamma")
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert text == "- [Gamma](gamma.md) — fallback desc\n"
    assert any("gamma.md" in w and "description" in w for w in warnings)


def test_single_fallback_not_truncated_when_room(tmp_path: Path) -> None:
    # One fallback file with plenty of cap headroom -> emitted in full (no needless truncation).
    _write(tmp_path, "big.md", description="x" * 500, name="Big")
    text, _ = memory_index.generate_index(str(tmp_path))
    assert "x" * 500 in text
    assert "…" not in text


def test_many_fallbacks_stay_under_cap(tmp_path: Path) -> None:
    # A bulk-fallback state (e.g. mid-migration, when a concurrent pull stripped index_line from
    # many files) must degrade to a COMPLETE, UNDER-cap index via dynamic per-line truncation —
    # never an over-cap one the loader would silently truncate.
    for i in range(200):
        _write(tmp_path, f"m{i:03d}.md", description="y" * 400, name=f"M{i}")
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert len(text.splitlines()) == 200  # every file still indexed
    assert len(text.encode("utf-8")) < memory_index.CAP_BYTES  # fits the cap by construction
    assert "…" in text  # lines were trimmed to fit
    assert not any(">= cap" in w for w in warnings)  # not the over-cap (prune) warning


def test_no_frontmatter_falls_back_to_body(tmp_path: Path) -> None:
    (tmp_path / "raw.md").write_text("# Raw heading\n\nsome content\n", encoding="utf-8")
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert text == "- [raw](raw.md) — Raw heading\n"
    assert any("raw.md" in w for w in warnings)


def test_both_frontmatter_dialects_read_name_and_description(tmp_path: Path) -> None:
    # nested metadata dialect
    (tmp_path / "nested.md").write_text(
        "---\nname: Nested\ndescription: nested desc\nmetadata:\n  type: feedback\n---\nbody\n",
        encoding="utf-8",
    )
    # flat dialect
    (tmp_path / "flat.md").write_text(
        "---\nname: Flat\ndescription: flat desc\ntype: project\n---\nbody\n", encoding="utf-8"
    )
    text, _ = memory_index.generate_index(str(tmp_path))
    assert "- [Nested](nested.md) — nested desc\n" in text
    assert "- [Flat](flat.md) — flat desc\n" in text


def test_backfill_adds_frontmatter_to_bare_file(tmp_path: Path) -> None:
    (tmp_path / "bare.md").write_text("just a body, no frontmatter\n", encoding="utf-8")
    (tmp_path / "MEMORY.md").write_text("- [Bare](bare.md) — the hook\n", encoding="utf-8")
    memory_index.backfill(str(tmp_path))
    raw = (tmp_path / "bare.md").read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    assert "just a body, no frontmatter" in raw  # body preserved
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert text == "- [Bare](bare.md) — the hook\n"
    assert warnings == []  # now has index_line


def test_cap_enforcement_warns_when_over(tmp_path: Path) -> None:
    for i in range(200):
        _write(tmp_path, f"m{i:03d}.md", index_line=f"[M{i}](m{i:03d}.md) — " + "y" * 150)
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert len(text.encode("utf-8")) >= memory_index.CAP_BYTES
    assert any("cap" in w.lower() for w in warnings)  # loud, not silent


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    _write(tmp_path, "alpha.md", description="d", name="A")
    (tmp_path / "MEMORY.md").write_text("- [A](alpha.md) — hook\n", encoding="utf-8")
    memory_index.backfill(str(tmp_path))
    once = (tmp_path / "alpha.md").read_text(encoding="utf-8")
    memory_index.backfill(str(tmp_path))
    twice = (tmp_path / "alpha.md").read_text(encoding="utf-8")
    assert once == twice  # no duplicate index_line blocks
