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


def test_over_cap_curated_lines_are_trimmed_not_dropped(tmp_path: Path) -> None:
    # Too many curated lines to fit: the generator must still emit an UNDER-cap index (the loader
    # would otherwise silently drop the alphabetical tail — project_*/user_* entries), shorten
    # only the longest hooks, keep every link intact, and say loudly that pruning is the fix.
    for i in range(200):
        _write(tmp_path, f"m{i:03d}.md", index_line=f"[M{i}](m{i:03d}.md) — " + "y" * 150)
    _write(tmp_path, "short.md", index_line="[S](short.md) — tiny")
    text, warnings = memory_index.generate_index(str(tmp_path))
    lines = text.splitlines()
    assert len(lines) == 201  # every file still indexed
    assert len(text.encode("utf-8")) < memory_index.CAP_BYTES
    assert "- [S](short.md) — tiny" in lines  # short line byte-identical
    trimmed = [ln for ln in lines if ln.endswith("…")]
    assert len(trimmed) == 200 and all(ln.startswith("- [M") and "](m" in ln for ln in trimmed)
    assert all(memory_index._LINK_RE.search(ln) for ln in lines)  # no link ever cut
    assert any("trimmed" in w and "prune" in w for w in warnings)  # loud, not silent
    assert not any(">= cap" in w for w in warnings)  # it fit, so not the hopeless case


@pytest.mark.parametrize("short_hook", ["x", "xy", "é"])
def test_short_hooks_do_not_make_a_feasible_index_exceed_the_cap(
    tmp_path: Path, short_hook: str
) -> None:
    # The minimum rendering must not enlarge a one- or two-byte hook into the
    # three-byte ellipsis and falsely reject an index that can fit by trimming
    # its only long hook. Include a multibyte character to pin byte accounting.
    hook_bytes = len(short_hook.encode("utf-8"))
    prefix_bytes = 129 - hook_bytes
    original = []
    for i in range(190):
        slug = f"topic_{i:03d}.md"
        title_bytes = prefix_bytes - len(f"- []({slug}) — ".encode())
        prefix = f"- [{'T' * title_bytes}]({slug}) — "
        assert len(prefix.encode("utf-8")) == prefix_bytes
        hook = short_hook if i < 189 else "z" * (100 + hook_bytes)
        original.append(prefix + hook)
        _write(tmp_path, slug, index_line=(prefix + hook)[2:])
    assert len(("\n".join(original) + "\n").encode("utf-8")) == 24800

    text, warnings = memory_index.generate_index(str(tmp_path))

    assert len(text.encode("utf-8")) <= memory_index.CAP_BYTES - memory_index.CAP_MARGIN_BYTES
    lines = text.splitlines()
    assert len(lines) == 190
    assert lines[:189] == original[:189]
    assert lines[-1].endswith("…")
    assert any("1 index line(s) trimmed" in warning for warning in warnings)
    assert not any(">= cap" in warning for warning in warnings)


def test_generate_is_idempotent_when_trimming(tmp_path: Path) -> None:
    for i in range(200):
        _write(tmp_path, f"m{i:03d}.md", index_line=f"[M{i}](m{i:03d}.md) — " + "y" * 150)
    first, _ = memory_index.generate_index(str(tmp_path))
    second, _ = memory_index.generate_index(str(tmp_path))
    assert first == second


def test_trim_never_cuts_inside_link_or_title_with_em_dash(tmp_path: Path) -> None:
    # The hook separator is searched AFTER the link, so an em-dash inside the title is not a
    # split point, and under a hard trim the whole "[title](slug.md) — " prefix survives.
    title_line = "[Deploy green — not live](deploy.md) — " + "z" * 300
    _write(tmp_path, "deploy.md", index_line=title_line)
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert text == "- " + title_line + "\n"  # verbatim when it fits
    assert warnings == []
    for i in range(300):  # now force a trim
        _write(tmp_path, f"m{i:03d}.md", index_line=f"[M{i}](m{i:03d}.md) — " + "y" * 150)
    text, _ = memory_index.generate_index(str(tmp_path))
    deploy = next(ln for ln in text.splitlines() if "(deploy.md)" in ln)
    assert deploy.startswith("- [Deploy green — not live](deploy.md) — z")
    assert deploy.endswith("…")


def test_curated_line_without_hook_separator_is_never_trimmed(tmp_path: Path) -> None:
    bare = "[Bare](bare.md) " + "w" * 400  # no " — " after the link -> all prefix
    _write(tmp_path, "bare.md", index_line=bare)
    for i in range(300):
        _write(tmp_path, f"m{i:03d}.md", index_line=f"[M{i}](m{i:03d}.md) — " + "y" * 150)
    text, _ = memory_index.generate_index(str(tmp_path))
    assert "- " + bare in text.splitlines()


def test_hopeless_over_cap_still_emits_everything_and_shouts(tmp_path: Path) -> None:
    # So many entries that even link-only lines exceed the cap: nothing more can be trimmed,
    # so emit the link-only index (loader will cut it) and raise the ">= cap" warning.
    for i in range(700):
        _write(
            tmp_path, f"m{i:03d}.md", index_line=f"[Memory number {i}](m{i:03d}.md) — " + "y" * 60
        )
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert len(text.splitlines()) == 700
    assert all(ln.endswith(" — …") for ln in text.splitlines())
    assert any(">= cap" in w for w in warnings)


def test_mixed_curated_and_fallback_over_cap_trims_both_and_warns(tmp_path: Path) -> None:
    # Fallback lines share the same uniform cap (never a bare "…" while room remains) and a
    # trim that lands on fallback lines is still reported — the old near-cap warning used to
    # cover that mid-migration state; silence there would hide a shortened index.
    for i in range(150):
        _write(tmp_path, f"c{i:03d}.md", index_line=f"[C{i}](c{i:03d}.md) — " + "y" * 150)
    for i in range(150):
        _write(tmp_path, f"f{i:03d}.md", description="x" * 150, name=f"F{i}")
    text, warnings = memory_index.generate_index(str(tmp_path))
    lines = text.splitlines()
    assert len(lines) == 300
    assert len(text.encode("utf-8")) < memory_index.CAP_BYTES
    assert not any(ln.endswith(" — …") for ln in lines)  # no bare-ellipsis hooks
    assert all(memory_index._LINK_RE.search(ln) for ln in lines)
    assert any("300 index line(s) trimmed" in w and "(150 curated)" in w for w in warnings)


def test_fallback_only_over_cap_still_warns_about_trimming(tmp_path: Path) -> None:
    for i in range(200):
        _write(tmp_path, f"m{i:03d}.md", description="y" * 400, name=f"M{i}")
    _, warnings = memory_index.generate_index(str(tmp_path))
    assert any("trimmed" in w and "(0 curated)" in w for w in warnings)


def test_prefix_outlier_among_normal_lines_keeps_link_and_fits(tmp_path: Path) -> None:
    # One line whose link prefix alone exceeds the computed cap: its hook collapses to "…",
    # the link survives, every other line is trimmed normally, and the total still fits.
    long_title = "T" * 400
    _write(tmp_path, "outlier.md", index_line=f"[{long_title}](outlier.md) — hook text")
    for i in range(300):
        _write(tmp_path, f"m{i:03d}.md", index_line=f"[M{i}](m{i:03d}.md) — " + "y" * 150)
    text, _ = memory_index.generate_index(str(tmp_path))
    outlier = next(ln for ln in text.splitlines() if "(outlier.md)" in ln)
    assert outlier == f"- [{long_title}](outlier.md) — …"
    assert len(text.encode("utf-8")) < memory_index.CAP_BYTES


def test_size_inside_margin_band_is_trimmed_below_target(tmp_path: Path) -> None:
    # A full index between the trim target and the hard cap must be trimmed to <= target,
    # not left in the margin band the loader is given no guarantee about.
    target = memory_index.CAP_BYTES - memory_index.CAP_MARGIN_BYTES
    hook = "y" * 200
    per_line = len(f"- [M000](m000.md) — {hook}\n".encode())
    n = (target // per_line) + 1  # just over target, well under the cap
    for i in range(n):
        _write(tmp_path, f"m{i:03d}.md", index_line=f"[M{i:03d}](m{i:03d}.md) — {hook}")
    full = n * per_line
    assert target < full < memory_index.CAP_BYTES, (target, full)
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert len(text.encode("utf-8")) <= target
    assert "…" in text and any("trimmed" in w for w in warnings)


def test_multibyte_hooks_never_split_a_character_at_the_cut(tmp_path: Path) -> None:
    for i in range(200):
        _write(tmp_path, f"m{i:03d}.md", index_line=f"[M{i}](m{i:03d}.md) — " + "日本語—é" * 40)
    text, _ = memory_index.generate_index(str(tmp_path))
    raw = text.encode("utf-8")
    assert len(raw) < memory_index.CAP_BYTES
    raw.decode("utf-8")  # strict: a split multibyte char would raise
    assert all(ln.endswith("…") for ln in text.splitlines())


def test_main_writes_bare_newlines_even_when_stdout_translates(tmp_path: Path) -> None:
    # The cap accounting counts one byte per newline; on native Windows text-mode stdout would
    # write "\r\n" and silently inflate the file past what was measured.
    import io
    import sys

    _write(tmp_path, "a.md", index_line="[A](a.md) — hook")
    buf = io.BytesIO()
    fake = io.TextIOWrapper(buf, encoding="utf-8", newline="\r\n", write_through=True)
    real = sys.stdout
    sys.stdout = fake
    try:
        assert memory_index.main(["memory_index.py", "generate", str(tmp_path)]) == 0
    finally:
        sys.stdout = real
    fake.flush()
    assert buf.getvalue() == b"- [A](a.md) \xe2\x80\x94 hook\n"


def test_backfill_accepts_a_genuine_ellipsis_ending_line(tmp_path: Path) -> None:
    # A hook that really ends in "…" (never trimmed) matches the file's own full rendering,
    # so it is usable — both for a curated file (idempotent) and a description-only file.
    _write(tmp_path, "keep.md", index_line="[Keep](keep.md) — trails off…")
    _write(tmp_path, "desc.md", description="also trails off…", name="Desc")
    text, _ = memory_index.generate_index(str(tmp_path))
    (tmp_path / "MEMORY.md").write_text(text, encoding="utf-8")
    changed, missing = memory_index.backfill(str(tmp_path))
    assert sorted(changed) == ["desc.md", "keep.md"]
    assert missing == []
    again, _ = memory_index.generate_index(str(tmp_path))
    assert again == text


def test_backfill_skips_generator_trimmed_lines(tmp_path: Path) -> None:
    # A "…"-terminated index line is a generator artifact, not a curated hook; writing it back
    # into index_line would permanently lose the full text. It must be reported, not applied.
    _write(tmp_path, "keep.md", index_line="[Keep](keep.md) — the full curated hook text")
    _write(tmp_path, "other.md", description="d", name="Other")
    (tmp_path / "MEMORY.md").write_text(
        "- [Keep](keep.md) — the full cur…\n- [Other](other.md) — fine\n", encoding="utf-8"
    )
    changed, missing = memory_index.backfill(str(tmp_path))
    assert changed == ["other.md"]
    assert missing == ["keep.md"]
    text, _ = memory_index.generate_index(str(tmp_path))
    assert "- [Keep](keep.md) — the full curated hook text" in text.splitlines()


def test_reads_nested_metadata_index_line(tmp_path: Path) -> None:
    # The harness moves index_line into a nested `metadata:` block as an inline-quoted string;
    # the generator must read it there, not fall back to the description.
    (tmp_path / "n.md").write_text(
        "---\nname: N\ndescription: verbose desc\nmetadata:\n"
        '  node_type: memory\n  index_line: "[N](n.md) — curated hook"\n  type: feedback\n'
        "---\nbody\n",
        encoding="utf-8",
    )
    text, warnings = memory_index.generate_index(str(tmp_path))
    assert text == "- [N](n.md) — curated hook\n"
    assert warnings == []  # used the nested index_line, did NOT fall back


def test_reads_nested_index_line_with_escaped_quotes(tmp_path: Path) -> None:
    (tmp_path / "q.md").write_text(
        '---\nname: Q\nmetadata:\n  index_line: "[Q](q.md) — a \\"quoted\\" hook"\n---\nbody\n',
        encoding="utf-8",
    )
    text, _ = memory_index.generate_index(str(tmp_path))
    assert text == '- [Q](q.md) — a "quoted" hook\n'


def test_backfill_replaces_nested_index_line_without_duplicating(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text(
        '---\nname: X\nmetadata:\n  index_line: "[X](x.md) — old"\n  type: feedback\n---\nbody\n',
        encoding="utf-8",
    )
    (tmp_path / "MEMORY.md").write_text("- [X](x.md) — new\n", encoding="utf-8")
    memory_index.backfill(str(tmp_path))
    raw = (tmp_path / "x.md").read_text(encoding="utf-8")
    assert raw.count("index_line:") == 1  # replaced, not duplicated
    assert "  type: feedback" in raw  # sibling metadata key preserved
    text, _ = memory_index.generate_index(str(tmp_path))
    assert text == "- [X](x.md) — new\n"


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    _write(tmp_path, "alpha.md", description="d", name="A")
    (tmp_path / "MEMORY.md").write_text("- [A](alpha.md) — hook\n", encoding="utf-8")
    memory_index.backfill(str(tmp_path))
    once = (tmp_path / "alpha.md").read_text(encoding="utf-8")
    memory_index.backfill(str(tmp_path))
    twice = (tmp_path / "alpha.md").read_text(encoding="utf-8")
    assert once == twice  # no duplicate index_line blocks
