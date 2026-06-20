"""Unit tests for the MEMORY.md orphan detector in .claude/hooks/lib.sh.

`claude_list_unindexed_memories <dir>` is the warn-only safeguard against the
concurrent-S3-sync failure mode: MEMORY.md is rewritten wholesale by every
session, so an overlapping (often cross-platform) session can overwrite it and
drop another session's freshly added index line, leaving the topic file present
but unindexed. The detector lists exactly those orphans so session-start.sh can
warn. These tests pin its matching (exact link target, skips the index itself,
no-ops when the index is absent).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIB = PROJECT_ROOT / ".claude" / "hooks" / "lib.sh"


def _list_unindexed(memdir: Path) -> list[str]:
    res = subprocess.run(
        ["bash", "-c", '. "$1"; claude_list_unindexed_memories "$2"', "_", str(LIB), str(memdir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.split()


def _index(memdir: Path, *slugs: str) -> None:
    (memdir / "MEMORY.md").write_text(
        "".join(f"- [{s}]({s}.md) — hook\n" for s in slugs), encoding="utf-8"
    )


def test_lists_orphan_and_skips_indexed(tmp_path: Path) -> None:
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "alpha.md").write_text("a", encoding="utf-8")
    (mem / "beta.md").write_text("b", encoding="utf-8")
    _index(mem, "alpha")  # alpha indexed, beta is an orphan

    out = _list_unindexed(mem)
    assert out == ["beta.md"]
    assert "alpha.md" not in out
    assert "MEMORY.md" not in out  # the index itself is never reported


def test_no_index_file_reports_nothing(tmp_path: Path) -> None:
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "alpha.md").write_text("a", encoding="utf-8")
    # No MEMORY.md yet -> read-only no-op, not "everything is an orphan".
    assert _list_unindexed(mem) == []


def test_complete_index_reports_nothing(tmp_path: Path) -> None:
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "alpha.md").write_text("a", encoding="utf-8")
    (mem / "beta.md").write_text("b", encoding="utf-8")
    _index(mem, "alpha", "beta")
    assert _list_unindexed(mem) == []


def test_substring_slug_is_not_a_false_negative(tmp_path: Path) -> None:
    mem = tmp_path / "memory"
    mem.mkdir()
    # foobar.md is indexed; foo.md is NOT. A substring match would wrongly treat
    # foo.md as indexed -> the exact "](foo.md)" target must distinguish them.
    (mem / "foo.md").write_text("f", encoding="utf-8")
    (mem / "foobar.md").write_text("fb", encoding="utf-8")
    _index(mem, "foobar")
    assert _list_unindexed(mem) == ["foo.md"]


def test_missing_dir_arg_is_safe(tmp_path: Path) -> None:
    # Empty/missing dir argument must not error (fail-open in the hook).
    res = subprocess.run(
        ["bash", "-c", '. "$1"; claude_list_unindexed_memories ""', "_", str(LIB)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout == ""
