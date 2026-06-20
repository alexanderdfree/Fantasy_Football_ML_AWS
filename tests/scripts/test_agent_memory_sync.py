from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_agent_memory_sync_uses_separate_remote_prefixes(tmp_path: Path) -> None:
    call_log = tmp_path / "aws-calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$AWS_CALL_LOG"\n',
        encoding="utf-8",
    )
    fake_aws.chmod(0o755)

    env = {
        **os.environ,
        "AWS_CALL_LOG": str(call_log),
        "AWS_PROFILE": "dummy",
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "FF_CLAUDE_MEMORY_S3_PREFIX": "claude-memory/test-repo",
        "FF_CODEX_MEMORY_S3_PREFIX": "codex-memory/test-repo",
        "FF_GEMINI_MEMORY_S3_PREFIX": "gemini-memory/test-repo",
        "FF_MEMORY_S3_BUCKET": "test-bucket",
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    subprocess.run(
        ["bash", "scripts/agent-memory-sync.sh", "all", "status"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    joined = "\n".join(calls)

    assert "s3://test-bucket/claude-memory/test-repo/memory" in joined
    assert "s3://test-bucket/codex-memory/test-repo/memories" in joined
    # `all` now also covers Gemini/Antigravity (its own prefix + leaf).
    assert "s3://test-bucket/gemini-memory/test-repo/memory" in joined
    # Codex syncs exclude .git (SQLite/runtime state) and *.DS_Store (macOS cruft).
    assert "--exclude .git --exclude .git/* --exclude *.DS_Store" in joined

    claude_calls = [
        call for call in calls if "s3://test-bucket/claude-memory/test-repo/memory" in call
    ]
    assert claude_calls
    # Claude excludes MEMORY.md (a generated, machine-local index — never shared mutable state),
    # but NOT .git / .DS_Store (those are codex-scoped); topic files still sync verbatim.
    assert all("--exclude MEMORY.md" in call for call in claude_calls)
    assert all("--exclude .git" not in call for call in claude_calls)
    assert all("--exclude *.DS_Store" not in call for call in claude_calls)


def test_path_command_prints_local_memory_dirs(tmp_path: Path) -> None:
    # `path` resolves the local memory dir without S3/credentials (it short-circuits
    # before preflight); session-start.sh consumes it for the orphan-index check.
    env = {
        **os.environ,
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "HOME": str(tmp_path / "home"),
    }

    claude = subprocess.run(
        ["bash", "scripts/agent-memory-sync.sh", "claude", "path"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    # ~/.claude/projects/<slug>/memory under the overridden HOME.
    assert claude.stdout.strip().endswith("/memory")
    assert str(tmp_path / "home") in claude.stdout

    codex = subprocess.run(
        ["bash", "scripts/agent-memory-sync.sh", "codex", "path"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert codex.stdout.strip() == str(tmp_path / "codex-home" / "memories")

    gemini = subprocess.run(
        ["bash", "scripts/agent-memory-sync.sh", "gemini", "path"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    # Default: ~/.gemini/tmp/<project-slug>/memory under the overridden HOME.
    out = gemini.stdout.strip()
    assert out.startswith(str(tmp_path / "home" / ".gemini" / "tmp"))
    assert out.endswith("/memory")


def test_gemini_memory_dir_override(tmp_path: Path) -> None:
    # GEMINI_MEMORY_DIR is the authoritative override for Antigravity's local
    # memory path (its project slug is not derivable here).
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "GEMINI_MEMORY_DIR": str(tmp_path / "custom-gemini-mem"),
    }
    result = subprocess.run(
        ["bash", "scripts/agent-memory-sync.sh", "gemini", "path"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(tmp_path / "custom-gemini-mem")


def test_generate_rebuilds_claude_index_from_topic_files(tmp_path: Path) -> None:
    # `generate` rebuilds MEMORY.md from the topic files' index_line, needs no S3 (short-circuits
    # before preflight), and writes atomically. This is the SessionStart regeneration step.
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "codex-home"),
    }
    memdir = subprocess.run(
        ["bash", "scripts/agent-memory-sync.sh", "claude", "path"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    Path(memdir).mkdir(parents=True, exist_ok=True)
    (Path(memdir) / "alpha.md").write_text(
        "---\nname: A\nindex_line: |-\n  [A](alpha.md) — hook\n---\nbody\n", encoding="utf-8"
    )

    subprocess.run(
        ["bash", "scripts/agent-memory-sync.sh", "claude", "generate"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (Path(memdir) / "MEMORY.md").read_text(encoding="utf-8") == "- [A](alpha.md) — hook\n"


def test_generate_is_noop_for_codex(tmp_path: Path) -> None:
    # Codex has no MEMORY.md index; `codex generate` must be a clean no-op, not an error.
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "codex-home"),
    }
    result = subprocess.run(
        ["bash", "scripts/agent-memory-sync.sh", "codex", "generate"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
