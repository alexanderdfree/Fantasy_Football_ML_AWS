"""Exercise remote bootstrap isolation and the environment inherited by later shells."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
HOOK = Path(__file__).resolve().parents[2] / ".claude/hooks/session-start.sh"


def _bootstrap(tmp_path, *, fail=False):
    project = tmp_path / "project with spaces"
    project.mkdir(exist_ok=True)
    env_file = tmp_path / "session.env"
    calls = tmp_path / "uv.jsonl"
    fake_uv = tmp_path / "fake_uv.py"
    fake_uv.write_text(
        "import json, os, pathlib, sys\n"
        "with open(os.environ['BOOTSTRAP_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'venv':\n"
        "    root = pathlib.Path(sys.argv[-1]) / 'bin'\n"
        "    root.mkdir(parents=True, exist_ok=True)\n"
        "    for name in ['python', 'pytest', 'ruff']:\n"
        "        tool = root / name\n"
        "        tool.write_text('#!/bin/bash\\nexit 0\\n')\n"
        "        tool.chmod(0o755)\n"
        "elif sys.argv[1:3] == ['pip', 'install']:\n"
        "    sys.exit(int(os.environ.get('BOOTSTRAP_FAIL', '0')))\n"
    )
    env = {
        **os.environ,
        "CLAUDE_PROJECT_DIR": str(project),
        "CLAUDE_CODE_REMOTE": "true",
        "CLAUDE_ENV_FILE": str(env_file),
        "BOOTSTRAP_PYTHON": sys.executable,
        "BOOTSTRAP_UV": str(fake_uv),
        "BOOTSTRAP_CALLS": str(calls),
        "BOOTSTRAP_FAIL": "7" if fail else "0",
    }
    command = (
        'uv() { "$BOOTSTRAP_PYTHON" "$BOOTSTRAP_UV" "$@"; }; export -f uv; '
        'pip() { return 99; }; export -f pip; bash "$1"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "bootstrap-test", str(HOOK)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, project, env_file, [json.loads(line) for line in calls.read_text().splitlines()]


def test_cold_and_warm_remote_bootstrap_exports_the_target_environment(tmp_path):
    for _ in range(2):
        result, project, env_file, calls = _bootstrap(tmp_path)
        assert result.returncode == 0, result.stderr
        shell = subprocess.run(
            [
                "bash",
                "-c",
                '. "$1"; test "$VIRTUAL_ENV" = "$2/.venv" && '
                'test "$PYTHONPATH" = "$2" && test "$(command -v pytest)" = "$2/.venv/bin/pytest"',
                "check-exports",
                str(env_file),
                str(project),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert shell.returncode == 0, shell.stderr
    assert sum(call[0] == "venv" for call in calls) == 1
    installs = [call for call in calls if call[:2] == ["pip", "install"]]
    assert len(installs) == 2
    assert all(
        call[2:] == ["--python", str(project / ".venv/bin/python"), "-r", "requirements-dev.txt"]
        for call in installs
    )


def test_failed_install_does_not_publish_session_exports(tmp_path):
    result, _, env_file, _ = _bootstrap(tmp_path, fail=True)
    assert result.returncode == 7
    assert not env_file.exists()
