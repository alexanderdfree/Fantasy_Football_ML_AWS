"""Configured hook commands must execute from valid checkout paths with spaces."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("bash") is None, reason="bash required"),
]


def _commands():
    for provider in ("claude", "gemini"):
        settings = json.loads((ROOT / f".{provider}/settings.json").read_text())
        commands = {
            hook["command"]
            for groups in settings["hooks"].values()
            for group in groups
            for hook in group["hooks"]
        }
        for command in sorted(commands):
            yield pytest.param(
                provider, command, id=f"{provider}-{Path(shlex.split(command)[0]).name}"
            )


@pytest.mark.parametrize("provider,command", list(_commands()))
@pytest.mark.parametrize("directory", ["checkout", "checkout with spaces"])
def test_configured_hook_preserves_project_path_and_input(tmp_path, provider, command, directory):
    project = tmp_path / directory
    variable = f"{provider.upper()}_PROJECT_DIR"
    executable = shlex.split(command)
    assert len(executable) == 1
    hook = project / executable[0].removeprefix(f"${variable}/")
    hook.parent.mkdir(parents=True)
    # Exercise the real configured shell command without running actual hooks
    # (which may sync memories or contact external services).
    hook.write_text("#!/bin/sh\nprintf 'hook-ran:'\ncat\n")
    hook.chmod(0o755)
    payload = '{"tool_name":"Write","tool_input":{"file_path":"file with spaces.py"}}'
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=project,
        env={**os.environ, variable: str(project)},
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"hook-ran:{payload}"
