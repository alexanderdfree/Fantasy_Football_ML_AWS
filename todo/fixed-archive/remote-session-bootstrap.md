### [FIXED] Remote SessionStart installed outside the intended virtual environment

- **File(s):** `.claude/hooks/session-start.sh`, `.claude/settings.json`,
  `tests/scripts/test_claude_remote_bootstrap.py`; original fix
  `2c131d5d1` (PR #1533), consolidated with PR #1557.
- **What:** `uv venv` created an environment without pip, and the hook's bare
  `pip install` selected the system interpreter. A failed install prevented the
  session exports from being written. The old hook also duplicated a Torch pin
  instead of using the current dependency manifest.
- **Fix:** Seed pip, install the canonical development requirements through the
  explicit venv interpreter, recreate an incomplete environment, and verify
  imports and tool entrypoints. Write exports only after installation succeeds
  and quote paths so a workspace containing spaces survives the next shell.
  Original remote evidence recorded cold/warm success in approximately 25/5
  seconds; those are dated observations, not current timing guarantees.
- **Lesson:** Environment creation, package installation and later shell exports
  must refer to the same interpreter. Tests execute cold/warm and failed-install
  paths; a successful command listing alone does not establish a working session.
