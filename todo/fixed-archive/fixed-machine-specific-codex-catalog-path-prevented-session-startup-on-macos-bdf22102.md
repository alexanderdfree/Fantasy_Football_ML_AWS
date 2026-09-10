> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Machine-specific Codex catalog path prevented session startup on macOS
- **File(s):** [../.codex/config.toml](../../.codex/config.toml), [../CODEX.md](../../CODEX.md).
- **What:** The tracked config referenced `/home/alex/.codex/my_catalog.json`, a WSL-local file absent on macOS. Desktop `config/read` failed with `failed to resolve feature override precedence: No such file or directory (os error 2)`, and `codex features list` failed to load configuration before any hook could run.
- **Fix:** Remove the repository catalog override so Codex discovers its catalog normally; retain the other project defaults. The desktop app-server config request and CLI feature listing both succeeded after removing the path, and all 54 existing Codex hook tests passed.
- **Lesson:** Keep host-specific catalog overrides in user configuration. When startup reports a missing file, reproduce configuration loading before changing or disabling hooks.
