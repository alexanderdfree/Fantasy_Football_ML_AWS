> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Codex static model catalog verified for 1M WSL local context accounting
- **File(s):** [TODO.md](../../TODO.md) (removed the completed Open item), `todo/codex_static_model_catalog_1m_priority.md` (handoff notes, since removed). Machine-local Codex files (`~/.codex/config.toml`, `models_cache.json`, `my_catalog.json`) remain outside git.
- **What:** The Open item tracked a community-documented `model_catalog_json` workaround for WSL Codex: point startup at a static model catalog JSON so local `gpt-5.4` / `gpt-5.5` context-window accounting stays at 1M instead of being overwritten by upstream metadata on restart. The only remaining task was manual restart/backend smoke verification.
- **Fix:** User manually confirmed the WSL static-catalog setup is implemented and the local 1M context accounting survives startup, so the priority item is closed. No repository code or committed machine-local config changed.
- **Lesson:** Tool-local config work belongs in TODO only until the actual environment has been restarted and smoke-checked. Keep the operational caveat attached to the historical note: a static local catalog can fix Codex-side accounting, but it is still not an official OpenAI guarantee that every backend call will accept a 1M-token request.
