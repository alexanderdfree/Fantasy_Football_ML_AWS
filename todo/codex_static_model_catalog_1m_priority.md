# Codex static model catalog for 1M local context accounting

> Priority handoff. Goal: make the WSL Codex install pick up the same 1M-token
> local context accounting by pinning Codex to a static model catalog instead of
> the upstream metadata refresh.

## Context

Research found an undocumented Codex config key:

```toml
# ~/.codex/config.toml
model_catalog_json = "/home/alex/.codex/my_catalog.json"
```

When set, Codex reportedly uses a static model manager at startup, loads the JSON
file from disk, and skips the upstream model-metadata refresh. The intended use
is to clone the current `models_cache.json` into `my_catalog.json`, edit the
target model window fields to the desired 1M values, then point `config.toml` at
that static catalog.

Current local WSL state: both `~/.codex/my_catalog.json` and
`~/.codex/models_cache.json` have `gpt-5.4` and `gpt-5.5` patched to
`context_window = 1000000` and `max_context_window = 1000000`, and
`~/.codex/config.toml` points at the static catalog. Restart Codex before
calling this verified, because the active session may still reflect the model
catalog loaded at process start.

This should prevent a restart from overwriting the local window values, because
the upstream metadata endpoint is not queried while the static catalog is
configured.

## Why this matters

The upstream metadata currently caps `gpt-5.5` at `max_context_window = 272000`.
If Codex trusts the upstream cache on each startup, local edits to the cache can
be reverted before a WSL session starts. A static catalog would make the WSL
install deterministic and keep it aligned with the 1M local setup.

## Caveat

This is a community-documented workaround, not an official OpenAI guarantee. The
server may still reject, truncate, or otherwise constrain an API call even if the
local Codex accounting accepts a 1M-token window. Treat this as a local-accounting
override to test, not proof that the backend will honor 1M on every model.

## Pickup steps for WSL

1. Confirm the Codex version supports `model_catalog_json`.
2. Copy the current cache:

   ```bash
   cp ~/.codex/models_cache.json ~/.codex/my_catalog.json
   ```

3. Edit only the target model entries in both `~/.codex/my_catalog.json` and
   `~/.codex/models_cache.json`, setting the relevant context-window fields to
   `1000000`.
4. Add the static catalog path to `~/.codex/config.toml`:

   ```toml
   model_catalog_json = "/home/alex/.codex/my_catalog.json"
   ```

5. Restart Codex and verify the loaded model catalog still reports the 1M local
   window after startup.
6. Run a small smoke test before relying on a huge prompt: local accounting can
   pass while the server still enforces the upstream cap.

## Do not

- Do not commit `~/.codex/config.toml`, `models_cache.json`, or
  `my_catalog.json`; those are machine-local Codex files.
- Do not present this as an official OpenAI-supported setting until it is
  verified against official docs or source.
- Do not assume this bypasses server-side model limits.

## Source note

The reported key and behavior came from community documentation at
`codex.danielvaughan.com`. Verify against the installed Codex build before
depending on it.
