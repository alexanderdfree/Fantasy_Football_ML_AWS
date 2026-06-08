@AGENTS.md

# CODEX.md - Codex specifics

Everything imported from `AGENTS.md` is the shared project brain. This file documents the Codex-specific machinery that mirrors the Claude Code setup in `CLAUDE.md`.

## Hooks

Codex loads project hooks from `.codex/hooks.json` when the project is trusted. Review and trust them with `/hooks` after a hook file changes.

- `.codex/hooks/session-start.sh` adds project-specific startup context, warns when the session did not start from a clean Codex worktree, and runs a best-effort Codex memory pull from S3 via `scripts/agent-memory-sync.sh codex pull`. It cannot persist shell exports the way Claude's remote `SessionStart` hook writes `CLAUDE_ENV_FILE`, and it cannot move an already-running Codex session into a new worktree, so environment bootstrap remains a SETUP.md/manual step and fresh-worktree startup belongs in `scripts/codex-fresh-worktree.sh`.
- `.codex/hooks/guard-worktree-path.sh` blocks `apply_patch`/edit-style tool calls that target the main checkout while Codex is running in a worktree.
- `.codex/hooks/ruff-format.sh` formats touched Python files after `apply_patch`.
- `.codex/hooks/pre-pr.sh` wraps the repo's existing `.claude/hooks/pre-pr.sh` with `CLAUDE_PROJECT_DIR` set to the Codex project root, so the deterministic PR gate stays single-sourced.
- `.codex/hooks/post-pr-create.sh` emits a compact pointer to the prompt-backed post-PR review/CI/merge workflow after `gh pr create`; the workflow runs `scripts/codex-review-quiet.sh` so Codex/plugin loader warnings do not flood the chat context.
- `.codex/hooks/memory-sync-stop.sh` runs a best-effort `scripts/agent-memory-sync.sh all push` on `Stop`, pushing changed local Claude/Codex memory trees to their separate S3 prefixes.

Known limitation: Codex hooks are guardrails, not a complete enforcement boundary. They cover `apply_patch`, simple Bash hook events, and MCP calls that Codex exposes to hooks; they do not reliably intercept every possible shell-side file write. Keep using `apply_patch` for edits.

## Fresh Worktree Launcher

Start new local Codex sessions for this repo through:

```bash
scripts/codex-fresh-worktree.sh
```

The launcher reuses the current checkout only when it is a clean Codex-owned worktree under `${CODEX_HOME:-$HOME/.codex}/worktrees/*/Final-Project`. From the main checkout, a dirty worktree, or any other checkout shape, it creates `${CODEX_HOME:-$HOME/.codex}/worktrees/<id>/Final-Project` on `codex/session-<id>` from `origin/main`, best-effort links ignored `data/raw` and `data/splits` from the main checkout, and then runs `codex --cd <that-worktree>`.

Useful options: `--force-new`, `--base <ref>`, `--branch <name>`, `--no-fetch`, and `--print-path`. Use `--` before Codex arguments when a prompt or Codex option could be confused with a launcher option.

## Slash Prompts

Codex custom prompts are user-home scoped, not repo-scoped. The version-controlled prompt templates live in `.codex/prompts/`; install/update local copies with:

```bash
scripts/bootstrap-codex-local.sh
```

Restart Codex after installing. The prompts appear as:

- `/prompts:pre-pr-judge`
- `/prompts:pre-pr-gate`
- `/prompts:post-pr-followup`
- `/prompts:post-session-critique`
- `/prompts:solve-issues`

OpenAI now marks custom prompts deprecated in favor of skills, but prompts are still the closest Codex equivalent to Claude slash commands for this repo. If these workflows become stable enough to share across machines without a bootstrap copy step, graduate them into Codex skills.

## Gaps Versus Claude Code

- Claude has project-local slash skills under `.claude/skills/`; Codex custom prompts only load from `$CODEX_HOME/prompts`, so the repo tracks templates plus a bootstrap installer.
- Claude's `SessionStart` can mutate the remote session environment through `CLAUDE_ENV_FILE`; Codex hooks can add context but cannot persist `VIRTUAL_ENV`, `PATH`, `PYTHONPATH`, or a new working directory for later shell tools.
- Claude's worktree guard sees explicit `file_path` fields from Edit/Write tools. Codex primarily edits through `apply_patch`, so the guard parses patch headers and does not attempt broad Bash write detection.
- Claude's `/review` skill is an interactive tool. Codex's closest local equivalent is `scripts/codex-review-quiet.sh --base origin/main`, a wrapper around `codex review` that filters known loader-warning noise.

## Audit Automation Wrapper

Claude and Codex share audit behavior from [`routines/audit/instructions.md`](routines/audit/instructions.md). The Claude cloud routine sets `AUDIT_LABEL=claude-audit`; the tracked Codex wrapper at [`.codex/automations/audit/prompt.md`](.codex/automations/audit/prompt.md) sets `AUDIT_LABEL=codex-audit`. Both producers dedupe against open and closed severity-labeled issues from `claude-audit` and `codex-audit`, while keeping separate checkpoint history.

This repo tracks the Codex wrapper only; it does not create an active Codex app cron automation. If one is created later, point it at `.codex/automations/audit/prompt.md` and run it from a repo workspace. `/prompts:solve-issues` consumes both audit labels.

## Auto-memory

Claude and Codex memories are both machine-local recall caches, not sources of truth. They sync across the owner's machines via [scripts/agent-memory-sync.sh](scripts/agent-memory-sync.sh), but to separate S3 prefixes: `claude-memory/<repo>/memory/` and `codex-memory/<repo>/memories/`. Codex sync covers `${CODEX_HOME:-~/.codex}/memories/` only and excludes Codex's `.git` internals / SQLite runtime state. The project `SessionStart` hook pulls Codex memory; the `Stop` hook pushes both local memory trees if present, so cross-agent memory updates made locally reach both remotes. Durable, share-worthy project knowledge still belongs in [AGENTS.md](AGENTS.md).
