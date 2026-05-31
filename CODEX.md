@AGENTS.md

# CODEX.md - Codex specifics

Everything imported from `AGENTS.md` is the shared project brain. This file documents the Codex-specific machinery that mirrors the Claude Code setup in `CLAUDE.md`.

## Hooks

Codex loads project hooks from `.codex/hooks.json` when the project is trusted. Review and trust them with `/hooks` after a hook file changes.

- `.codex/hooks/session-start.sh` adds project-specific startup context and runs a best-effort Codex memory pull from S3 via `scripts/agent-memory-sync.sh codex pull`. It cannot persist shell exports the way Claude's remote `SessionStart` hook writes `CLAUDE_ENV_FILE`, so environment bootstrap remains a SETUP.md/manual step.
- `.codex/hooks/guard-worktree-path.sh` blocks `apply_patch`/edit-style tool calls that target the main checkout while Codex is running in a worktree.
- `.codex/hooks/ruff-format.sh` formats touched Python files after `apply_patch`.
- `.codex/hooks/pre-pr.sh` wraps the repo's existing `.claude/hooks/pre-pr.sh` with `CLAUDE_PROJECT_DIR` set to the Codex project root, so the deterministic PR gate stays single-sourced.
- `.codex/hooks/post-pr-create.sh` injects Codex's post-PR review/CI/merge workflow after `gh pr create`.
- `.codex/hooks/memory-sync-stop.sh` runs a best-effort `scripts/agent-memory-sync.sh all push` on `Stop`, pushing changed local Claude/Codex memory trees to their separate S3 prefixes.

Known limitation: Codex hooks are guardrails, not a complete enforcement boundary. They cover `apply_patch`, simple Bash hook events, and MCP calls that Codex exposes to hooks; they do not reliably intercept every possible shell-side file write. Keep using `apply_patch` for edits.

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
- Claude's `SessionStart` can mutate the remote session environment through `CLAUDE_ENV_FILE`; Codex hooks can add context but cannot persist `VIRTUAL_ENV`, `PATH`, or `PYTHONPATH` for later shell tools.
- Claude's worktree guard sees explicit `file_path` fields from Edit/Write tools. Codex primarily edits through `apply_patch`, so the guard parses patch headers and does not attempt broad Bash write detection.
- Claude's `/review` skill is an interactive tool. Codex's closest local equivalent is `codex review --base origin/main`.
- Claude's scheduled audit routine is a claude.ai cloud routine. Codex has no matching scheduled local routine in this repo yet; Codex can consume the resulting `claude-audit` issue backlog through `/prompts:solve-issues`.

## Auto-memory

Claude and Codex memories are both machine-local recall caches, not sources of truth. They sync across the owner's machines via [scripts/agent-memory-sync.sh](scripts/agent-memory-sync.sh), but to separate S3 prefixes: `claude-memory/<repo>/memory/` and `codex-memory/<repo>/memories/`. Codex sync covers `${CODEX_HOME:-~/.codex}/memories/` only and excludes Codex's `.git` internals / SQLite runtime state. The project `SessionStart` hook pulls Codex memory; the `Stop` hook pushes both local memory trees if present, so cross-agent memory updates made locally reach both remotes. Durable, share-worthy project knowledge still belongs in [AGENTS.md](AGENTS.md).
