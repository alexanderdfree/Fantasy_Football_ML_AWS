@AGENTS.md

# GEMINI.md - Gemini CLI specifics

Everything above (imported from `AGENTS.md`) is the shared, provider-neutral project brain — read it first. **This file adds only the Gemini-CLI-specific machinery**: skills, auto-memory, and operational differences from Claude Code and Codex. Keep the shared disciplines in sync across `CLAUDE.md`, `CODEX.md`, and this file by updating `AGENTS.md`.

## Agent Skills (`.agents/skills/`)
Gemini has access to specialized project skills located in `.agents/skills/`. To invoke them, use the `activate_skill` tool (e.g., `activate_skill(name="solve-issues")`).
- **`solve-issues`**: Triage the `claude-audit` open issue backlog into file-disjoint tier-by-risk PRs for user approval.
- **`pre-pr-judge`**: Spawn a worker to diff the active branch against `origin/main` and flag scope creep before opening a PR.
- **`post-session-critique`**: Capture prompt lessons after a non-routine session to update `AGENTS.md` or memory.
- **`worktree-cleanup`**: Produce a per-worktree FLAG/DELETE/KEEP plan for stale `.Codex/worktrees/` and offer to execute it.

## Auto-Memory & State
Unlike Claude and Codex, which maintain local SQL/JSON memory databases and sync them across machines via `scripts/agent-memory-sync.sh` to S3, Gemini CLI manages state through plain Markdown files:
- **Private Project Memory:** `~/.gemini/tmp/final-project/memory/MEMORY.md`. Use this for personal-to-the-user, project-specific notes, local setup facts, or private workflows that should **not** be committed to the repository.
- **Global Personal Memory:** `~/.gemini/GEMINI.md`. Use this for cross-project user preferences.
- **Durable Cross-Agent Project Facts:** Any fact, ML method, or Git workflow that applies to the entire team or multiple agents (Claude/Codex/Gemini) belongs in **`AGENTS.md`** — never just in Gemini's private memory.

## Hooks and Workflows
Currently, Gemini CLI relies on standard system hooks and manual invocation of the scripts in `scripts/` where appropriate. While Claude and Codex have dedicated guardrail hooks (e.g., `.claude/hooks/guard-worktree-path.sh`), Gemini's core behavioral constraints and context filtering are defined natively by its overarching system prompt and its adherence to `AGENTS.md`. 
