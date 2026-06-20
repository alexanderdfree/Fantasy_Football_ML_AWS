@AGENTS.md

# GEMINI.md - Gemini CLI specifics

Everything above (imported from `AGENTS.md`) is the shared, provider-neutral project brain — read it first. **This file adds only the Gemini-CLI-specific machinery**: skills, auto-memory, and operational differences from Claude Code and Codex. Keep the shared disciplines in sync across `CLAUDE.md`, `CODEX.md`, and this file by updating `AGENTS.md`.

## Agent Skills (`.agents/skills/`)
Gemini has access to specialized project skills located in `.agents/skills/`. To invoke them, use the `activate_skill` tool (e.g., `activate_skill(name="solve-issues")`).
- **`solve-issues`**: Triage the `claude-audit` open issue backlog into file-disjoint tier-by-risk PRs for user approval.
- **`pre-pr-judge`**: Spawn a worker to diff the active branch against `origin/main` and flag scope creep before opening a PR.
- **`post-session-critique`**: Capture prompt lessons after a non-routine session to update `AGENTS.md` or memory.

Each is a thin wrapper that reads the shared, provider-neutral `agent-workflows/<name>/instructions.md` (the same source the Claude skills and Codex prompts use). (`worktree-cleanup` is Claude-only — it has no shared instructions file, so Gemini does not wrap it.)

## Auto-Memory & State
Unlike Claude and Codex, which maintain local SQL/JSON memory databases and sync them across machines via `scripts/agent-memory-sync.sh` to S3, Gemini CLI manages state through plain Markdown files:
- **Private Project Memory:** `~/.gemini/tmp/final-project/memory/MEMORY.md`. Use this for personal-to-the-user, project-specific notes, local setup facts, or private workflows that should **not** be committed to the repository.
- **Global Personal Memory:** `~/.gemini/GEMINI.md`. Use this for cross-project user preferences.
- **Durable Cross-Agent Project Facts:** Any fact, ML method, or Git workflow that applies to the entire team or multiple agents (Claude/Codex/Gemini) belongs in **`AGENTS.md`** — never just in Gemini's private memory.

## Hooks and Workflows
Claude and Codex enforce deterministic guardrails (the worktree-path guard, the pre-PR ruff/pytest gate) as blocking PreToolUse/PostToolUse hooks. **Gemini CLI supports the equivalent** — `BeforeTool`/`AfterTool` hooks (configured in `.gemini/settings.json` under a `hooks` key, default-on since v0.26.0) can block a tool call via exit code 2 or stdout `{"decision":"deny"}` — but **none are wired in this repo yet**. Until they are, Gemini's behavioral constraints are **prompt-enforced, not hook-enforced**: it relies on adherence to the MUST-rules in `AGENTS.md` (the "edit the worktree, not the parent" rule; "before any `gh pr create`, run ruff + `pytest -m unit` and abort on failure"). Porting `guard-worktree-path.sh` → a `BeforeTool` hook (matcher `write_file|replace|run_shell_command`) and `pre-pr.sh` → a `BeforeTool` hook on `run_shell_command` is a tracked follow-up.

## CI Workflows (`.github/workflows/gemini-*.yml`)
The [`run-gemini-cli`](https://github.com/google-github-actions/run-gemini-cli) GitHub-App integration lives in `.github/workflows/gemini-*.yml`, with command definitions in `.github/commands/gemini-*.toml` and model config in `.gemini/settings.json`. `gemini-dispatch.yml` routes events to the reusable `gemini-{review,triage,invoke,plan-execute}.yml` workflows (`@gemini-cli` comments from OWNER/MEMBER/COLLABORATOR, auto-review on PR open, auto-triage on issue open); `gemini-scheduled-triage.yml` runs an hourly issue-triage cron.

**Gated OFF by default.** The two entry points (`gemini-dispatch.yml`, `gemini-scheduled-triage.yml`) only run when the repo **variable** `GEMINI_ENABLED == 'true'`, so merging the workflows never auto-runs-and-fails (or spams error comments) before the backend is configured. To enable:
1. Stand up the Google backend — a Gemini API key, **or** Vertex AI via Workload Identity Federation — and install the gemini-cli GitHub App.
2. Configure repo **secrets** (`APP_PRIVATE_KEY`, `GEMINI_API_KEY` and/or `GOOGLE_API_KEY`) and **variables** (`APP_ID`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `SERVICE_ACCOUNT_EMAIL`, `GCP_WIF_PROVIDER`, `GEMINI_MODEL`, `GEMINI_CLI_VERSION`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_GENAI_USE_GCA`, `UPLOAD_ARTIFACTS`) — see each workflow's `with:` block for the authoritative set.
3. Set the repo variable `GEMINI_ENABLED=true`.

## Audit Routine Wrappers (`.agents/routines/`)
Gemini-side wrappers for the shared, read-only audit routines live under `.agents/routines/<name>/prompt.md`. They set `AUDIT_PROVIDER=Gemini` and — because Gemini has no provider-specific audit label — file under the shared `claude-audit` label, deduping against both providers' pools (the project decision is to reuse the existing `claude-audit`/`codex-audit` labels and the `agent-audit/v1` schema, not mint new ones). Tracked today:
- **audit** (general code audit): [`.agents/routines/audit/prompt.md`](.agents/routines/audit/prompt.md) → shared [`routines/audit/instructions.md`](routines/audit/instructions.md) (the general per-area + cross-cutting codebase audit; findings feed the `solve-issues` backlog).
- **tests-audit**: [`.agents/routines/tests-audit/prompt.md`](.agents/routines/tests-audit/prompt.md) → shared [`routines/tests-audit/instructions.md`](routines/tests-audit/instructions.md) (deep-audits the test suite).
- **infrastructure-audit**: [`.agents/routines/infrastructure-audit/prompt.md`](.agents/routines/infrastructure-audit/prompt.md) → shared [`routines/infrastructure-audit/instructions.md`](routines/infrastructure-audit/instructions.md) (deep-audits CI/CD, Batch/EC2, Docker, serving/ECS, IAM, artifact lifecycle).

These are tracked prompt templates; a Gemini run is pointed at one to execute it (no active automation is created here).
