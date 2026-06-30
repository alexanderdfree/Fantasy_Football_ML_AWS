@AGENTS.md

# GEMINI.md — Gemini CLI / Antigravity specifics

Everything above (imported from `AGENTS.md`) is the shared, provider-neutral project brain — read it first. **This file adds only the Gemini-specific machinery**: skills, auto-memory, and operational differences from Claude Code and Codex. Keep the shared disciplines in sync across `CLAUDE.md`, `CODEX.md`, and this file by updating `AGENTS.md`.

**Local runtime: Antigravity CLI (`agy`).** Locally this project's Gemini-family agent is run via Antigravity (`agy`), which is built on the Gemini-CLI lineage and reads the same config: root `AGENTS.md`, project skills in `.agents/skills/`, audit routines in `.agents/routines/`, and lifecycle hooks in `.gemini/settings.json` (under a `hooks` object — `BeforeTool`/`AfterTool`/`SessionStart`/`SessionEnd` events, regex `matcher` on `tool_name`). The CI surface (`run-gemini-cli` GitHub App, below) is the second runtime; both consume the same `.agents/` + `AGENTS.md` brain, so everything here applies to both unless noted.

## Agent Skills (`.agents/skills/`)
Gemini has access to specialized project skills located in `.agents/skills/`. To invoke them, use the `activate_skill` tool (e.g., `activate_skill(name="solve-issues")`).
- **`solve-issues`**: Triage the `claude-audit` open issue backlog into file-disjoint tier-by-risk PRs for user approval.
- **`pre-pr-judge`**: Spawn a worker to diff the active branch against `origin/main` and flag scope creep before opening a PR.
- **`post-session-critique`**: Capture prompt lessons after a non-routine session to update `AGENTS.md` or memory.

Each is a thin wrapper that reads the shared, provider-neutral `agent-workflows/<name>/instructions.md` (the same source the Claude skills and Codex prompts use). Codex also scans `.agents/skills/`, so these wrappers include both Codex and Gemini runtime values; Gemini uses the Gemini block. (`worktree-cleanup` is Claude-only — it has no shared instructions file, so Gemini does not wrap it.)

## Auto-Memory & State
Gemini CLI / Antigravity manages state through plain Markdown files (no SQL/JSON DB like Claude/Codex):
- **Private Project Memory:** `~/.gemini/tmp/final-project/memory/MEMORY.md`. Use this for personal-to-the-user, project-specific notes, local setup facts, or private workflows that should **not** be committed to the repository. **Synced across machines** (like Claude/Codex) via `scripts/agent-memory-sync.sh gemini {pull|push}` to `s3://$FF_MEMORY_S3_BUCKET/gemini-memory/<repo>/memory/` — wired into the `.gemini/` `SessionStart` (pull) / `SessionEnd` (push) hooks. Antigravity's project slug is not derivable by the script, so set **`GEMINI_MEMORY_DIR`** to the exact local memory path if the default (`~/.gemini/tmp/<main-checkout-basename-lowercased>/memory`) is wrong. The sync is a cross-machine cache of incidental recall, not a source of truth.
- **Global Personal Memory:** `~/.gemini/GEMINI.md`. Use this for cross-project user preferences (not synced).
- **Durable Cross-Agent Project Facts:** Any fact, ML method, or Git workflow that applies to the entire team or multiple agents (Claude/Codex/Gemini) belongs in **`AGENTS.md`** — never just in Gemini's private memory.

## Hooks (`.gemini/hooks/` + `.gemini/settings.json`)
Claude and Codex enforce deterministic guardrails as blocking hooks; Antigravity/Gemini CLI now does too. `.gemini/settings.json` registers them under the `hooks` object (events `BeforeTool`/`AfterTool`, regex `matcher` on the tool name); each command is `$GEMINI_PROJECT_DIR/.gemini/hooks/<name>.sh` and blocks via exit code 2 with the reason on stderr. The adapters are parity twins of `.codex/hooks/*` (a third `lib.sh`; Antigravity passes `tool_name` + `tool_input`, with `write_file`/`replace` carrying `.tool_input.file_path` and `run_shell_command` carrying `.tool_input.command`):

- **`guard-worktree-path.sh`** — `BeforeTool` on `write_file|replace`: blocks edits that target the parent checkout from inside a worktree (jq→python3 fallback keeps it armed without jq).
- **`pre-pr.sh`** — `BeforeTool` on `run_shell_command`: on a top-level `gh pr create`, delegates to the single-source gate `.claude/hooks/pre-pr.sh` (ruff + `pytest -m unit` + benchmark freshness), exactly as the Codex wrapper does.
- **`ruff-format.sh`** — `AfterTool` on `write_file|replace`: `ruff format` the touched `.py` (probes the worktree + main `.venv`).

Pinned by [`tests/scripts/test_gemini_hooks.py`](tests/scripts/test_gemini_hooks.py); cross-provider hook/skill/routine parity is pinned by [`tests/scripts/test_cross_model_parity.py`](tests/scripts/test_cross_model_parity.py). The `SessionStart`/`SessionEnd` memory-sync hooks are wired (#1283). These fire in the **local Antigravity (`agy`)** runtime; CI runs on a clean checkout where the worktree/pre-PR guards do not apply.

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
