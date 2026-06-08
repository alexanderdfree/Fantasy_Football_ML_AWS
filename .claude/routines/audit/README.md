# Scheduled codebase-audit routine

Claude wrapper for the shared code-audit routine. A Claude Code scheduled remote
agent runs every 2 hours, fans out parallel auditor subagents over the repo, dedupes
against open+closed `claude-audit` and `codex-audit` GitHub issues, and files
**one issue per finding** under `claude-audit` — labeled by severity
(`severity-high`/`severity-medium`) and area, HIGH/MED only — plus one closed
`[claude-audit] checkpoint ...` issue per fire recording the audited SHA (a
per-fire audit-trail entry; it carries no severity label, so it's excluded from
the actionable backlog). The per-finding issues it produces are consumed by the
[`solve-issues`](../../skills/solve-issues/SKILL.md) skill, which now triages
both `claude-audit` and `codex-audit`, severity-orders, and bundles them into
tier-by-risk PRs.

| | |
|---|---|
| Trigger ID | `trig_013gKH4q2g2TToBbCr4QDQcJ` |
| Dashboard | https://claude.ai/code/routines/trig_013gKH4q2g2TToBbCr4QDQcJ |
| Cron | `0 */2 * * *` UTC (every 2h) |
| Model | `claude-opus-4-7` |

## Architecture — the repo is what runs

The prompt deployed to the claude.ai dashboard is a **thin shim** ([`shim.md`](shim.md)). At
run time the routine checks out this repo at `main` HEAD and the shim reads the shared
master instructions at [`../../../routines/audit/instructions.md`](../../../routines/audit/instructions.md),
sets `AUDIT_LABEL=claude-audit`, applies Claude-only runtime rules, and executes the shared
workflow. So:

- **[`../../../routines/audit/instructions.md`](../../../routines/audit/instructions.md)** is the
  shared source of truth for *what the audit does*. Version-controlled and reviewed via PR.
- **[`shim.md`](shim.md)** is what's deployed to the dashboard — the Claude runtime wrapper.
  It rarely changes.
- **[`prompt.md`](prompt.md)** is a compatibility alias kept for any already-deployed dashboard
  shim that still reads the old path. Do not put shared audit logic there.
- **[`config.json`](config.json)** records the non-prompt deploy params (model, cron, environment,
  allowed tools, repo source, name).

Because the shim reads the shared instructions from the live checkout, an instructions change
takes effect the moment it lands on `main` — no dashboard push needed.

## Editing the routine

### Change what the audit *does* (the common case) — no push

1. Edit [`../../../routines/audit/instructions.md`](../../../routines/audit/instructions.md).
2. Open a PR, merge to `main`.
3. The next 2-hourly fire reads the new shared instructions automatically.

> **Invariant:** `routines/audit/instructions.md` must stay at exactly that path on `main`.
> If it's moved, renamed, or deleted, the wrappers STOP without auditing rather than improvising.
> Keep `.claude/routines/audit/prompt.md` tracked as a compatibility alias unless you have
> confirmed the deployed dashboard shim no longer reads it.

### Change a deploy param (model / cron / allowed tools) or the shim itself — push required

1. Edit [`config.json`](config.json) and/or [`shim.md`](shim.md); land on `main`.
2. Invoke `/schedule` (loads the `RemoteTrigger` tool) and say **"push the audit routine."**
3. The session:
   - runs `RemoteTrigger get` to fetch the **current full** `job_config`,
   - sets `job_config.ccr.events[0].data.message.content` to the contents of `shim.md`,
   - applies any `config.json` changes (model → `session_context.model`, cron →
     `cron_expression`, allowed tools → `session_context.allowed_tools`),
   - generates a **fresh** lowercase v4 UUID for `events[0].data.uuid`,
   - calls `RemoteTrigger update trigger_id=trig_013gKH4q2g2TToBbCr4QDQcJ body={...}`.
4. Verify: `RemoteTrigger get` → `next_run_at` is future-dated and `events[0].data.message.content`
   matches `shim.md`.

### Gotchas

- **Fresh UUID every push.** Reusing the previous `events[].data.uuid` can silently no-op the
  update — generate a new lowercase v4 UUID each time.
- **JSON-escape multi-line content** with `jq -Rs '.'` when building the body.
- **Send the whole `job_config`,** not a sparse subtree — nested partial-merge is unreliable and
  a sparse `job_config.ccr` can drop `session_context` / `environment_id` / `sources`.
- **Don't re-send `mcp_connections`.** It's a top-level sibling of `job_config`; a partial update
  that omits it leaves the existing connectors intact. (Re-sending risks clobbering them; use the
  `clear_mcp_connections` flag only if you truly mean to remove them.)
- **You can't delete the routine via the API** — do that at https://claude.ai/code/routines.
