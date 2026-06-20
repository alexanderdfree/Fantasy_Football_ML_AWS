# Tests-audit routine (on-demand)

Claude wrapper for the shared **tests-audit** routine — a deeper, test-suite-scoped
sibling of the general [codebase-audit routine](../audit/README.md). A Claude Code
remote agent runs on demand, works through per-shard test scopes and standing
test-specific cross-cutting lenses (coverage gaps, fixture/config drift,
test↔production parity, flakiness/isolation, assertion quality, stale/orphan
tests), dedupes against open+closed `claude-audit` and `codex-audit` GitHub
issues, and files **one issue per finding** under `claude-audit` — labeled by
severity, model regress-risk, and area — plus one closed
`[claude-audit] tests-audit checkpoint ...` issue per fire recording the audited
SHA (it carries no severity label, so it's excluded from the actionable backlog).

It **reuses the same `claude-audit`/`codex-audit` labels and `agent-audit/v1`
schema** as the general audit, so its per-finding issues are consumed by the same
[`solve-issues`](../../skills/solve-issues/SKILL.md) wrapper and dedupe against the
same shared pool (overlap with the general audit is auto-suppressed).

| | |
|---|---|
| Trigger ID | _unset — minted on first `/schedule` push (see "Deploying")_ |
| Dashboard | _set after first deploy_ |
| Schedule | disabled (on-demand only — `enabled: false`, empty cron) |
| Model | `claude-fable-5` |

## Architecture — the repo is what runs

The prompt deployed to the claude.ai dashboard is a **thin shim** ([`shim.md`](shim.md)).
At run time the routine checks out this repo at `main` HEAD and the shim reads the
shared master instructions at [`../../../routines/tests-audit/instructions.md`](../../../routines/tests-audit/instructions.md),
sets `AUDIT_LABEL=claude-audit`, applies Claude-only runtime rules, and executes the
shared workflow. So:

- **[`../../../routines/tests-audit/instructions.md`](../../../routines/tests-audit/instructions.md)**
  is the shared source of truth for *what the tests-audit does*. Version-controlled
  and reviewed via PR.
- **[`shim.md`](shim.md)** is what's deployed to the dashboard — the Claude runtime
  wrapper. It rarely changes.
- **[`prompt.md`](prompt.md)** is a compatibility alias mirroring the general audit's
  layout. Do not put shared audit logic there.
- **[`config.json`](config.json)** records the non-prompt deploy params (model, cron,
  environment, allowed tools, repo source, name). `trigger_id` is empty until first
  deploy.

Because the shim reads the shared instructions from the live checkout, an
instructions change takes effect the moment it lands on `main` — no dashboard push
needed.

## Deploying (first time)

This routine is **not deployed yet** — `config.json` has an empty `trigger_id`.
To stand it up:

1. Invoke `/schedule` (loads the `RemoteTrigger` tool) and say **"create the
   tests-audit routine"** (or create the trigger via https://claude.ai/code/routines).
2. Use [`shim.md`](shim.md) as the prompt content and the [`config.json`](config.json)
   params (model `claude-fable-5`, `enabled: false`, empty cron, the listed
   `allowed_tools`, the repo source).
3. Record the minted `trigger_id` + `dashboard_url` back into [`config.json`](config.json)
   and this table, then land that on `main`.

## Editing the routine

### Change what the tests-audit *does* (the common case) — no push

1. Edit [`../../../routines/tests-audit/instructions.md`](../../../routines/tests-audit/instructions.md).
2. Open a PR, merge to `main`.
3. The next fire reads the new shared instructions automatically.

> **Invariant:** `routines/tests-audit/instructions.md` must stay at exactly that
> path on `main`. If it's moved, renamed, or deleted, the wrapper STOPS without
> auditing rather than improvising.

### Change a deploy param (model / cron / allowed tools) or the shim itself — push required

1. Edit [`config.json`](config.json) and/or [`shim.md`](shim.md); land on `main`.
2. Invoke `/schedule` (loads the `RemoteTrigger` tool) and say **"push the
   tests-audit routine."**
3. The session fetches the current full `job_config`, sets
   `job_config.ccr.events[0].data.message.content` to the contents of `shim.md`,
   applies any `config.json` changes (model → `session_context.model`, cron →
   top-level `cron_expression`, enabled → top-level `enabled`, allowed tools →
   `session_context.allowed_tools`), generates a **fresh** lowercase v4 UUID for
   `events[0].data.uuid`, and calls `RemoteTrigger update`.
4. Verify: `RemoteTrigger get` → `events[0].data.message.content` matches `shim.md`.

### Gotchas

- **Fresh UUID every push.** Reusing the previous `events[].data.uuid` can silently
  no-op the update — generate a new lowercase v4 UUID each time.
- **JSON-escape multi-line content** with `jq -Rs '.'` when building the body.
- **Send the whole `job_config`,** not a sparse subtree.
- **Don't re-send `mcp_connections`.** Omitting it leaves existing connectors intact.
- **You can't delete the routine via the API** — do that at https://claude.ai/code/routines.

## Firing it manually

Once deployed, the recurring cron is disabled; fire by hand:

1. Invoke `/schedule` (loads the `RemoteTrigger` tool).
2. `RemoteTrigger run trigger_id=<this routine's trigger>` — one immediate run.
   If the API refuses to fire a disabled trigger, set a one-shot `run_once_at` a
   couple of minutes out via `RemoteTrigger update`; the platform auto-disables
   after it fires (`ended_reason: run_once_fired`).
