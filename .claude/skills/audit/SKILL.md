---
name: audit
description: Run the repo's codebase audit locally, in-session — the same shared instructions the scheduled cloud routine executes. `/audit` = general code audit; `/audit tests` = test-suite audit; `/audit infrastructure` = CI/CD + AWS infra audit. Read-only on repo files; files one GitHub issue per verified finding under `claude-audit` (severity + regress-risk + area labels) plus one closed checkpoint issue. Use for an audit fire without deploying/triggering the cloud routine; expect a long run (2h wall-clock cap).
---

# Audit wrapper (local)

This is the Claude Code **local skill** wrapper for the shared audit routines — the
in-session twin of the deployed cloud shim `.claude/routines/audit/shim.md`. (Codex
and Gemini already run locally from their tracked routine prompts,
`.codex/automations/<routine>/prompt.md` / `.agents/routines/<routine>/prompt.md`;
this skill is the Claude-side equivalent, so it is Claude-only by design.)

Resolve the skill argument to one shared instructions file:

| Argument | Shared instructions |
|---|---|
| *(none)* or `code` | `routines/audit/instructions.md` |
| `tests` | `routines/tests-audit/instructions.md` |
| `infrastructure` or `infra` | `routines/infrastructure-audit/instructions.md` |

Any other argument: stop and ask the user which routine they meant.

That instructions file, not this wrapper, defines the mission, scopes, issue schema,
dedupe rules, severity/regress-risk labels, stop rules, verification depth,
consolidation passes, and checkpoint behavior.

Claude runtime values (identical for all three routines):

- `AUDIT_PROVIDER=Claude`
- `AUDIT_LABEL=claude-audit`
- `DEDUPE_AUDIT_LABELS="claude-audit codex-audit"`

Claude-local runtime/tool rules (the deltas vs the cloud shim, which audits a fresh
checkout of `main` at HEAD):

- **Audit published main, not the session branch.** `git fetch origin main --quiet`;
  if `git rev-parse HEAD` == `git rev-parse origin/main` and `git status --porcelain`
  is empty, audit in place. Otherwise create a temporary detached worktree at
  `origin/main` outside the repo (e.g. `git worktree add --detach
  <scratchpad>/audit-main origin/main`), run the entire audit with that directory as
  the working root, and `git worktree remove` it afterward (best-effort;
  `git worktree prune` on failure). The audited SHA in findings and the checkpoint
  is that HEAD. Never audit a dirty or diverged checkout.
- For the fanout step, spawn Claude Agent subagents; let workers inherit the
  session's model (omit `model:`). Worker count, scope grouping, and batching are up
  to you — a worker may cover several scopes; avoid over-spawning.
- Keep the run read-only on repo files. The only permitted writes are those allowed
  by the resolved instructions file: `/tmp/*` bookkeeping (the session scratchpad is
  an acceptable substitute — the paths are internal), idempotent label creation,
  per-finding issue creation, comments on issues created in this run, and closing
  the checkpoint created in this run. No edits, commits, branches, or PRs.
- Before fanout, tell the user in one line which routine is running and the audited
  SHA.

Execution:

1. Resolve the routine from the argument (table above).
2. Read the resolved `routines/<routine>/instructions.md`.
3. If it is missing or empty, STOP NOW: do not create any GitHub issue and do not
   improvise an audit.
4. Otherwise, execute that file to completion using the runtime values and
   Claude-local rules above.
