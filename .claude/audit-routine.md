# Audit routine — handoff pointer

This file exists so any Claude Code session opening this repo can pick up the
scheduled audit routine and complete Option C (move the prompt source-of-truth
into the repo). Until that's done, the routine prompt lives only at claude.ai.

## What the routine does

Every 2 hours, fans out 12 parallel Opus 4.7 1M subagents over the codebase, dedupes
against past `claude-audit`-labeled GitHub issues, and creates **one new issue per fire**
with that run's findings (per-run model, ≤55k char body bound, F-numbers scoped per-issue,
"first seen" SHA stamped on each finding). Skips if HEAD-SHA has already been audited
(checked against last-10 audit issues' bodies + comments).

## Cloud routine identifiers

| | |
|---|---|
| Trigger ID | `trig_013gKH4q2g2TToBbCr4QDQcJ` |
| Dashboard | https://claude.ai/code/routines/trig_013gKH4q2g2TToBbCr4QDQcJ |
| Owner account | `45b12d4c-520d-4f1b-9644-974cebd5e6db` (Alex) |
| Repo source | https://github.com/alexanderdfree/Fantasy_Football_ML_AWS |
| Cron | `0 */2 * * *` UTC (every 2h) |
| Model | `claude-opus-4-7` |
| Environment ID | `env_01WCEhA61gYJ8u7K8aZ3AyVU` |
| Allowed tools | `Bash, Read, Write, Glob, Grep, Agent, TodoWrite` |

Findings land as new GitHub issues labeled `claude-audit` plus area labels
(`qb`, `rb`, `wr`, `te`, `k`, `dst`, `shared`, `data`, `serving`, `batch`, `ci`, `docs`).

## How to implement Option C from here

1. **Extract the cloud prompt.** Invoke `/schedule` to load the `RemoteTrigger` tool, then:

       RemoteTrigger action=get trigger_id=trig_013gKH4q2g2TToBbCr4QDQcJ

   Save `job_config.ccr.events[0].data.message.content` verbatim to
   `.claude/routines/audit/prompt.md`.

2. **Write `.claude/routines/audit/config.json`** with this shape:

       {
         "trigger_id": "trig_013gKH4q2g2TToBbCr4QDQcJ",
         "dashboard_url": "https://claude.ai/code/routines/trig_013gKH4q2g2TToBbCr4QDQcJ",
         "model": "claude-opus-4-7",
         "environment_id": "env_01WCEhA61gYJ8u7K8aZ3AyVU",
         "cron_expression": "0 */2 * * *",
         "allowed_tools": ["Bash", "Read", "Write", "Glob", "Grep", "Agent", "TodoWrite"],
         "sources": [
           {"git_repository": {"url": "https://github.com/alexanderdfree/Fantasy_Football_ML_AWS"}}
         ],
         "name": "Codebase audit fanout (every 2h, skip-if-HEAD-already-audited)"
       }

3. **Write `.claude/routines/audit/README.md`** documenting the push recipe:
   - Edit `prompt.md` (or `config.json` for non-prompt fields).
   - Invoke `/schedule` to load `RemoteTrigger`.
   - Say "push the audit routine"; the session generates a fresh UUID, builds the API body
     from the two files, and calls `RemoteTrigger update`.
   - Verify with `RemoteTrigger get` — confirm `next_run_at` is future-dated and the first
     ~200 chars of the prompt match the file.
   - Gotchas: always fresh UUID per push (re-use can no-op silently); use `jq -Rs '.'` to
     JSON-escape multi-line content; partial updates only — don't re-send `mcp_connections`.

4. **Update `CLAUDE.md`** with a short "Scheduled routines" section pointing at
   `.claude/routines/audit/`. So future sessions see it during orientation.

5. **Delete this handoff file** (`.claude/audit-routine.md`) once the structured directory
   is in place. The directory is the new home.

## Context about the routine's evolution

The prompt has been iterated many times on 2026-05-21 (per-run → rolling → re-verification
→ F-numbering → consolidation → back to per-run after hitting GitHub's 65k body cap).
The current cloud state is the per-run model with all guardrails. The earlier rolling
issue #320 was split into per-area issues #338–#348 during the rollback.

Don't try to "fix" design choices reflected in the cloud prompt without first reading
the closed `claude-audit` issues #281/#289/#296/#306/#316 + the per-area issues
#338–#348 — each iteration captured a real lesson, and the prompt's complexity exists
for reasons. Treat the cloud prompt as the source of truth in Step 1 and inherit it
verbatim before touching anything.
