---
name: solve-issues
description: Triage the open [claude-audit]/[codex-audit] GitHub issues (one finding per issue, labeled by severity + model regress-risk + area) and bundle the fixes into tier-by-risk PRs for user approval. Enters plan mode: verifies each finding, classifies FIX vs LEAVE (skipping stale claims, false positives, and stop-rule-violating feature drift), closes LEAVE issues, and partitions FIX work into file-disjoint Tier A/B/C bundles. Also has a verify-then-close mode for an already-remediated backlog passed explicitly (e.g. `/solve-issues #338-348`). Trigger with `/solve-issues` to clear the audit backlog; not for one-off bug fixes.
---

# Solve-issues wrapper

This is the Claude Code wrapper for the shared audit-issue workflow.

The authoritative instructions are version-controlled at `agent-workflows/solve-issues/instructions.md`. That file, not this wrapper, defines issue parsing, Mode A/B, FIX/LEAVE categories, regress-risk handling, tier bundling, approval gates, and close/merge rules.

Claude runtime values:

- `WORKFLOW_PROVIDER=Claude Code`
- `WORKFLOW_ENTRYPOINT=solve-issues`
- `WORKFLOW_WRAPPER=.claude/skills/solve-issues/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/solve-issues/instructions.md`
- `WORKFLOW_ISSUES=<optional issue/range/list supplied by the user>`
- `WORKFLOW_DRY_RUN=0 unless the user explicitly asks for dry-run planning only`
- `WORKFLOW_PLAN_MODE=EnterPlanMode/ExitPlanMode when available; otherwise stop for explicit approval`
- `WORKFLOW_SUBAGENTS=Claude Agent workers, using isolated worktrees for implementation bundles`
- `WORKFLOW_PRE_PR_GATE=.claude/hooks/pre-pr.sh`
- `WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT=pre-pr-judge`
- `WORKFLOW_REVIEW_TOOL=/review`
- `WORKFLOW_MEMORY_DESTINATION=Claude project auto-memory, plus AGENTS.md for durable cross-agent lessons`

Execution:

1. Read `agent-workflows/solve-issues/instructions.md`.
2. If it is missing or empty, STOP NOW: do not close issues, edit files, commit, push, open PRs, or improvise an audit-backlog workflow.
3. Otherwise, execute that file to completion using the Claude runtime values above.
