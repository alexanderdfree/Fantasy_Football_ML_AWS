---
description: Triage agent audit issues into tier-by-risk Codex PRs
argument-hint: [ISSUES="<numbers or range>"] [DRY_RUN=1]
---

Run the Final-Project Codex solve-issues wrapper.

The authoritative instructions are version-controlled at `agent-workflows/solve-issues/instructions.md`. That file, not this wrapper, defines issue parsing, Mode A/B, FIX/LEAVE categories, regress-risk handling, tier bundling, approval gates, and close/merge rules.

Codex runtime values:

- `WORKFLOW_PROVIDER=Codex`
- `WORKFLOW_ENTRYPOINT=/prompts:solve-issues`
- `WORKFLOW_WRAPPER=.codex/prompts/solve-issues.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/solve-issues/instructions.md`
- `WORKFLOW_ISSUES=ISSUES if supplied, otherwise enumerate the open severity-labeled backlog`
- `WORKFLOW_DRY_RUN=1 when invoked with DRY_RUN=1; otherwise 0`
- `WORKFLOW_PLAN_MODE=produce a compact plan and wait for user approval when explicit plan-mode tools are unavailable`
- `WORKFLOW_SUBAGENTS=Codex subagents when available; otherwise manual tier-by-tier execution in the same file-disjoint order`
- `WORKFLOW_PRE_PR_GATE=/prompts:pre-pr-gate`
- `WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT=/prompts:pre-pr-judge`
- `WORKFLOW_REVIEW_TOOL=scripts/codex-review-quiet.sh --base origin/main`
- `WORKFLOW_MEMORY_DESTINATION=$CODEX_HOME/memories, plus AGENTS.md for durable cross-agent lessons`

Execution:

1. Read `agent-workflows/solve-issues/instructions.md`.
2. If it is missing or empty, STOP NOW: do not close issues, edit files, commit, push, open PRs, or improvise an audit-backlog workflow.
3. Otherwise, execute that file to completion using the Codex runtime values and supplied arguments above.
