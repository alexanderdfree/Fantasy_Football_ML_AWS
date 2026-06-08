---
description: Vet the current branch against the original task before opening a PR
argument-hint: [ORIGINAL_TASK="<quoted task>"] [BASE=origin/main]
---

Run the Final-Project Codex pre-PR judge wrapper.

The authoritative instructions are version-controlled at `agent-workflows/pre-pr-judge/instructions.md`. That file, not this wrapper, defines the mission, skip rules, rebase requirement, verdict shape, and WARN handling.

Codex runtime values:

- `WORKFLOW_PROVIDER=Codex`
- `WORKFLOW_ENTRYPOINT=/prompts:pre-pr-judge`
- `WORKFLOW_WRAPPER=.codex/prompts/pre-pr-judge.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/pre-pr-judge/instructions.md`
- `WORKFLOW_BASE=BASE if supplied, otherwise origin/main`
- `WORKFLOW_ORIGINAL_TASK=ORIGINAL_TASK if supplied, otherwise infer from the current thread`
- `WORKFLOW_PRE_PR_GATE=/prompts:pre-pr-gate`
- `WORKFLOW_REVIEW_TOOL=scripts/codex-review-quiet.sh --base origin/main`
- `WORKFLOW_SUBAGENTS=Codex subagents when available; otherwise direct orchestrator judgment`

Execution:

1. Read `agent-workflows/pre-pr-judge/instructions.md`.
2. If it is missing or empty, STOP NOW: do not run `gh pr create` and do not improvise the judge.
3. Otherwise, execute that file to completion using the Codex runtime values and supplied arguments above.
