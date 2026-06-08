---
name: pre-pr-judge
description: Before `gh pr create`, vet the change against the original task — spawn a worker subagent that diffs the branch against `origin/main` and flags scope creep ("the agent did more than I asked"). Lightweight, agent-triggered companion to the user-triggered `/review`. Use after [.claude/hooks/pre-pr.sh](.claude/hooks/pre-pr.sh) passes but before opening the PR. Skip for trivial changes.
---

# Pre-PR judge wrapper

This is the Claude Code wrapper for the shared pre-PR judge workflow.

The authoritative instructions are version-controlled at `agent-workflows/pre-pr-judge/instructions.md`. That file, not this wrapper, defines the mission, skip rules, rebase requirement, verdict shape, and WARN handling.

Claude runtime values:

- `WORKFLOW_PROVIDER=Claude Code`
- `WORKFLOW_ENTRYPOINT=pre-pr-judge`
- `WORKFLOW_WRAPPER=.claude/skills/pre-pr-judge/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/pre-pr-judge/instructions.md`
- `WORKFLOW_BASE=origin/main` unless the user supplies another base
- `WORKFLOW_ORIGINAL_TASK=<infer from the current thread unless explicitly supplied>`
- `WORKFLOW_PRE_PR_GATE=.claude/hooks/pre-pr.sh`
- `WORKFLOW_REVIEW_TOOL=/review`
- `WORKFLOW_SUBAGENTS=Claude Agent tool, subagent_type=general-purpose`

Execution:

1. Read `agent-workflows/pre-pr-judge/instructions.md`.
2. If it is missing or empty, STOP NOW: do not run `gh pr create` and do not improvise the judge.
3. Otherwise, execute that file to completion using the Claude runtime values above.
