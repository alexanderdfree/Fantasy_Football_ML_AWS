---
name: pre-pr-judge
description: Before opening a PR, vet the change against the original task — diff the branch against `origin/main` and flag scope creep ("the agent did more than I asked"). Lightweight companion to the heavier `@gemini-cli /review`. Run after the deterministic pre-PR checks pass but before `gh pr create`. Skip for trivial changes.
---

# Pre-PR judge wrapper

This is the Gemini CLI wrapper for the shared pre-PR judge workflow.

The authoritative instructions are version-controlled at `agent-workflows/pre-pr-judge/instructions.md`. That file, not this wrapper, defines the mission, skip rules, rebase requirement, verdict shape, and WARN handling.

Gemini runtime values:

- `WORKFLOW_PROVIDER=Gemini CLI`
- `WORKFLOW_ENTRYPOINT=activate_skill(name="pre-pr-judge")`
- `WORKFLOW_WRAPPER=.agents/skills/pre-pr-judge/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/pre-pr-judge/instructions.md`
- `WORKFLOW_BASE=origin/main` unless the user supplies another base
- `WORKFLOW_ORIGINAL_TASK=<infer from the current thread unless explicitly supplied>`
- `WORKFLOW_PRE_PR_GATE=run the deterministic checks manually — ruff check . && ruff format --check . && pytest -m unit (Gemini CLI has no pre-PR hook)`
- `WORKFLOW_REVIEW_TOOL=the @gemini-cli /review PR workflow (.github/workflows/gemini-review.yml), heavier and post-PR`
- `WORKFLOW_SUBAGENTS=delegate the diff-vs-intent read to a Gemini subagent if available; otherwise perform the check directly`

Execution:

1. Read `agent-workflows/pre-pr-judge/instructions.md`.
2. If it is missing or empty, STOP NOW: do not run `gh pr create` and do not improvise the judge.
3. Otherwise, execute that file to completion using the Gemini runtime values above.
