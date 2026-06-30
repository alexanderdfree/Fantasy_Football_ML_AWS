---
name: pre-pr-judge
description: Before opening a PR, vet the change against the original task by diffing against `origin/main` and flagging scope creep. Use in Codex or Gemini after deterministic pre-PR checks pass and before `gh pr create`. Skip for trivial changes.
---

# Pre-PR judge wrapper

This is the shared Codex/Gemini skill wrapper for the pre-PR judge workflow.

The authoritative instructions are version-controlled at `agent-workflows/pre-pr-judge/instructions.md`. That file, not this wrapper, defines the mission, skip rules, rebase requirement, verdict shape, and WARN handling.

Codex runtime values:

- `WORKFLOW_PROVIDER=Codex`
- `WORKFLOW_ENTRYPOINT=$pre-pr-judge` or implicit skill invocation; `/prompts:pre-pr-judge` is the legacy prompt alias
- `WORKFLOW_WRAPPER=.agents/skills/pre-pr-judge/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/pre-pr-judge/instructions.md`
- `WORKFLOW_BASE=origin/main` unless the user supplies another base
- `WORKFLOW_ORIGINAL_TASK=<infer from the current thread unless explicitly supplied>`
- `WORKFLOW_PRE_PR_GATE=/prompts:pre-pr-gate when installed; otherwise run the deterministic checks manually`
- `WORKFLOW_REVIEW_TOOL=scripts/codex-review-quiet.sh --base origin/main`
- `WORKFLOW_SUBAGENTS=Codex subagents when available; otherwise direct orchestrator judgment`

Gemini runtime values:

- `WORKFLOW_PROVIDER=Gemini CLI`
- `WORKFLOW_ENTRYPOINT=activate_skill(name="pre-pr-judge")`
- `WORKFLOW_WRAPPER=.agents/skills/pre-pr-judge/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/pre-pr-judge/instructions.md`
- `WORKFLOW_BASE=origin/main` unless the user supplies another base
- `WORKFLOW_ORIGINAL_TASK=<infer from the current thread unless explicitly supplied>`
- `WORKFLOW_PRE_PR_GATE=the wired .gemini BeforeTool hook (.gemini/hooks/pre-pr.sh on run_shell_command, delegating to the single-source .claude/hooks/pre-pr.sh) gates a top-level gh pr create; run the deterministic checks manually — ruff check . && ruff format --check . && pytest -m unit — only as a fallback when the hook is unavailable`
- `WORKFLOW_REVIEW_TOOL=the @gemini-cli /review PR workflow (.github/workflows/gemini-review.yml), heavier and post-PR`
- `WORKFLOW_SUBAGENTS=delegate the diff-vs-intent read to a Gemini subagent if available; otherwise perform the check directly`

Execution:

1. Read `agent-workflows/pre-pr-judge/instructions.md`.
2. If it is missing or empty, STOP NOW: do not run `gh pr create` and do not improvise the judge.
3. Otherwise, execute that file to completion using the runtime values for the active provider.
