---
name: solve-issues
description: Triage the open [claude-audit]/[codex-audit] GitHub issues (one finding per issue, labeled by severity + model regress-risk + area) and bundle the fixes into tier-by-risk PRs for user approval. Verifies each finding, classifies FIX vs LEAVE (skipping stale claims, false positives, and stop-rule-violating feature drift), closes LEAVE issues, and partitions FIX work into file-disjoint Tier A/B/C bundles. Also has a verify-then-close mode for an already-remediated backlog passed explicitly. Trigger to clear the audit backlog; not for one-off bug fixes.
---

# Solve-issues wrapper

This is the shared Codex/Gemini skill wrapper for the audit-issue workflow.

The authoritative instructions are version-controlled at `agent-workflows/solve-issues/instructions.md`. That file, not this wrapper, defines issue parsing, Mode A/B, FIX/LEAVE categories, regress-risk handling, tier bundling, approval gates, and close/merge rules.

Codex runtime values:

- `WORKFLOW_PROVIDER=Codex`
- `WORKFLOW_ENTRYPOINT=$solve-issues` or implicit skill invocation; `/prompts:solve-issues` is the legacy prompt alias
- `WORKFLOW_WRAPPER=.agents/skills/solve-issues/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/solve-issues/instructions.md`
- `WORKFLOW_ISSUES=<optional issue/range/list supplied by the user>`
- `WORKFLOW_DRY_RUN=0 unless the user explicitly asks for dry-run planning only`
- `WORKFLOW_PLAN_MODE=produce a compact plan and wait for user approval when explicit plan-mode tools are unavailable`
- `WORKFLOW_SUBAGENTS=Codex subagents when available; otherwise manual tier-by-tier execution in the same file-disjoint order`
- `WORKFLOW_PRE_PR_GATE=/prompts:pre-pr-gate when installed; otherwise run the deterministic checks manually`
- `WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT=$pre-pr-judge or /prompts:pre-pr-judge`
- `WORKFLOW_REVIEW_TOOL=scripts/codex-review-quiet.sh --base origin/main`
- `WORKFLOW_MEMORY_DESTINATION=$CODEX_HOME/memories, plus AGENTS.md for durable cross-agent lessons`

Gemini runtime values:

- `WORKFLOW_PROVIDER=Gemini CLI`
- `WORKFLOW_ENTRYPOINT=activate_skill(name="solve-issues")`
- `WORKFLOW_WRAPPER=.agents/skills/solve-issues/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/solve-issues/instructions.md`
- `WORKFLOW_ISSUES=<optional issue/range/list supplied by the user>`
- `WORKFLOW_DRY_RUN=0 unless the user explicitly asks for dry-run planning only`
- `WORKFLOW_PLAN_MODE=present the plan and stop for explicit user approval before any mutation`
- `WORKFLOW_SUBAGENTS=delegate implementation bundles to Gemini subagents if available; otherwise work them sequentially`
- `WORKFLOW_PRE_PR_GATE=the wired .gemini BeforeTool hook (.gemini/hooks/pre-pr.sh on run_shell_command, delegating to the single-source .claude/hooks/pre-pr.sh) gates a top-level gh pr create; run the deterministic checks manually — ruff check . && ruff format --check . && pytest -m unit — only as a fallback when the hook is unavailable`
- `WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT=activate_skill(name="pre-pr-judge")`
- `WORKFLOW_REVIEW_TOOL=the @gemini-cli /review PR workflow (.github/workflows/gemini-review.yml)`
- `WORKFLOW_MEMORY_DESTINATION=Gemini Markdown memory, plus AGENTS.md for durable cross-agent lessons`

Execution:

1. Read `agent-workflows/solve-issues/instructions.md`.
2. If it is missing or empty, STOP NOW: do not close issues, edit files, commit, push, open PRs, or improvise an audit-backlog workflow.
3. Otherwise, execute that file to completion using the runtime values for the active provider.
