# Shared pre-PR judge instructions

These are the provider-neutral instructions for the repo's scope-creep judge.
Provider wrappers must read this file, set the runtime values below, apply their
own tool/worktree rules, and then execute the shared workflow.

## Runtime contract

The wrapper must define these values before executing the workflow:

- `WORKFLOW_PROVIDER`: human-readable provider name, for example `Claude Code` or `Codex`.
- `WORKFLOW_ENTRYPOINT`: the user-facing invocation, for example `pre-pr-judge` or `/prompts:pre-pr-judge`.
- `WORKFLOW_WRAPPER`: the wrapper file that loaded this instruction file.
- `WORKFLOW_SHARED_INSTRUCTIONS`: `agent-workflows/pre-pr-judge/instructions.md`.
- `WORKFLOW_BASE`: the base ref supplied by the user or `origin/main` by default.
- `WORKFLOW_ORIGINAL_TASK`: an explicitly supplied original task, or empty when the provider must infer it from the current thread.
- `WORKFLOW_PRE_PR_GATE`: the deterministic pre-PR gate for the provider. For Codex this is `/prompts:pre-pr-gate`; do not execute `.codex/hooks/pre-pr.sh` directly because it is a PreToolUse hook that expects hook JSON.
- `WORKFLOW_REVIEW_TOOL`: the heavier post-PR review tool, for example `/review` or `scripts/codex-review-quiet.sh --base origin/main`.
- `WORKFLOW_SUBAGENTS`: whether the provider can delegate the diff-vs-intent read to a worker/subagent.

Provider wrappers own dispatch mechanics: whether subagents are available, how to
quote the original task from the session, and how to surface a warning. Do not
let provider mechanics change the skip list, verdict shape, or requirement to
stop before `gh pr create` on WARN.

## Provider entrypoints

- Claude wrapper: `.claude/skills/pre-pr-judge/SKILL.md`.
- Codex wrapper: `.codex/prompts/pre-pr-judge.md`.

The wrappers stay discoverable at those paths. This file is the behavioral source
of truth.

## Mission

Before `gh pr create`, vet the current branch against the original user task.
The deterministic gate catches what ruff, pytest, format checks, and benchmark
freshness can express. This workflow catches what they cannot: whether the agent
did more, less, or something different than the user asked.

`WORKFLOW_REVIEW_TOOL` is the heavyweight, post-PR review surface. This workflow
is the lightweight, pre-PR scope check. It judges scope alignment only; it is not
a code-quality review.

## When to run

Run after `WORKFLOW_PRE_PR_GATE` passes and immediately before `gh pr create`.
Use the session's prior turns and the git diff. Do not invoke this workflow from
inside another worker; the verdict needs the orchestrator's view of the original
prompt.

Skip entirely when:

- the change is a one-line typo, formatting-only fix, lockfile bump, or comment/docstring-only edit;
- the change is a mass mechanical sweep the user asked for, such as ruff autofixes across the tree or a docs cross-reference sweep;
- the user explicitly broadened scope mid-task (`while you're at it, also fix X`); that addition is in scope by definition.

## How to run

1. Resolve `WORKFLOW_BASE` to the supplied value or `origin/main`.
2. Rebase before judging: run `git fetch origin main --quiet` and then `git rebase WORKFLOW_BASE`. If the rebase conflicts, run `git rebase --abort`, report the conflict, and stop. Do not judge or open a PR from a stale or conflicted branch.
3. Resolve the original task:
   - if `WORKFLOW_ORIGINAL_TASK` is supplied, treat it as authoritative;
   - otherwise infer it from the current thread, including only scope refinements the user explicitly approved.
4. Collect `git status --short`, `git log WORKFLOW_BASE..HEAD --oneline`, and `git diff WORKFLOW_BASE...HEAD`.
5. Judge only scope alignment. Do not do a general code review; deterministic hooks and `WORKFLOW_REVIEW_TOOL` cover code quality.
6. If `WORKFLOW_SUBAGENTS` is available, delegate the diff-vs-intent read to one worker and ask for a verdict under 200 words. Otherwise perform the same check directly.

Worker brief when delegation is available:

1. Quote the original task verbatim. Add a short note on any mid-session scope refinements explicitly approved by the user. Do not paraphrase favorably.
2. Include `git diff WORKFLOW_BASE...HEAD` plus `git status --short`.
3. Include `git log WORKFLOW_BASE..HEAD --oneline`.
4. Tell the worker: `Return a verdict under 200 words. You are looking for scope creep: files or changes that landed but were not part of the asked-for task. You are not reviewing code quality; deterministic checks already passed.`

## Output shape

Return exactly this shape:

**Verdict**: PASS or WARN

**In-scope**:
- One bullet per logical change that matches the task.

**Out-of-scope**:
- One bullet per change that was not requested, with file path and reason.

**Missing**:
- Any task-implied change absent from the diff.

**Recommendation**: open as-is / mention drift in the PR description / split before PR

Never return VETO. The judge advises; the human decides.

## How to act on the verdict

- **PASS**: report a one-line summary such as `pre-pr-judge: PASS`, then proceed with `gh pr create`.
- **WARN**: surface the full report to the user and ask whether to open as-is, mention the drift in the PR description, or split. Do not run `gh pr create` until the user picks.

The judge never blocks mechanically. It makes scope drift visible to the human
before a PR opens.

## What this catches

Concrete project examples from `todo/fixed-archive.md`:

- Shared-venv CI optimization (#110/#111): fixing one CI issue, the agent also restructured venv handling. The restructure was the drift.
- Gunicorn `--preload` pre-warm (#148/#149): fixing cold-start latency, the agent added pre-warm at module import that broke ALB health checks. The pre-warm location was the drift.

Both passed lint, tests, and benchmarks. Both were drift the human would have
caught if diff-vs-intent had been surfaced before the PR opened.

## Format example

```markdown
**Verdict**: WARN

**In-scope**:
- src/qb/config.py:42 — added `NN_DROPOUT` knob (asked-for change)
- tests/qb/test_config.py — covers the new knob

**Out-of-scope**:
- src/rb/config.py:38 — also added `NN_DROPOUT` to RB. User asked about QB only.
- src/wr/features.py — refactored an unrelated helper to use a list comprehension.

**Missing**:
- ATTN_STATIC_FEATURES not updated. AGENTS.md says adding to INCLUDE_FEATURES alone does not feed the attention branch.

**Recommendation**: Split before PR.
```
