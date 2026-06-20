# Shared post-session critique instructions

These are the provider-neutral instructions for capturing a non-routine session
lesson. Provider wrappers must read this file, set the runtime values below,
apply their own memory/write rules, and then execute the shared workflow.

## Runtime contract

The wrapper must define these values before executing the workflow:

- `WORKFLOW_PROVIDER`: human-readable provider name, for example `Claude Code` or `Codex`.
- `WORKFLOW_ENTRYPOINT`: the user-facing invocation, for example `post-session-critique` or `/prompts:post-session-critique`.
- `WORKFLOW_WRAPPER`: the wrapper file that loaded this instruction file.
- `WORKFLOW_SHARED_INSTRUCTIONS`: `agent-workflows/post-session-critique/instructions.md`.
- `WORKFLOW_AGENT_DOC`: provider-specific repo doc to consider for provider-only rules, for example `CLAUDE.md`, `CODEX.md`, or `GEMINI.md`.
- `WORKFLOW_MEMORY_DESTINATION`: provider memory location and write policy.
- `WORKFLOW_WRITE_MEMORY`: whether this invocation may write provider memory automatically.

Provider wrappers own their local memory mechanics. Do not let provider mechanics
change the skip rule, the under-200-word output shape, or the rule that durable
cross-agent lessons belong in `AGENTS.md`.

## Provider entrypoints

- Claude wrapper: `.claude/skills/post-session-critique/SKILL.md`.
- Codex wrapper: `.codex/prompts/post-session-critique.md`.
- Gemini/Antigravity wrapper: `.agents/skills/post-session-critique/SKILL.md`.

The wrappers stay discoverable at those paths. This file is the behavioral source
of truth.

## Mission

After a non-routine agent session, capture the prompt or process lesson that
would have prevented a wrong turn or sped up the right call. The `todo/fixed-archive.md`
fixed archive captures code lessons; this workflow captures instruction and
memory lessons.

## When to run

Run after a session where:

- the user corrected the agent's approach mid-flight;
- a non-obvious project convention changed the right move, such as feature whitelist, loss-weight rebalance, training/inference drift, worktree path, or stop-rule behavior;
- a missing precondition caused the agent to act when it should not have, or hesitate when it should have acted;
- the session went unusually well because a specific rule or memory worked and should be preserved.

Skip with one sentence if the session was routine. Thin archive is not a failure;
over-documentation is.

## Before proposing anything

Check for duplicates first:

- read `AGENTS.md` for durable cross-agent guidance;
- read `WORKFLOW_AGENT_DOC` for provider-specific machinery;
- check the provider memory summary or memory index when available.

Do not propose a new rule if the same rule already exists in those places.

## What to produce

Produce a reflection under 200 words with these sections:

**What happened**: one sentence on the wrong turn, missing precondition, or notable moment.

**What was missing**: the missing rule, example, setup note, or memory cue that would have changed behavior earlier.

**Proposed change**: one concrete proposal:

- a markdown snippet for `AGENTS.md` when the lesson is durable and cross-agent;
- a markdown snippet for `WORKFLOW_AGENT_DOC` when the lesson is provider-specific;
- a provider memory note when the lesson is local recall rather than repo source of truth.

Durable cross-agent lessons belong in `AGENTS.md`. Provider-only execution,
hook, prompt, or harness details belong in `WORKFLOW_AGENT_DOC` or provider
memory.

## Memory write policy

- Claude wrapper: memory-worthy notes may be written under the provider's Claude project memory and indexed there when the skill has already filtered for a worthwhile lesson.
- Codex wrapper: write memory only when `WORKFLOW_WRITE_MEMORY=1`. Otherwise propose the memory text without writing it. Codex memory notes go under `$CODEX_HOME/memories/extensions/ad_hoc/notes/`, falling back to `~/.codex/memories/extensions/ad_hoc/notes/` when `CODEX_HOME` is unset, and must follow the active Codex memory rules.
- Gemini/Antigravity wrapper: memory is plain Markdown under `~/.gemini/` (project memory under `~/.gemini/tmp/<project>/memory/`); it is not authoritative. Propose the note and reserve durable cross-agent lessons for `AGENTS.md`.

## What to skip

- Generic observations such as `could have been clearer`.
- Duplicates of already documented rules.
- Code-level lessons that belong in `todo/fixed-archive.md` through the normal PR workflow.
- Session summaries that are only status reports.

## Format example

```markdown
**What happened**: I proposed a shared-venv CI optimization; user reminded me it was tried and reverted.

**What was missing**: The reverted optimization was memory-only and did not load for this CI task.

**Proposed change**: Add this durable cross-agent rule to AGENTS.md `Stop rules`: `Shared-venv CI optimization was reverted in #110/#111; artifact download was slower than warm uv install, so wall-clock wins over compute.`
```
