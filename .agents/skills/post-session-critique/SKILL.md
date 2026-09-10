---
name: post-session-critique
description: Reflect on a non-routine Codex or Gemini session and propose a targeted update to existing project guidance, provider references, or authorized memory. Check duplicates and context budgets before adding a rule.
---

# Post-session critique wrapper

This is the shared Codex/Gemini skill wrapper for the post-session critique workflow.

The authoritative instructions are version-controlled at `agent-workflows/post-session-critique/instructions.md`. That file, not this wrapper, defines when to run, when to skip, the reflection shape, and where durable lessons belong.

Codex runtime values:

- `WORKFLOW_PROVIDER=Codex`
- `WORKFLOW_ENTRYPOINT=$post-session-critique` or implicit skill invocation; `/prompts:post-session-critique` is the legacy prompt alias
- `WORKFLOW_WRAPPER=.agents/skills/post-session-critique/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/post-session-critique/instructions.md`
- `WORKFLOW_AGENT_DOC=CODEX.md`
- `WORKFLOW_MEMORY_DESTINATION=$CODEX_HOME/memories/extensions/ad_hoc/notes/` with fallback to `~/.codex/memories/extensions/ad_hoc/notes/`
- `WORKFLOW_WRITE_MEMORY=1 only when the user explicitly authorized saving memory in this session (including WRITE_MEMORY=1); otherwise propose memory text without writing. An explicit no-write instruction keeps this at 0.`

Gemini runtime values:

- `WORKFLOW_PROVIDER=Gemini CLI`
- `WORKFLOW_ENTRYPOINT=activate_skill(name="post-session-critique")`
- `WORKFLOW_WRAPPER=.agents/skills/post-session-critique/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/post-session-critique/instructions.md`
- `WORKFLOW_AGENT_DOC=GEMINI.md`
- `WORKFLOW_MEMORY_DESTINATION=~/.gemini/tmp/<project>/memory/MEMORY.md (Gemini's Markdown memory; durable shared lessons go to the relevant agent-guides topic or ADR)`
- `WORKFLOW_WRITE_MEMORY=0 — Gemini memory is plain Markdown and not authoritative, so propose the memory text without auto-writing; reserve durable shared lessons for the relevant agent-guides topic or ADR`

Execution:

1. Read `agent-workflows/post-session-critique/instructions.md`.
2. If it is missing or empty, STOP NOW: do not write memory and do not improvise a critique workflow.
3. Otherwise, execute that file to completion using the runtime values for the active provider.
