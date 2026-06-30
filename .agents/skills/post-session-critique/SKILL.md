---
name: post-session-critique
description: Reflect on the current Codex or Gemini session and propose AGENTS.md, provider-doc, or memory updates that would have prevented a wrong turn or sped up the right call. Use only after a non-routine session.
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
- `WORKFLOW_WRITE_MEMORY=1 only when invoked with WRITE_MEMORY=1; otherwise propose memory text without writing`

Gemini runtime values:

- `WORKFLOW_PROVIDER=Gemini CLI`
- `WORKFLOW_ENTRYPOINT=activate_skill(name="post-session-critique")`
- `WORKFLOW_WRAPPER=.agents/skills/post-session-critique/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/post-session-critique/instructions.md`
- `WORKFLOW_AGENT_DOC=GEMINI.md`
- `WORKFLOW_MEMORY_DESTINATION=~/.gemini/tmp/<project>/memory/MEMORY.md (Gemini's Markdown memory; durable cross-agent lessons go to AGENTS.md)`
- `WORKFLOW_WRITE_MEMORY=1 for worthwhile Gemini memory notes after duplicate checks`

Execution:

1. Read `agent-workflows/post-session-critique/instructions.md`.
2. If it is missing or empty, STOP NOW: do not write memory and do not improvise a critique workflow.
3. Otherwise, execute that file to completion using the runtime values for the active provider.
