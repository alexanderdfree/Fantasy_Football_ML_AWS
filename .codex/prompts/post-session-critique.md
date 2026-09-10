---
description: Propose a scoped guidance or authorized memory update after a non-routine session
argument-hint: '[WRITE_MEMORY=0|1]'
---

Run the repo's Codex post-session critique wrapper.

The authoritative instructions are version-controlled at `agent-workflows/post-session-critique/instructions.md`. That file, not this wrapper, defines when to run, when to skip, the reflection shape, and where durable lessons belong.

Codex runtime values:

- `WORKFLOW_PROVIDER=Codex`
- `WORKFLOW_ENTRYPOINT=/prompts:post-session-critique`
- `WORKFLOW_WRAPPER=.codex/prompts/post-session-critique.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/post-session-critique/instructions.md`
- `WORKFLOW_AGENT_DOC=CODEX.md`
- `WORKFLOW_MEMORY_DESTINATION=$CODEX_HOME/memories/extensions/ad_hoc/notes/` with fallback to `~/.codex/memories/extensions/ad_hoc/notes/`
- `WORKFLOW_WRITE_MEMORY=1 only when the user explicitly authorized saving memory in this session (including WRITE_MEMORY=1); otherwise propose memory text without writing. An explicit no-write instruction keeps this at 0.`

Execution:

1. Read `agent-workflows/post-session-critique/instructions.md`.
2. If it is missing or empty, STOP NOW: do not write memory and do not improvise a critique workflow.
3. Otherwise, execute that file to completion using the Codex runtime values and supplied arguments above.
