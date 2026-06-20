---
name: post-session-critique
description: Reflect on the current Gemini CLI session and propose AGENTS.md, GEMINI.md, or memory updates that would have prevented a wrong turn or sped up the right call. Use after a session where the user had to correct you, a non-obvious project convention bit you, or something went unusually well because of a specific rule.
---

# Post-session critique wrapper

This is the Gemini CLI wrapper for the shared post-session critique workflow.

The authoritative instructions are version-controlled at `agent-workflows/post-session-critique/instructions.md`. That file, not this wrapper, defines when to run, when to skip, the reflection shape, and where durable lessons belong.

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
3. Otherwise, execute that file to completion using the Gemini runtime values above.
