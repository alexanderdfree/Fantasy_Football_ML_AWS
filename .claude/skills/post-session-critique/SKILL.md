---
name: post-session-critique
description: Reflect on the current Claude Code session and propose AGENTS.md, CLAUDE.md, or memory updates that would have prevented a wrong turn or sped up the right call. Use after a session where the user had to correct you, a non-obvious project convention bit you, or something went unusually well because of a specific rule.
---

# Post-session critique wrapper

This is the Claude Code wrapper for the shared post-session critique workflow.

The authoritative instructions are version-controlled at `agent-workflows/post-session-critique/instructions.md`. That file, not this wrapper, defines when to run, when to skip, the reflection shape, and where durable lessons belong.

Claude runtime values:

- `WORKFLOW_PROVIDER=Claude Code`
- `WORKFLOW_ENTRYPOINT=post-session-critique`
- `WORKFLOW_WRAPPER=.claude/skills/post-session-critique/SKILL.md`
- `WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/post-session-critique/instructions.md`
- `WORKFLOW_AGENT_DOC=CLAUDE.md`
- `WORKFLOW_MEMORY_DESTINATION=~/.claude/projects/<project-slug>/memory/`
- `WORKFLOW_WRITE_MEMORY=1 for worthwhile Claude memory notes after duplicate checks`

Execution:

1. Read `agent-workflows/post-session-critique/instructions.md`.
2. If it is missing or empty, STOP NOW: do not write memory and do not improvise a critique workflow.
3. Otherwise, execute that file to completion using the Claude runtime values above.
