---
description: Capture a non-routine Codex session lesson for AGENTS.md or memory
argument-hint: [WRITE_MEMORY=0|1]
---

Reflect on the current Codex session only if it was non-routine: the user corrected the approach, a project convention changed the right move, a stop-rule prevented wasted work, or a specific instruction made the session unusually efficient.

Skip with one sentence if the session was routine.

Before proposing anything, check `AGENTS.md`, `CLAUDE.md`, and the current Codex memory summary for duplicates.

Produce under 200 words:

**What happened**: one sentence.

**What was missing**: the missing rule, example, or setup note.

**Proposed change**: either a concrete AGENTS.md/CODEX.md snippet or a Codex memory note.

Durable cross-agent lessons belong in `AGENTS.md`. Codex-only local recall may go to memory only when this command is invoked with `WRITE_MEMORY=1`; otherwise propose the memory text without writing it. If writing memory, follow the active Codex memory rules and put the note under `/Users/alex/.codex/memories/extensions/ad_hoc/notes/`.
