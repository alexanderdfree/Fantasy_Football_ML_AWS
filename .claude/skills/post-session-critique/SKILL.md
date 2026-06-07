---
name: post-session-critique
description: Reflect on the current Claude Code session and propose AGENTS.md, CLAUDE.md, or memory updates that would have prevented a wrong turn or sped up the right call. Use after a session where the user had to correct you, a non-obvious project convention bit you, or something went unusually well because of a specific rule.
---

# Post-session critique

Inspired by Spotify's Honk Part 2: *"After a session, the agent itself is in a surprisingly good position to tell you what was missing."* The [todo/fixed-archive.md](todo/fixed-archive.md) "Fixed archive" already captures **code** lessons; this captures **prompt** lessons.

## When to run

After a session where:
- The user corrected your approach mid-flight.
- A non-obvious convention bit you (e.g., feature whitelist, loss-weight rebalance, training/inference path drift).
- A precondition was missing — you acted when you shouldn't have, or hesitated when you should have acted.
- Conversely: the session went unusually well because of something specific in CLAUDE.md or memory — capture *that this works* so it doesn't get lost.

Skip if the session was routine. Thin archive ≠ failure; over-documentation is.

## What to produce

A reflection under 200 words with these sections:

1. **What happened** — one sentence on the wrong turn or notable moment.
2. **What was missing** — what would have changed your behavior earlier? A new rule in AGENTS.md (durable, cross-agent) or CLAUDE.md (Claude-specific)? A memory entry? An example in an existing section?
3. **Proposed change** — a concrete diff. Either:
   - A markdown snippet to add/edit in the right doc, naming the section — **[AGENTS.md](AGENTS.md)** for a durable cross-agent lesson (ML method, git/PR/CI/env workflow, project fact; the cross-agent source of truth), **[CLAUDE.md](CLAUDE.md)** only for Claude-Code-specific machinery (hooks, skills, harness). Claude-internal harness quirks stay in auto-memory. OR
   - A draft memory file (frontmatter + body) following the conventions of the auto-memory system at `~/.claude/projects/<project-slug>/memory/` (the slug is auto-derived from the project's absolute path on each contributor's machine, so the exact directory differs per user). Use `feedback_*` for prompt-style corrections, `project_*` for state/context, `user_*` for user preferences. Include the one-line entry to add to that directory's `MEMORY.md`.

- **Memory-entry proposals** — auto-write the file under `~/.claude/projects/<project-slug>/memory/` and update that directory's `MEMORY.md` index. Surface a one-line summary of what you wrote; no need to ask permission. The skill has already filtered for "worth capturing"; gating each write adds friction and discourages capture.
- **AGENTS.md / CLAUDE.md edit proposals** — surface the diff in the conversation and let the user decide. They live in the project repo, need a commit/PR, and the user may want to tune wording before it lands in repo history.

## What to skip

- Generic observations ("could have been clearer", "more examples would help").
- Things already documented — `Read` [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md) and the project's auto-memory `MEMORY.md` (at `~/.claude/projects/<slug>/memory/MEMORY.md`) first to check for duplicates.
- Code-level lessons that belong in [todo/fixed-archive.md](todo/fixed-archive.md) — those go through the normal PR workflow, not here.

## Format example

```
**What happened**: I proposed a shared-venv CI optimization; user reminded me it was tried and reverted (PRs #110/#111).

**What was missing**: CLAUDE.md "Conventions that bite" doesn't list reverted optimizations. Memory has it (feedback_shared_venv_ci.md), but I didn't load it because the session topic was unrelated.

**Proposed change**: graduate the shared-venv rule from memory into AGENTS.md's "Stop rules" subsection (a durable cross-agent lesson) so it loads unconditionally. Diff:

    ### Stop rules — things that have been tried and reverted
    - Shared-venv CI optimization (reverted PRs #110/#111). Artifact download (25s/shard) > warm uv install (~10s). Wall-clock, not compute, is the metric.
```
