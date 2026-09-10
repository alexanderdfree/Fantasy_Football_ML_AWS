@AGENTS.md

# Codex

Use [the Codex reference](agent-guides/providers/codex.md) for hooks, worktree
startup, skills/prompts, audit wrappers and memory sync. Read only the relevant
section before using or changing that mechanism. Shared behavior lives in
`agent-workflows/`; wrappers supply provider-specific runtime values.

- Start through `scripts/codex-fresh-worktree.sh` when launching from the CLI.
  SessionStart can warn and add context, but cannot move an active session.
- Trust changed project hooks through `/hooks`. Hooks do not intercept every
  possible shell write; use `apply_patch` and verify the active checkout.
- Prefer `.agents/skills/` workflows (`pre-pr-judge`, `post-session-critique`,
  `solve-issues`); `.codex/prompts/` holds legacy templates installed into the
  user home by `scripts/bootstrap-codex-local.sh`.
- Read actual user/project configuration and runtime state for model, reasoning
  and compaction settings. Do not restore historical values from memory.
- Codex memory updates require explicit user authorization and the supported
  update-note path; do not edit generated memory summaries/indexes directly.
  Keep personal memory out of git. See [context maintenance](agent-guides/context-maintenance.md).
