@AGENTS.md

# Claude Code

Use [the Claude reference](agent-guides/providers/claude.md) before changing
hooks, running the pre/post-PR workflow, delegating remediation, deploying audit
routines, or maintaining Claude memory. Its sections preserve the provider's
execution and approval contracts; shared workflow behavior lives in
`agent-workflows/` and `routines/`.

- `.claude/settings.json` wires the worktree guard, formatter, PR gates and
  memory hooks. SessionStart links worktree data; remote sessions also bootstrap
  the environment. Local sessions use SETUP.md for environment setup.
- Invoke `pre-pr-judge` before a non-trivial PR and follow the post-PR review/CI
  workflow. `solve-issues` PRs require explicit owner merge sign-off.
- Read the sub-agent contract before delegation: large cleanups use file-disjoint
  draft worker commits and one PR per risk tier, integrated by the orchestrator.
- After a non-routine session use `post-session-critique` when applicable. Check
  for existing guidance and update the relevant topic instead of appending a copy
  to AGENTS.md. See [context maintenance](agent-guides/context-maintenance.md).
- Claude's memory index is generated from topic `index_line` frontmatter and is
  excluded from S3 sync. Preserve those fields; do not hand-maintain the index.
