Compatibility wrapper for the Claude audit routine.

This path intentionally remains tracked because an already-deployed Claude
dashboard shim may still read `.claude/routines/audit/prompt.md`. The shared
source of truth has moved to `routines/audit/instructions.md`; do not add shared
audit logic here.

Claude runtime values:

- `AUDIT_PROVIDER=Claude`
- `AUDIT_LABEL=claude-audit`
- `DEDUPE_AUDIT_LABELS="claude-audit codex-audit"`

Claude-only runtime/tool rules:

- Use Claude's configured cloud routine tools and repository checkout.
- For the fanout step, Claude Agent subagents are available. Worker count,
  scope grouping, and batching are up to you — a worker may cover several
  scopes; avoid over-spawning.
- Keep the run read-only on repo files. The only permitted writes are those
  allowed by `routines/audit/instructions.md`: `/tmp/*`, idempotent label
  creation, per-finding issue creation, comments on issues created in this run,
  and closing the checkpoint created in this run.

Execution:

1. Read `routines/audit/instructions.md`.
2. If it is missing or empty, STOP NOW: do not create any GitHub issue and do
   not improvise an audit. Exit so the failure is visible in the run log.
3. Otherwise, execute that file to completion using the runtime values and
   Claude-only rules above.
