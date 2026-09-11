### [FIXED] Post-PR hook revived a merge bypass and omitted deletion verification

- **File(s):** `.claude/hooks/post-pr-create.sh`,
  `tests/scripts/test_claude_hooks.py`; baseline `586a08da` (PR #1557).
- **What:** The ordinary-branch hook still emitted a manual `--admin` merge
  fallback for the historical silent-stop CI anomaly, contradicting the shared
  no-bypass rule. Both branch workflows scheduled remote deletion without an
  explicit check of the merged state and latest squash content. Old CLAUDE.md
  section references also bypassed the new topic routes.
- **Fix:** Retain the narrow user-confirmed local-test exception without any
  branch-protection or unrelated-check bypass. After an authorized squash merge,
  require MERGED state, fetch and inspect the latest fixes, then delete the
  remote branch separately. Preserve architectural-review stops and audit-tier
  sign-off. Tests execute the hook on isolated ordinary/audit branches and inspect
  the emitted commands and gate ordering.
- **Lesson:** Consolidating prose is incomplete if a hook still injects a copied,
  contradictory workflow. Check emitted instructions against the authoritative
  [merge gates](../../agent-guides/delivery.md#pr-and-merge-gates), including
  exceptional paths and branch-specific approval rules.
