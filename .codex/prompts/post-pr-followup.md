---
description: Continue after opening a PR with review, CI, and merge discipline
argument-hint: '[PR=<number>] [BASE=origin/main]'
---

Run the repo's Codex post-PR workflow.

Resolve `PR` from `gh pr view --json number,url` if not supplied. Resolve and hold the PR head branch with `gh pr view PR --json headRefName --jq .headRefName` before merging. If the current branch is not that PR head branch, switch to it first; if Git reports that branch is checked out in another worktree, stop and rerun from that worktree. Use `BASE` if supplied; otherwise use `origin/main`.

1. Refresh `origin/main` and any different remote ref supplied as `BASE`, then rebase the verified PR checkout onto that resolved base. Resolve routine conflicts within the authorized scope; surface only decisions that cannot be inferred. Re-run affected checks after the tested tree changes, and push rewritten history with `git push --force-with-lease` only after resolution and validation.
2. Run `scripts/codex-review-quiet.sh --base BASE` from the repo root. This preserves review findings while filtering known Codex/plugin loader stderr noise from the chat context.
3. Apply localized review findings: nits, style, naming, dead imports, simple logic fixes, localized security fixes, missing small tests, doc/comment tweaks. Do not auto-apply architectural or design-level findings; list those for the user (these gate the merge — see step 6).
4. If fixes were applied, commit with subject `review: address codex review nits` and push.
5. Wait for current green CI with `gh pr checks PR --watch`. Investigate failures and fix them within the authorized scope; report any unresolved blocker or decision that requires the user. Do not merge with red/pending checks or use `--admin`. The documented `Run Tests` silent-stop exception in `agent-guides/operations.md` requires local validation before an otherwise-authorized merge; it does not waive branch protection or the explicit owner sign-off below.
6. **Architectural-findings gate (all branches):** if step 3 surfaced any architectural/design-level finding, STOP — summarize each in one line for the user and do **NOT** merge. The user decides whether to address them.
7. Merge once CI is green and the step-6 gate is clear:
   - **`audit-*/tier-*` branches:** show `gh pr diff PR` plus any `regress-risk-high` benchmark deltas, and ask for **explicit merge sign-off**. After approval, run `gh pr merge PR --squash`; verify MERGED state and the latest reviewed squash content before separately running `git push origin --delete <headRefName>`.
   - **Other branches:** `gh pr merge PR --squash`. Verify MERGED state and latest squash content before `git push origin --delete <headRefName>`. If branch protection still blocks merging after the documented fallback, surface that gate; do not use `--admin`. Do not use `gh pr merge --delete-branch` from a worktree.
8. After a successful merge, if this session had a non-routine moment — the user corrected your approach mid-flight, a stop-rule bit you, or something went unusually well because of a specific rule — invoke the `post-session-critique` workflow to capture the prompt lesson. Skip if the session was routine (do not run it just because this workflow ran).

End with a concise status: review fixes, surfaced issues, CI result, merge result.
