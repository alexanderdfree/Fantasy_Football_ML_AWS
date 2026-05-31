---
description: Continue after opening a PR with review, CI, and merge discipline
argument-hint: [PR=<number>] [BASE=origin/main]
---

Run the Final-Project Codex post-PR workflow.

Resolve `PR` from `gh pr view --json number,url` if not supplied. Use `BASE` if supplied; otherwise use `origin/main`.

1. Rebase onto latest main: `git fetch origin main --quiet` then `git rebase origin/main`. If it changes the branch, push with `git push --force-with-lease`. If it conflicts, abort and surface the conflict to the user.
2. Run `codex review --base BASE` from the repo root.
3. Apply localized review findings: nits, style, naming, dead imports, simple logic fixes, localized security fixes, missing small tests, doc/comment tweaks. Do not auto-apply architectural or design-level findings; list those for the user.
4. If fixes were applied, commit with subject `review: address codex review nits` and push.
5. Wait for green CI with `gh pr checks PR --watch`. If CI fails, stop and report the failing check.
6. For `audit-*/tier-*` branches, show `gh pr diff PR` and ask for explicit merge sign-off before merging.
7. For other branches with green CI and no surfaced design findings, merge with `gh pr merge PR --squash`, then delete the remote branch with `git push origin --delete <branch>`. Do not use `gh pr merge --delete-branch` from a worktree.

End with a concise status: review fixes, surfaced issues, CI result, merge result.
