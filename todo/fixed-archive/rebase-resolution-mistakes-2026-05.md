### [FIXED] Rebase resolution committed conflict markers and rebased the wrong branch

- **File(s):** [worktree workflow](../../agent-guides/delivery.md#worktree-workflow)
  (process rule; no code change). 2026-05-21 UTC: branch
  `claude/tune-nn-kdst-and-adr`, where branch commit `ca94d354` carried the
  markers and `03661630` removed them. 2026-05-31: `claude/min-games-per-position`
  for #656 (`f749e538`), rebased after #658 (`79fbba4c`).
- **What:** On 2026-05-21 a conflict-resolution edit to `docs/ARCHITECTURE.md`
  reported success but did not apply because a linter changed the file first;
  `git add` and `git rebase --continue` then committed literal conflict markers,
  which a later commit on the branch removed. On 2026-05-31
  `git checkout claude/min-games-per-position` failed because another worktree
  held that branch, but the `git rebase origin/main` on the next line still ran
  on the just-merged ADR-split branch and produced add/add conflicts across the
  newly split `docs/adr/` files.
- **Fix:** `03661630` removed the markers, and `git rebase --abort` undid the
  wrong-branch rebase. The standing rules are in the worktree workflow.
- **Lesson:** Commands on separate lines of one batch run independently, and a
  tool's success message is not the artifact. Verify the branch and file
  content before the step that depends on them.
