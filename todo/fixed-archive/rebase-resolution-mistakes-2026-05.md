### [FIXED] Rebase resolution committed conflict markers and rebased the wrong branch

- **File(s):** [worktree workflow](../../agent-guides/delivery.md#worktree-workflow)
  (process rule; no code change). Branches `claude/tune-nn-kdst-and-adr`
  (2026-05-21) and `claude/min-games-per-position` for #656 after #658
  (2026-05-31).
- **What:** On 2026-05-21 a conflict-resolution edit to `docs/ARCHITECTURE.md`
  reported success but did not apply because a linter changed the file first;
  `git add` and `git rebase --continue` then committed literal conflict
  markers, which the next commit removed. On 2026-05-31
  `git checkout claude/min-games-per-position` failed because another worktree
  held that branch, but the `git rebase origin/main` on the next line still ran
  on the just-merged ADR-split branch and produced add/add conflicts in 17
  `docs/adr/*.md` files. `git rebase --abort` recovered it.
- **Fix:** Check `git diff --check` or grep for markers before staging a
  resolution. Confirm a checkout with `git branch --show-current`; rebase a
  branch held by another worktree there (`git -C <path>`) or from a detached
  `origin/<branch>`. To keep `origin/main`'s version of a conflicted file, run
  `git checkout origin/main -- <file>`: `--ours` and `--theirs` are swapped
  during a rebase.
- **Lesson:** Commands on separate lines of one batch run independently, and a
  tool's success message is not the artifact. Verify the branch and file
  content before the step that depends on them.
