### [FIXED] Chaining a gate with its dependent action shipped a failing test and closed PRs #622 and #627

- **File(s):** [delivery gates](../../agent-guides/delivery.md#pr-and-merge-gates)
  (process rule; no code change). #622 merged as `974a8d9d` and #630 as
  `1e400f1d` (2026-05-31); #1122 (2026-06-11).
- **What:** Agent sessions put a verification step and the irreversible action
  that depended on it in one tool batch. During #622 a `pytest` run with one
  failure shared a batch with `git commit && git push`, so the failing test
  reached CI. `gh pr merge 622 --squash` then failed under branch protection, a
  trailing `echo` masked its exit status, and the chained
  `git push origin --delete` closed the PR. Minutes later
  `gh pr checks --watch && git push origin --delete` repeated the pattern: the
  watch returned before the reopened PR's checks had registered. On #627 the
  merge failed on a TODO.md conflict and the chained deletion closed the PR; it
  was superseded by #630 rather than reopened. A later session guessed a PR
  number and nearly ran `gh pr merge --admin` against unrelated #588 before the
  call was cancelled. On #1122 the pre-PR hook blocked a single
  `git add && git commit && git push && gh pr create` invocation as a whole, so
  the commit never ran and a later bare `gh pr create` opened the PR without it.
- **Fix:** Run each gate alone and read its result before issuing the dependent
  action. Confirm `gh pr view <N> --json state` reports `MERGED` before deleting
  a branch. Treat a `--watch` that returns immediately after a push or reopen
  as checks not yet registered. Derive the PR number from the current branch.
  Run hook-gated commands in their own call. #622 was reopened and merged once
  its test was fixed.
- **Lesson:** Deleting the head branch of an unmerged PR closes it, and a
  masked exit status turns a failed gate into a destructive step. Keep every
  irreversible action a separate call after the evidence it depends on.
