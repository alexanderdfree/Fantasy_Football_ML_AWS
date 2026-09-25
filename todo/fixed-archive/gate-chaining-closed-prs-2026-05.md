### [FIXED] Chaining a gate with its dependent action shipped a failing test and closed PRs #622 and #627

- **File(s):** [delivery gates](../../agent-guides/delivery.md#pr-and-merge-gates)
  (process rule; no code change). #622 merged as `974a8d9d` and #630 as
  `1e400f1d` (2026-05-31 UTC); #1122 merged as `1687bcb9` (2026-06-11 UTC).
- **What:** Agent sessions issued a verification step and the irreversible
  action that depended on it in the same tool batch. During #622 a `pytest` run
  with one failure shared a batch with `git commit && git push`, so the failing
  test reached CI. `gh pr merge 622 --squash` then failed under branch
  protection, a trailing `echo` masked its exit status, and the chained
  `git push origin --delete` closed the PR. Minutes later a `gh pr checks --watch`
  and a branch deletion issued in the same batch repeated the pattern: the watch
  returned before the reopened PR's checks had registered, and the deletion
  closed the PR again. On #627 the merge failed on a TODO.md conflict and the
  chained deletion closed the PR. A later session guessed a PR number and nearly
  ran `gh pr merge --admin` against unrelated #588 before the call was
  cancelled. On #1122 the pre-PR hook blocked a single
  `git add && git commit && git push && gh pr create` invocation as a whole, so
  the commit never ran and a later bare `gh pr create` opened the PR without it.
- **Fix:** #622 was reopened and merged once its test was fixed; #630 replaced
  #627. The standing rules are in the delivery gates.
- **Lesson:** Deleting the head branch of an unmerged PR closes it, and a
  masked exit status turns a failed gate into a destructive step. Keep every
  irreversible action a separate call after the evidence it depends on.
