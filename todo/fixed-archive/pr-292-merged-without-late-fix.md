### [FIXED] PR #292 merged before its userdata ordering fix was committed

- **File(s):** `infra/batch/userdata.sh` (historical; deleted by #953,
  `0ed3a9ff`). #292 squash `66f98464`; the fix was restored by #295
  (`d1b74576`) on 2026-05-21.
- **What:** #292 (SOCI lazy-loading) squash-merged at 07:49:12 UTC with its
  single PR commit `76773ddd`. The snapshotter-before-containerd ordering fix
  `e59aa1a9`, whose parent is that PR commit, was committed at 07:49:39 UTC, 27
  seconds after the merge, so it never reached the PR head or `origin/main`.
  #295's title describes the fix as lost in the squash; the commit timestamps
  show a commit-after-merge race rather than a squash defect. Infrastructure
  applied from the local worktree already contained the fix, so live state and
  `main` diverged until #295.
- **Fix:** #295 restored the ordering fix 36 minutes later, alongside the MIME
  UserData change. The standing rule, to finish committing before merging and
  then inspect the squash content, is in the
  [delivery gates](../../agent-guides/delivery.md#pr-and-merge-gates).
- **Lesson:** Green checks describe only the head they ran on. A fix committed
  around merge time can miss the squash, and infrastructure applied from local
  files can hide the gap until the repository is checked directly.
