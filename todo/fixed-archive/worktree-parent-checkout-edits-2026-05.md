### [FIXED] Worktree sessions wrote their edits into the parent checkout

- **File(s):** `.claude/hooks/guard-worktree-path.sh`, added by #382
  (`25e31247`, 2026-05-29). Affected work: #284 (2026-05-21) and #354, #370,
  #378 and #381 (2026-05-29).
- **What:** Plan files and Explore/Plan sub-agents reported parent-absolute
  paths. Using them verbatim for Edit/Write changed the parent checkout, which
  was on `main` and sometimes held unrelated uncommitted work, instead of the
  feature worktree. The worktree's `git status` stayed clean and benchmarks
  re-ran unchanged code, so every per-target MAE delta was exactly 0.0000; #284
  lost about 30 minutes of GPU benchmark time. Written guidance in both agent
  memory and the startup instructions failed to prevent five recurrences. A
  sixth, the same day, happened on a worktree rebased onto a `main` that
  predated the guard.
- **Fix:** The PreToolUse guard blocks Edit/Write/MultiEdit/NotebookEdit paths
  under the parent root but outside the active worktree and prints the
  corrected path. Recovery when an edit slips through: copy the file when the
  committed baselines match, or pipe `git -C <parent> diff -- <file>` into
  `git -C <worktree> apply`, then restore only your files in the parent.
- **Lesson:** Prose rules did not fire when paths were constructed; a
  deterministic check at write time did. Re-prefix sub-agent paths to the
  worktree and check its `git status` after the first edit, not after a batch.
