---
description: Vet the current branch against the original task before opening a PR
argument-hint: [ORIGINAL_TASK="<quoted task>"] [BASE=origin/main]
---

Run the Final-Project pre-PR scope judge.

Use `BASE` if supplied; otherwise use `origin/main`. If `ORIGINAL_TASK` is supplied, treat it as authoritative. Otherwise infer the original task from the current thread, including only scope refinements the user explicitly approved.

Steps:

1. Ensure the branch is current enough for a fair diff: run `git fetch origin main --quiet`. If the branch has not been rebased recently, report that and recommend rebasing before the final judge.
2. Collect `git status --short`, `git log BASE..HEAD --oneline`, and `git diff BASE...HEAD`.
3. Judge only scope alignment. Do not do a general code review; the deterministic hooks and `codex review` cover code quality.
4. If Codex subagents are available, delegate the diff-vs-intent read to one subagent and ask for a verdict under 200 words. Otherwise perform the same check directly.

Return this shape:

**Verdict**: PASS or WARN

**In-scope**:
- One bullet per logical change that matches the task.

**Out-of-scope**:
- One bullet per change that was not requested, with file path and reason.

**Missing**:
- Any task-implied change absent from the diff.

**Recommendation**: open as-is / mention drift in the PR description / split before PR

If the verdict is WARN, stop and ask the user how to proceed before running `gh pr create`.
