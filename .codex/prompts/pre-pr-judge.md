---
description: Vet the current branch against the original task before opening a PR
argument-hint: [ORIGINAL_TASK="<quoted task>"] [BASE=origin/main]
---

Run the Final-Project pre-PR scope judge.

Use `BASE` if supplied; otherwise use `origin/main`. If `ORIGINAL_TASK` is supplied, treat it as authoritative. Otherwise infer the original task from the current thread, including only scope refinements the user explicitly approved.

Steps:

1. Resolve `BASE` to the supplied value or `origin/main`.
2. Rebase before judging: run `git fetch origin main --quiet` and then `git rebase BASE`. If the rebase conflicts, run `git rebase --abort`, report the conflict, and stop. Do not judge or open a PR from a stale or conflicted branch.
3. Collect `git status --short`, `git log BASE..HEAD --oneline`, and `git diff BASE...HEAD`.
4. Judge only scope alignment. Do not do a general code review; the deterministic hooks and `codex review` cover code quality.
5. If Codex subagents are available, delegate the diff-vs-intent read to one subagent and ask for a verdict under 200 words. Otherwise perform the same check directly.

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
