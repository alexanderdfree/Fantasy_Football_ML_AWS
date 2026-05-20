---
name: pre-pr-judge
description: Before `gh pr create`, vet the change against the original task — spawn a worker subagent that diffs the branch against `origin/main` and flags scope creep ("the agent did more than I asked"). Lightweight, agent-triggered companion to the user-triggered `/ultrareview`. Use after [.claude/hooks/pre-pr.sh](.claude/hooks/pre-pr.sh) passes but before opening the PR. Skip for trivial changes.
---

# Pre-PR judge

Inspired by Spotify's Honk Part 3: *"the judge vetoes ~25% of agent sessions; of those, the agent course-corrects ~50% of the time."* The deterministic gates in [.claude/hooks/pre-pr.sh](.claude/hooks/pre-pr.sh) catch what ruff, pytest, and benchmark freshness can express. This skill catches what they can't — *did the change match what was asked?*

`/ultrareview` is the heavyweight, user-triggered, multi-agent review that runs *after* the PR opens. This is the lightweight, agent-triggered, single-subagent check that runs *before*.

## When to run

After [.claude/hooks/pre-pr.sh](.claude/hooks/pre-pr.sh) passes, immediately before `gh pr create`. The session's prior turns and `git diff` are everything the judge needs — do not invoke this skill from inside another subagent (the verdict needs the orchestrator's view of the original prompt).

Skip when:
- The change is a one-line typo, formatting fix, lockfile bump, or comment-only edit.
- The change is a mass mechanical sweep (e.g. ruff fixes across the tree) — diff-against-intent doesn't yield signal.
- The user has explicitly said "while you're at it, also fix X" — that's in-scope by definition.

## How to run

Spawn one worker subagent (`Agent` tool, `subagent_type=general-purpose`). Brief it with:

1. **The original task, quoted verbatim.** Copy the user's first message in this session (or the most recent task hand-off), then add a short note on any mid-session scope refinements the user explicitly approved. Do not paraphrase favorably — the judge's value depends on seeing what was asked, not what you wish had been asked.
2. **The diff.** Hand over `git diff origin/main...HEAD` plus the output of `git status` (uncommitted work the orchestrator is about to include).
3. **The commit list.** `git log origin/main..HEAD --oneline`.
4. **The instruction.** "Return a verdict under 200 words. You are looking for *scope creep* — files or changes that landed but weren't part of the asked-for task. You are not reviewing code quality; deterministic checks already passed."

Ask for this output shape:

- **Verdict**: PASS or WARN. Never VETO — the judge advises, the human decides.
- **In-scope**: one bullet per logical change that matches the original task.
- **Out-of-scope**: one bullet per change that wasn't requested. Be specific about file/line and why it's drift.
- **Missing**: changes the task implies but the diff doesn't include.
- **Recommendation**: "open the PR as-is" / "open the PR, mention the drift in the PR description" / "split — open one PR for the asked-for change, revert the rest".

## How to act on the verdict

- **PASS** — one-line summary to the user (`pre-pr-judge: PASS`), then proceed with `gh pr create`. Do not expand the report.
- **WARN** — surface the full report to the user, then ask: open as-is, mention the drift in the PR description, or split. Do not run `gh pr create` until the user picks.

The judge never blocks on its own. It exists to make scope drift visible *to the human* before a PR opens, not to gate the PR mechanically. The deterministic gates already gate.

## What this catches that ruff/pytest don't

Concrete project examples from [TODO.md](TODO.md)'s Fixed archive (PRs that shipped, then reverted):
- **Shared-venv CI optimization** ([#110](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/110) / [#111](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/111)) — fixing one CI issue, the agent also restructured venv handling. The restructure was the drift.
- **Gunicorn `--preload` pre-warm** ([#148](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/148) / [#149](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/149)) — fixing cold-start latency, the agent added a pre-warm at module import that broke ALB health checks. The pre-warm location was the drift.

Both passed lint, tests, and benchmarks. Both were drift the human would have caught if the diff-vs-intent had been surfaced before the PR opened.

## Format example

```
**Verdict**: WARN

**In-scope**:
- src/qb/config.py:42 — added `NN_DROPOUT` knob (asked-for change)
- tests/qb/test_config.py — covers the new knob

**Out-of-scope**:
- src/rb/config.py:38 — also added `NN_DROPOUT` to RB. User asked about QB only.
- src/wr/features.py — refactored an unrelated helper to use a list comprehension.

**Missing**:
- ATTN_STATIC_FEATURES not updated. CLAUDE.md ("Attention static-feature whitelist is separate per position") says adding to INCLUDE_FEATURES alone doesn't feed the attention branch. If the dropout knob was supposed to affect the attention NN's static branch, this is a real gap; if it was scalar-only, ignore.

**Recommendation**: Split. Open the QB PR; revert the RB and WR changes and propose them separately if still desired.
```
