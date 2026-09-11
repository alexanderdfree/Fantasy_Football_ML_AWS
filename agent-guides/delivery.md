# Worktrees and delivery

## Worktree workflow

- Verify the active checkout and `git status` before editing. Resolve reported
  relative or parent-absolute paths inside this worktree, then verify the edit
  there. A clean status after an intended edit can mean the parent was changed.
  Provider guard hooks are described in [CLAUDE.md](../CLAUDE.md) and
  [CODEX.md](../CODEX.md).
- Codex startup uses [scripts/codex-fresh-worktree.sh](../scripts/codex-fresh-worktree.sh).
  A `SessionStart` hook can warn but cannot move cwd. The launcher reuses a clean
  Codex-owned worktree under `${CODEX_HOME:-~/.codex}/worktrees/*/<repo-basename>`
  or creates `codex/session-<id>` from `origin/main`, links ignored `data/raw` and
  `data/splits` from the main checkout, and starts Codex with `--cd` there. The
  basename comes from the main checkout; do not hardcode a historical folder name.
- Fetch `origin/main` and inspect its recent commits at planning time and again
  before a PR. Check open PRs for overlap in shared files such as TODO, configs
  and tuning code. A later merge or concurrent PR may supersede the planned fix
  (#383/#516, #634 superseded #629).
- Answer shipped-state and dead-link questions from `origin/main:<path>`, not
  the worktree or parent's local `main`. The provider `post-pr-merge.sh` hooks
  fast-forward the parent only when it is clean and on `main`; they skip WIP.
- Verify checkout succeeded before rebasing; a branch held by another worktree
  must be rebased there (`git -C <path>`) or from `--detach origin/<branch>`.
  After resolving conflicts, check that **all conflict markers** are gone before
  staging and `rebase --continue`.

<a id="git-pr-ci-workflow"></a>
<a id="git--pr--ci-workflow"></a>

## PR and merge gates

- Follow feature branch → appropriate local checks → pre-PR scope judge → PR →
  current green CI/review → merge when authorized. Provider details live in
  [CODEX.md](../CODEX.md), [CLAUDE.md](../CLAUDE.md) and [GEMINI.md](../GEMINI.md).
  Preserve explicit owner approval gates, including `solve-issues` sign-off.
  Do not use `--no-verify` (including on merge-resolution commits) or `--admin`.
- Wait for current checks with `gh pr checks <N> --watch`; fix red/pending checks.
  The sole documented silent-stop exception is [CI operations](operations.md#ci-training):
  when `Run Tests` stops firing on rapid force-push, run `pytest` locally before
  an otherwise-authorized merge. This is not a general CI bypass.
- Run a gate separately from dependent mutations: never batch
  `test && commit && push` or `merge && delete`. A masked merge failure followed
  by branch deletion closed PR #622 and auto-closed #627. An authorized merge can
  use `gh pr merge <N> --squash --auto` to wait on checks.
- In worktrees, use `gh pr merge <N> --squash` without `--delete-branch` (the latter
  tries to check out the parent's `main`). Verify **MERGED**, fetch, and inspect
  the final squash content for the latest fix before separately deleting the
  remote branch. The local feature branch can stay. A tracked file on disk is
  not shipped until its change is merged.
- For stacked PRs, verify the GitHub base retarget before deleting the merged
  base, rebase to trigger CI after a base change, and give reviewers the explicit
  `gh pr diff`.
  Required test/CodeQL workflows accept `main` and `codex/**` as PR bases; push
  and production deployment/image triggers remain scoped to `main`.
- Use `gh api --paginate` for a complete inventory (default pages missed findings
  in #319). Match image tags / `head_sha` to PRs with the **full SHA**; the workflow
  log's `HEAD is now at` identifies the executed revision.
- For a suspected regression in commit X, search later `origin/main` commits for
  a revert first (#189), then inspect **every** PR in `baseline..HEAD` rather
  than selecting only the most thematic suspect.
- Copilot review comments saying it encountered an error or rate limit are
  infrastructure noise; do not address or reply to those comments.

## When making changes

- NN/feature/loss/target changes require an actual affected-position pipeline
  comparison before merge. Follow [production-path validation](validation.md#production-path)
  and [subgroup/seed requirements](validation.md#metrics-and-subgroups); unit
  tests and green CI do not establish metric neutrality. Update feature/target
  fixtures with their configuration changes.
- Large (>10-item) parallel cleanups use file-disjoint bundles, draft commits per
  bundle and one PR per risk tier, safest first. File-disjointness prevents edit
  collisions, not shared-API incompatibility: inspect every caller of a changed
  signature, including operator CLIs, and import-smoke those CLIs. The
  [113-finding remediation record](../todo/fixed-archive/fixed-code-review-remediation-110-of-113-findings-landed-across-3-file-disjo-b9c7951c.md)
  records the pattern; the 2026-05-21 Tier A `_train_nn` conflict demonstrates its
  caller-boundary limitation.
- Do not add error handling, fallbacks or validation for impossible cases;
  network/data-source boundaries are real and should be defensive.
- Scope, pending design decisions and infeasible requests follow
  [investigation gates](investigation.md#scope-and-pending-decisions).

## Docs-only exception

`[docs-only]` is a trust-based **commit-subject** opt-in only when every change is
non-behavioral: comments, docstrings, formatting, `is*` typos or import reorder
with no metric/runtime impact. A squash title or constituent subject retained as
a `* ` bullet counts; commit-body prose does not (consumers use a subject-line
awk filter). The author owns correctness; CI cannot establish it.

The tag skips the `tests.yml` matrix (`tests-pass` accepts `skipped`),
`batch-image.yml` build, `_detect-positions.yml` training and provider pre-PR
hooks. Lint and detection still run. [deploy.yml](../.github/workflows/deploy.yml)
uses path filters rather than this tag because wiki documents need deployment.
Do not put the literal tag in a subject/title when changing the opt-out machinery
(#293), and do not use it for rendered response strings such as `POSITION_INFO`
formula values. Equal metrics are not evidence that a change is non-behavioral.

## Decision and incident records

Search the [fixed-issue index](../todo/fixed-archive.md) before repeating a tried
approach. Keep each durable explanation in its canonical source; follow
[context maintenance](context-maintenance.md#evidence-and-duplication) when
removing duplication or superseded advice.

- For a non-trivial architectural change, amend the relevant
  [ADR](../docs/adr) and append its dated `## Changelog` entry. Add one terse line
  to [the ADR changelog](../docs/adr/CHANGELOG.md):
  `YYYY-MM-DD · summary · (PR #N) · → ADR-00NN`. A new decision needs the next free
  number and an [architecture-index](../docs/ARCHITECTURE.md) row. A superseding
  decision gets a new ADR and the old ADR's status becomes superseded; preserve
  its history. ADR files register in the wiki through its glob.
- For non-trivial fixes, resolve the TODO entry and record the incident once in
  `todo/fixed-archive/`, indexed from `todo/fixed-archive.md`, with
  `### [FIXED] Title`, **File(s)** (paths and commit SHA), **What**, **Fix** and
  **Lesson**. Add a focused record for previously untracked fixes.
- Truly trivial typos, formatting, lockfile bumps and comment-only edits need
  neither an ADR change nor an incident. Do not duplicate an existing lesson.
