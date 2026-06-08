---
description: Triage agent audit issues into tier-by-risk Codex PRs
argument-hint: [ISSUES="<numbers or range>"] [DRY_RUN=1]
---

Run the Final-Project Codex workflow for `claude-audit` and `codex-audit` findings. This is the Codex equivalent of the Claude `solve-issues` skill, adapted for Codex prompts, optional Codex subagents, `.codex/hooks/pre-pr.sh`, and `/prompts:pre-pr-judge`.

If `DRY_RUN=1`, stop after the classification and tier plan. Do not edit files, close issues, commit, push, or open PRs.

## Phase 1: Fetch And Pick Mode

Read `AGENTS.md` first, especially stop-rules, worktree workflow, large-cleanup PR partitioning, and Codex specifics.

Resolve the target issues:

- If `ISSUES` is supplied, expand one issue, a range such as `#338-348`, or a comma list such as `#338,#341`.
- Otherwise enumerate the open actionable backlog across both audit labels:
  `for label in claude-audit codex-audit; do gh issue list --label "$label" --state open --json number,title,labels,updatedAt --limit 400 --jq '.[] | select(any(.labels[]; .name | test("^severity-(docs|low|medium|high)$")))'; done | jq -s 'unique_by(.number) | sort_by(.number)'`
- Fetch each target with `gh issue view <N> --json number,title,body,labels,comments`.

Parse each issue into finding records:

- Per-finding issue: one GitHub issue equals one finding. Read `severity`, `regress_risk`, and `area` from labels. Prefer the final fenced JSON block with `schema=="agent-audit/v1"` or the legacy `schema=="claude-audit/v1"`; it carries `file`, `line`, `category`, `first_seen_sha`, and for new issues `regress_risk`. Strip CRLF before slicing/parsing. Fall back to prose fields such as `**File**:` only when JSON is absent or malformed. If any issue shape lacks `regress_risk`, infer it conservatively from touched files during triage.
- Explicit legacy or split issue: if the issue is a dated `[claude-audit] <date> - N findings` or `[codex-audit] <date> - N findings` batch, a `#NNN split`, or has area sections such as `### QB` / `### Shared`, parse each `#### ...` finding block into its own record, including `regress_risk` from labels/JSON when available or conservative inference otherwise. Use the parent issue plus an area/index suffix as the finding id, but close the parent issue only when all of its findings are confirmed.
- FIX PR bodies must cite `Closes #N` for per-finding issues. LEAVE per-finding issues are closed directly; legacy/split parent issues follow the Mode B close rules.

Check freshness and choose the mode:

- Compare each finding's `first_seen_sha` or title `@<sha>` to `git rev-parse origin/main`. If it is more than a handful of commits behind, note that the finding may already be stale or remediated and look for remediation comments before assuming it is open.
- **Mode B: verify-and-close** only when comments map every finding to a merged fix PR or categorized LEAVE decision, and the cited PRs are present on `origin/main` (`git log origin/main --grep=<marker>` or equivalent).
- Otherwise use **Mode A: triage-and-fix**.

## Mode A: Triage And Fix

Verify every finding before planning a fix. A grep-only verdict is not enough.

For each finding:

- Read the cited `file:line` in this worktree.
- Re-confirm the claim with the cheapest decisive check: targeted test, caller grep, config-layer read, `origin/main:<path>` read, or direct code inspection as appropriate.
- For behavioral, shared-code, or infra claims, use the same depth as the Claude workflow: targeted tests for behavior, grep every caller for shared-code contracts, and check all Batch/ECS/workflow config layers.
- Classify as `FIX`, `LEAVE`, or `UNCERTAIN`.

> **Blanket scope rule:** any finding that would change a design choice, feature selection, model architecture/hyperparameters, scoring, or otherwise move model accuracy as a matter of tuning or judgment (not fixing a defect) is **LEAVE** — `feature_drift` (cite the stop-rule if one applies; else note "design/tuning choice, not a defect"). **UNLESS IT IS A CLEAR, NON-CONTROVERSIAL CORRECTNESS BUG.**

Use these LEAVE categories:

- `stale`: cited code no longer matches the claim or was already changed.
- `false_positive`: current code is correct as written.
- `feature_drift`: suggestion violates stop-rules or is model tuning/design, not a defect. Cite the relevant `AGENTS.md` stop-rule by section name.
- `out_of_scope`: real concern, but it belongs in a separate effort.
- `speculative`: no concrete trigger or reproducible failure mode.

Important closure rule:

- `stale`, `false_positive`, `feature_drift`, and `speculative` are noise categories and may be labeled `leave` and closed after plan approval.
- `out_of_scope` and unresolved `UNCERTAIN` findings stay open unless the user explicitly decides otherwise. Do not label them `leave`.

Ask the user before proceeding if any `UNCERTAIN` finding would materially change the plan — batch all such questions into a single prompt (max 4 questions); do not spread them across turns. Do not hide uncertain calls inside a PR.

Partition the `FIX` findings by risk tier:

- **Tier A**: docs, tests, dead-symbol cleanup, operator tools, and local scripts with no production behavior change.
- **Tier B**: behavior-equivalent or localized production fixes with low blast radius.
- **Tier C**: training, serving, shared pipeline, feature, target, cache, data-leakage, GPU-guarded, or model-output changes. `regress-risk-high` is the explicit signal that relevant pipeline/benchmark evidence is expected.

Within each tier, order FIX findings and bundles by regress-risk ascending: `docs`, `low`, `medium`, `high`. Use area and issue id as tiebreakers.

For more than ten fixes, use file-disjoint bundles:

- Use Codex subagents when available. Workers draft commits only; they do not push or open PRs.
- If subagents are unavailable, work tier by tier manually in the same file-disjoint order.
- Target one PR per non-empty tier, 2-3 PRs total. If a tier must split, split and open PRs in regress-risk ascending order before using file area as the tiebreaker, but keep the total at four PRs or fewer.
- Prefer not to mix `regress-risk-high` findings with lower-risk findings when file-disjointness and the PR-count cap allow separation; if mixed, mark the bundle and PR by the highest regress-risk it contains.
- If a bundle changes a shared function signature, grep every caller before committing. Re-bundle or add an orchestrator bridge commit if needed.

Before editing or closing issues, write a compact plan and stop for user approval:

- Master backlog sorted by severity (`HIGH`, `MEDIUM`, `LOW`, `DOCS`), then regress-risk ascending (`docs`, `low`, `medium`, `high`), then area and issue id: severity, regress-risk, area, issue/finding id, title, verdict, and tier.
- Triage table with `file:line`, regress-risk, LEAVE category, and evidence.
- LEAVE rationale and exact issue-close list, excluding `out_of_scope` and unresolved `UNCERTAIN`.
- FIX bundles by tier and regress-risk with max regress-risk, files touched, worker brief, and file-disjointness verification.
- Execution order, including regress-risk ascending order within each tier, test requirements, `regress-risk-high` benchmark requirements, and any user questions.

If `DRY_RUN=1`, stop here. Otherwise, do not mutate until the user approves the plan.

After plan approval, retire noise LEAVE issues first:

`gh label create leave --color CCCCCC --description "Audit finding triaged as noise (false positive / stale / stop-rule drift)" 2>/dev/null || true`

Then for each closeable LEAVE per-finding issue:

- `gh issue edit <N> --add-label leave`
- `gh issue close <N> --reason "not planned" --comment "Triaged LEAVE (<category>): <reason / stop-rule section>. Not a fix target."`

For each non-empty tier, execute sequentially; when a tier has multiple PRs, execute them from lower to higher regress-risk:

1. Create a staging branch from latest `origin/main`, named `audit-<issue-or-batch>/tier-<A|B|C>`.
2. Apply or cherry-pick the bundle commits for that tier in the planned regress-risk ascending bundle order. After any conflict resolution, verify `grep -c '<<<<<<<' <file>` is zero before staging.
3. Add an orchestrator bridge commit if cross-bundle test-contract gaps or shared-signature changes require it.
4. Run `ruff check .`, `ruff format --check .`, and `pytest -m unit -q` in the foreground.
5. For any `regress-risk-high` finding, run the affected position pipeline or benchmark and record deltas. If a bundle touches GPU-guarded code, include a mandatory Batch dry-run callout.
6. Rebase onto latest `origin/main` before judging/opening: `git fetch origin main --quiet` then `git rebase origin/main`.
7. Run `/prompts:pre-pr-judge` before `gh pr create`.
8. Open one PR for the tier/risk slice. Include summary with max regress-risk, `Closes #N` lines, bundles grouped by regress-risk then area, behavior risk, model regress-risk, deferred items, test plan, and `regress-risk-high` metric evidence when applicable.
9. Wait for green CI with `gh pr checks <N> --watch` before opening the next tier PR.
10. For `audit-*/tier-*` PRs, show the final `gh pr diff <N>` and any `regress-risk-high` benchmark deltas, then ask for explicit merge sign-off. Do not auto-merge on green CI alone.
11. After approval, merge with `gh pr merge <N> --squash`, then delete the remote branch with `git push origin --delete <branch>`. Do not use `--delete-branch` from a worktree.
12. Confirm closure: spot-check fixed finding issues with `gh issue view <N> --json state`; manually close only if GitHub did not process a valid `Closes #N`.

After all tier PRs land, the open severity-labeled backlog should contain only `out_of_scope` and unresolved/deferred findings.

## Mode B: Verify And Close

Use this when issues already carry remediation comments. The goal is confirm, then close; do not re-fix unless a GAP appears.

For each finding:

- If claimed FIXED, confirm the cited PR is merged to `origin/main` and the relevant code/doc/test change is actually present. Do not treat "PR merged" as sufficient evidence by itself.
- If claimed LEAVE, re-validate the category against current code and `AGENTS.md` stop-rules. A claimed LEAVE that is actually a real unfixed bug is a `GAP`.
- For behavioral, shared-code, or infra claims, rerun the same depth checks as Mode A.

Return a confirmation table:

`id | area | claim | verdict CONFIRMED/GAP | evidence`

Write a close plan and stop for user approval before mutating:

- Per-issue close list, including fix PR list or LEAVE category to cite in comments.
- Any GAP findings and their Mode A tier/bundle plan.
- Legacy/split parent issue closure only when all child findings are confirmed.

If there are no GAPs and the user approves:

- For each confirmed FIXED finite issue, post a confirmation comment and close with `gh issue close <N> --reason completed`.
- For each confirmed noise LEAVE per-finding issue, preserve the noise accounting: ensure the `leave` label exists, apply it, and close with `gh issue close <N> --reason "not planned"` plus a one-line LEAVE reason.
- Verify the issues no longer appear in the open actionable backlog.

If any GAP remains:

- Keep the affected issue open.
- Feed the GAP finding back into Mode A as a scoped fix.
- Close only issues whose findings are fully confirmed.
