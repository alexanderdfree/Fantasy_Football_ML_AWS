---
description: Triage claude-audit issues into tier-by-risk Codex PRs
argument-hint: [ISSUES="<numbers or range>"] [DRY_RUN=1]
---

Run the Final-Project Codex workflow for `claude-audit` findings. This is the Codex equivalent of the Claude `solve-issues` skill, adapted for Codex prompts, optional Codex subagents, `.codex/hooks/pre-pr.sh`, and `/prompts:pre-pr-judge`.

If `DRY_RUN=1`, stop after the classification and tier plan. Do not edit files, close issues, commit, push, or open PRs.

## Phase 1: Fetch And Pick Mode

Read `AGENTS.md` first, especially stop-rules, worktree workflow, large-cleanup PR partitioning, and Codex specifics.

Resolve the target issues:

- If `ISSUES` is supplied, expand the numbers or range and fetch those issues.
- Otherwise enumerate the open actionable backlog:
  `gh issue list --label claude-audit --state open --json number,title,labels,updatedAt --limit 400 --jq '.[] | select(any(.labels[]; .name=="severity-high" or .name=="severity-medium"))'`
- Fetch each target with `gh issue view <N> --json number,title,body,labels,comments`.

Parse each issue into finding records:

- Prefer the final fenced JSON block with `schema=="claude-audit/v1"`. It carries `file`, `line`, `severity`, `area`, `category`, and `first_seen_sha`.
- Strip CRLF before parsing. Fall back to prose fields such as `**File**:` only when JSON is absent or malformed.
- Use the GitHub issue number as the finding id. FIX PR bodies must cite `Closes #N`; LEAVE issues are closed directly.

Choose the mode after reading issue comments:

- **Mode A: triage-and-fix** when findings are open and not already remediated.
- **Mode B: verify-and-close** when comments map the findings to merged fix PRs or categorized LEAVE decisions and those PRs are present on `origin/main`.

## Mode A: Triage And Fix

Verify every finding before planning a fix. A grep-only verdict is not enough.

For each finding:

- Read the cited `file:line` in this worktree.
- Re-confirm the claim with the cheapest decisive check: targeted test, caller grep, config-layer read, `origin/main:<path>` read, or direct code inspection as appropriate.
- Classify as `FIX`, `LEAVE`, or `UNCERTAIN`.

Use these LEAVE categories:

- `stale`: cited code no longer matches the claim or was already changed.
- `false_positive`: current code is correct as written.
- `feature_drift`: suggestion violates stop-rules or is model tuning/design, not a defect.
- `out_of_scope`: real concern, but it belongs in a separate effort.
- `speculative`: no concrete trigger or reproducible failure mode.

Ask the user before proceeding if any `UNCERTAIN` finding would materially change the plan. Do not hide uncertain calls inside a PR.

Partition the `FIX` findings by risk tier:

- **Tier A**: docs, tests, dead-symbol cleanup, operator tools, and local scripts with no production behavior change.
- **Tier B**: behavior-equivalent or localized production fixes with low blast radius.
- **Tier C**: training, serving, shared pipeline, feature, target, cache, data-leakage, or model-output changes. Requires benchmark or pipeline evidence.

For more than ten fixes, use file-disjoint bundles:

- Use Codex subagents when available. Workers draft commits only; they do not push or open PRs.
- If subagents are unavailable, work tier by tier manually in the same file-disjoint order.
- If a bundle changes a shared function signature, grep every caller before committing. Re-bundle or add an orchestrator bridge commit if needed.

Before editing, write a compact plan:

- Master backlog: severity, area, issue, title, verdict, and tier.
- LEAVE rationale and issue-close list.
- FIX bundles by tier with files touched and worker brief.
- File-disjointness verification.
- Execution order and any user questions.

If `DRY_RUN=1`, stop here.

After plan approval, retire LEAVE issues first:

`gh label create leave --color CCCCCC --description "Audit finding triaged as noise (false positive / stale / stop-rule drift)" 2>/dev/null || true`

Then for each LEAVE issue:

- `gh issue edit <N> --add-label leave`
- `gh issue close <N> --reason "not planned" --comment "Triaged LEAVE (<category>): <reason>. Not a fix target."`

For each non-empty tier:

1. Create a staging branch from latest `origin/main`, named `audit-<issue-or-batch>/tier-<A|B|C>`.
2. Apply or cherry-pick the bundle commits for that tier.
3. Run `ruff check .`, `ruff format --check .`, and `pytest -m unit -q`.
4. For Tier C, run the affected position pipeline or benchmark and record the evidence.
5. Run `.codex/hooks/pre-pr.sh` indirectly by opening the PR, or run the same deterministic gate manually first if needed.
6. Run `/prompts:pre-pr-judge` before `gh pr create`.
7. Open one PR for the tier. Include summary, `Closes #N` lines, bundles, risk, test plan, and Tier C metric evidence when applicable.
8. Wait for green CI with `gh pr checks <N> --watch`.
9. For `audit-*/tier-*` PRs, show the final `gh pr diff <N>` and ask for explicit merge sign-off. Do not auto-merge on green CI alone.
10. After approval, merge with `gh pr merge <N> --squash`, then delete the remote branch with `git push origin --delete <branch>`.

## Mode B: Verify And Close

Use this when issues already carry remediation comments.

For each finding:

- If claimed FIXED, confirm the cited PR is merged to `origin/main` and the relevant code/doc/test change is actually present.
- If claimed LEAVE, re-validate the category against current code and `AGENTS.md` stop-rules.
- For behavioral, shared-code, or infra claims, rerun the same depth checks as Mode A.

Return a confirmation table:

`id | area | claim | verdict CONFIRMED/GAP | evidence`

If there are no GAPs:

- Post a confirmation comment on each fully verified issue.
- Close with `gh issue close <N> --reason completed`.
- Verify the issues no longer appear in the open actionable backlog.

If any GAP remains:

- Keep the affected issue open.
- Feed the GAP finding back into Mode A as a scoped fix.
- Close only issues whose findings are fully confirmed.
