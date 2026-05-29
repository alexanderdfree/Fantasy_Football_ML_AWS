---
name: solve-issues
description: Triage the rolling [claude-audit] GH issue, then plan parallel-worker fixes bundled into 2–3 tier-by-risk PRs for user approval. Enters plan mode. Verifies each open finding, classifies FIX vs LEAVE (skipping stale claims, false positives, and feature-drift suggestions that violate project stop-rules), partitions FIX set into file-disjoint bundles across Tier A/B/C risk tiers, then exits plan mode for approval. Trigger with `/solve-issues` when the user wants to clear the rolling audit backlog; not for one-off bug fixes. Also runs a verify-then-close mode for an already-remediated backlog passed explicitly (e.g. a `#NNN split` set like `/solve-issues #338-348`): confirms each finding's fix is on `main` (or its LEAVE still holds), then closes the finite tracking issues — never the rolling issue.
---

# Solve audit-job issues

The rolling `[claude-audit] rolling — N open findings @<sha>` GitHub issue collects auditor-flagged code smells per cycle. A meaningful fraction are real bugs; the rest are noise: stale claims, false positives, or suggestions that re-introduce reverted designs (rolling features into the attention static branch, training on `fantasy_points`, loss-config knobs in `tune_nn.py`, etc. — see [CLAUDE.md](CLAUDE.md) "Stop rules").

This skill enters plan mode, triages each open finding into **FIX** or **LEAVE** (with a category), then drafts the fix plan using the project's tier-by-risk PR consolidation pattern (CLAUDE.md "Sub-agent contract — two shapes" + auto-memory `feedback_tier_by_risk_pr_consolidation`). It produces a verdict + bundle plan for `ExitPlanMode` approval — **no branches cut, no workers spawned for code changes, until the user approves**.

## Two modes

This skill runs in one of two modes, chosen after Phase 1 fetches the target issue(s):

- **Mode A — triage-and-fix** (the default; everything in Phases 1–5 and "How to act on the approved plan" below). Findings are fresh/unaddressed → triage each FIX/LEAVE → bundle the FIX set into tier-by-risk PRs. This is the mode for the live `rolling` issue.
- **Mode B — verify-and-close** (the "## Mode B" section below). The target issue(s) **already carry a remediation record** — a comment mapping each finding to a merged fix PR or a categorized LEAVE — **and** are *finite* tracking issues (a `#NNN split — <Area>`, a dated per-cycle dump, or a consolidated issue), not the perpetual `rolling` collector. The job is to CONFIRM the remediation actually landed on `main` and is sound, then CLOSE the finite issue(s).

**Detecting the mode** (do this right after Phase 1's fetch): choose **Mode B** iff all three hold —

1. a comment on the issue matches `/remediation .*complete/i`, or otherwise maps each finding to a PR # / LEAVE category;
2. the cited fix PRs are merged to `main` (`git log origin/main --grep=<area/PR marker>`);
3. the issue title is **not** `rolling`.

Otherwise run **Mode A**. A large SHA-drift (the issue's `@<sha>` is many commits behind `main`, per Phase 1 step 4) is a strong hint to look for a remediation comment *before* assuming the findings are still open.

## When to run

- User invokes `/solve-issues` (optionally with an explicit `<issue-number>`, a range `#A-B`, or a list `#A,#B,…` to override auto-detection).
- User asks to "work down the audit findings", "triage the rolling audit issue", "clear the audit backlog", or similar → usually **Mode A**.
- User asks to "confirm/close an already-worked audit backlog", "verify the split issues are done", "mark the audit issues done if fixed", or points at a `#NNN split` set → **Mode B** (verify-then-close).
- After a fresh `[claude-audit]` cycle has added findings → **Mode A**.

Skip when:

- The user is fixing one specific bug — use a normal flow, not this skill.
- The target is a **non-audit** issue (a user-reported bug, a generic task issue) — use plain plan mode. (Finite *audit* issues — `#NNN split`, dated per-cycle, or consolidated — are **in scope**: run **Mode B**.)
- The user has indicated PRs should not be opened in this session — skip Mode A. (Mode B's verify-then-close opens no PRs unless a gap is found, so it may still apply.)

## How to run

If not already in plan mode, call `EnterPlanMode` first. The skill produces a triage + bundling plan for `ExitPlanMode` approval — no code edits, no commits, no branches, no PRs during the plan phase.

### Phase 1 — Fetch the target issue(s)

1. Resolve the target:
   - Explicit `/solve-issues <N>` → use `<N>`.
   - Explicit **range/list** `/solve-issues #A-B` or `#A,#B,…` → expand to the set and run per-issue. For a `#NNN split` set each issue **is** its own area partition, so skip the area-header parse in step 3.
   - Otherwise auto-detect the rolling issue: `gh issue list --state open --search '[claude-audit] rolling in:title' --json number,title,updatedAt --limit 5`; if multiple, pick highest-numbered.
2. Fetch full body + comments: `gh issue view <N> --json number,title,body,comments`.
3. Parse the body into per-finding records: `{id, severity (HIGH|MED), area, title, file, line, what, why_suspect, suggested_action, evidence_snippet}`. The rolling issue body groups findings under area headers (`### QB`, `### Shared`, …) — use those as the Phase 2 worker partition. (For a `#NNN split` set, the issue itself is the area.)
4. Compare the head SHA in the issue title (`@<sha>`) against `git rev-parse origin/main`. If drifted by > a handful of commits, note it ("issue auditor SHA is N commits behind `main`; some findings may already be stale or remediated at head").
5. **Pick the mode** (see "## Two modes"): if the issue(s) already carry a remediation comment whose cited PRs are merged and the title is not `rolling` → **Mode B** (jump to "## Mode B: verify-then-close"). Otherwise → **Mode A** (continue to Phase 2).

### Phase 2 — Verify in parallel (file-disjoint workers, one per area)

Spawn **one `Explore` subagent per area with open findings** (typically 6–11 workers). Areas are file-disjoint by construction (each finding cites a `file:line` in its own position's tree or in `src/shared/`), so workers run truly parallel with no cross-talk.

Worker brief (template — fill the `{...}` slots, send all workers in one parallel batch):

> You are verifying audit findings for area **{AREA}** from rolling issue **#{N}**. Working dir: `{worktree_path}`. Findings list (verbatim from the issue body):
>
> {findings_verbatim}
>
> For each finding return one record:
>
> ```
> - id: {area}-{N}
>   verdict: FIX | LEAVE | UNCERTAIN
>   leave_category: <if LEAVE — one of: stale | false_positive | feature_drift | out_of_scope | speculative>
>   evidence: <what you observed — quote 1–3 lines from the cited file, or test output, or grep result>
>   reasoning: <one sentence — why this verdict>
>   fix_tier_if_FIX: A | B | C
>   files_touched_if_FIX: [list]
>   ```
>
> **Verification rubric** — read [CLAUDE.md](CLAUDE.md) "Conventions that bite" and "Stop rules" before starting. Default verdict is **FIX**; only LEAVE with a category and reason. The five LEAVE categories:
>
> | Category | What it means |
> |---|---|
> | `stale` | Cited `file:line` no longer matches the claim — code was already changed, or the file was deleted. |
> | `false_positive` | Code is correct as-is; auditor misread intent (e.g. flagged a deliberate guard as a bug). |
> | `feature_drift` | Suggestion violates project stop-rules: promoting rolling/L3/L5/L8/ewma/trend into `ATTN_STATIC_FEATURES`; training on `fantasy_points`; adding `HUBER_DELTAS`/`LOSS_WEIGHTS`/`head_losses`/`gated_targets` to `tune_nn.py`'s search space; adding a feature to one position's model that doesn't fit its targets; resurrecting reverted optimizations (shared-venv CI, `--preload` pre-warm). See [CLAUDE.md](CLAUDE.md) "Stop rules" for the canonical list. |
> | `out_of_scope` | Real concern but belongs in a separate effort (major refactor, infra change, new design). |
> | `speculative` | "Could possibly cause" with no reproducible failure mode. |
>
> **Verification depth** — verdict from grep alone is forbidden (memory `feedback_audit_run_the_test`). For each finding:
>
> - **Always** read the cited `file:line` directly (use absolute path inside this worktree — memory `feedback_edit_tool_worktree_path`).
> - **Behavioral claims** (data leakage, wrong aggregation, sign error, regression risk): run the most-targeted relevant test, e.g. `pytest tests/{pos}/<file>::<test> -xvs` (foreground only — memory `feedback_background_pytest_terminates_agents`), or `python -m src.{pos}.run_pipeline` if the claim is pipeline-level.
> - **Shared-code claims** (`src/shared/*.py`, `src/data/*.py`): grep every caller before verdicting (memory `feedback_grep_endpoint_when_changing_contract`). A claim that's true for the cited callsite but breaks 5 other callers is FIX with `fix_tier: C`, not LEAVE.
> - **"Is X on main?" / docs-vs-code claims**: read `origin/main:<path>` via `git show origin/main:<path>`, not the worktree file (memory `feedback_check_origin_main_in_worktree`).
> - **Infra/config claims** (Batch, ECS, workflows): check all layers — per-submission `submit_job(...)` overrides can invalidate the resource default (memory `feedback_layered_config_overrides`).
>
> **Do NOT** commit, push, open PRs, modify files, or rebase. Verification only. Output one block per finding; group by verdict (FIX first, then LEAVE, then UNCERTAIN). Time budget: ~10–15 min per worker.

### Phase 3 — Consolidate verdicts

Orchestrator merges worker outputs into one master table.

For each `UNCERTAIN` verdict:

1. Read the cited code directly (orchestrator has full repo access).
2. If still unresolved, batch all UNCERTAIN questions into **one** `AskUserQuestion` call (max 4 questions). Do not spread questions across multiple turns.

For each `LEAVE` verdict with `category: feature_drift`, cross-reference the cited stop-rule in [CLAUDE.md](CLAUDE.md) by section name in the rationale — this is the part the user reviews most carefully and a precise pointer beats prose.

### Phase 4 — Bundle FIX set into tier-by-risk PRs

Apply the project's tier definitions (CLAUDE.md "Sub-agent contract" + memory `feedback_tier_by_risk_pr_consolidation`):

- **Tier A** — tests, docstrings, dead-symbol cleanup, operator tools (`src/qb/diagnose_outliers.py`, `src/rb/analyze_errors.py`), CLI scripts under `src/scripts/`. **No production behavior change.**
- **Tier B** — behavior-equivalent fixes: refactors, dedup, in-place → return, mechanical wiring, new validators. **May touch training-adjacent files; no MAE delta.**
- **Tier C** — bounded behavior changes: feature plumbing, data-leakage fixes, bug fixes in training/serving paths, cache fingerprint changes. **Requires per-position benchmark verification by the worker; benchmark deltas reported in the PR body.**

Within each tier, partition findings into **file-disjoint bundles** (one worker per bundle, target ~8–13 bundles per tier — the established sweet spot from PRs [#312](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/312)–[#326](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/326)). Each bundle:

- Lists its `finding_id`s
- Lists its `files_touched` (file-disjointness verified across all bundles in the tier — write the table in the plan)
- Has a one-line task summary for the worker brief

**PR count target: 2–3 PRs** (one per non-empty tier). If Tier C has > ~10 bundles, split Tier C into C1/C2 by file area (e.g. position-specific vs shared) — but **max 4 PRs total** to keep `tests.yml`'s 7-shard matrix CI load light. If a tier is empty after triage, skip it entirely.

For shared-code signature changes (a worker bundle modifies a function's signature in `src/shared/`), add **"grep every caller of any function whose signature you change"** to that worker's brief (CLAUDE.md "File-disjointness is for parallelism, not correctness"). If the grep finds callers in other bundles, the orchestrator either re-bundles to combine them or plans an **orchestrator-bridge commit** on the staging branch (memory `feedback_tier_by_risk_pr_consolidation` — orchestrator-bridge pattern).

### Phase 5 — Write the plan + exit

Write the plan to the plan file with these sections:

1. **Triage table** — one row per finding: `id | severity | area | file:line | verdict | category-if-LEAVE | tier-if-FIX`
2. **LEAVE rationale** — grouped by category, one line per finding pointing to the stop-rule or evidence
3. **FIX bundles per tier** — Tier A → B → C, each tier listing bundles with `bundle_id | finding_ids | files | worker_brief_summary`
4. **File-disjointness verification** — per-tier table proving no file appears in two bundles
5. **Execution sequence** — Tier A PR → wait green/merge → Tier B PR → … (the CI-light cadence)
6. **Open questions for user** — anything UNCERTAIN that needs the user's call before workers spawn

Then call `ExitPlanMode`.

## Mode B: verify-then-close (alternate to Phases 2–5)

Use this when Phase 1 selected Mode B: the findings are already claimed-remediated and the issue(s) are finite. The goal is **confirm, then close** — Phases 2–4's FIX bundling does not run unless verification turns up a gap.

### Phase 2V — Verify in parallel (one `Explore` worker per area)

Spawn **one `Explore` subagent per area / per split issue**, all in one parallel batch. Worker brief (fill the `{...}` slots):

> You are VERIFYING already-claimed remediation of audit findings for area **{AREA}**, issue(s) **#{N}**. Working dir: `{worktree_path}` — confirm it is at `origin/main` HEAD; read files with **absolute paths** (memory `feedback_edit_tool_worktree_path`). Each issue body lists findings (id + `file:line` + what + why) and carries a remediation comment mapping each finding to a FIX (with PR #) or a LEAVE (category + reason); the fix PRs are already merged to `main`. CONFIRM the remediation is real and sound — do **not** re-fix, edit, commit, or push.
>
> Fetch `gh issue view <N> --json body,comments`. For each finding:
>
> - **claimed FIXED** → open the cited `file:line` and confirm the described bug is gone / the fix is present; quote 1–3 lines as evidence. Docs findings: confirm the doc now matches the code it describes (the stale claim is gone **and** the new text is accurate).
> - **claimed LEAVE** → re-validate the category against current code (`false_positive` = genuinely correct as-is; `stale` = genuinely resolved elsewhere; `feature_drift` = the suggestion would violate a CLAUDE.md stop-rule). **A LEAVE that is actually a real, unfixed bug is the critical thing to catch → flag it GAP.**
> - Behavioral / shared-code / infra claims: same depth as the Mode A rubric — run the most-targeted test **foreground only** (memory `feedback_background_pytest_terminates_agents`); grep every caller for `src/shared/*` claims (memory `feedback_grep_endpoint_when_changing_contract`); check all config layers for Batch/ECS claims (memory `feedback_layered_config_overrides`).
>
> Return one block per finding, **GAP-first then CONFIRMED**:
>
> ```
> - id: <finding-id>  issue #<N>
>   claim: FIXED via #<pr> | LEAVE (<category>)
>   verdict: CONFIRMED | GAP
>   evidence: <quoted line(s) / grep / test result>
>   note: <one sentence; for GAP: what's actually wrong + suggested tier A/B/C>
> ```
>
> End with a one-line tally (e.g. "QB 4/4 confirmed; GAPs: none"). Time budget ~12–15 min.

Verdict from grep alone is forbidden (memory `feedback_audit_run_the_test`) — and don't treat "the PR merged" as proof the fix is live: confirm against current `main` (memory `feedback_squash_merge_verify_content`).

### Phase 3V — Consolidate & decide

Merge the worker outputs into one **confirmation table** (`id | area | claim | verdict | evidence`).

- **0 GAPs** → every finding is fixed-on-main or correctly-left → proceed to close (the plan's only write step is the close).
- **Any GAP** → those findings re-enter **Mode A** (Phase 4 bundling → tier-by-risk PR) as a scoped sub-effort. The affected issue **stays open** until its gap PR merges and a re-verify comes back clean. Partial close is allowed: issues whose findings are all CONFIRMED close now; gap-bearing issues wait.

### Phase 5V — Write the plan + exit

Write the plan with: (1) the **confirmation table**; (2) the **per-issue close list** (which issues close, with the fix-PR list to cite in each close comment); (3) any **gap → tier-PR sub-plan** (reusing Phase 4's bundling). Then call `ExitPlanMode`.

## How to act on the approved plan

Once the user approves the plan via `ExitPlanMode`, execute one tier at a time. For each tier:

1. **Staging branch from `origin/main`**:
   ```
   git fetch origin main --quiet
   git checkout -b audit-NNN/tier-X origin/main
   ```
2. **Spawn all bundle workers in parallel** with `Agent` + `isolation: "worktree"`. Each worker:
   - Symlinks data dirs in its worktree (memory `feedback_worktree_data_symlink`): `ln -sf /Users/alex/compsci372/Final-Project/data/{splits,raw} data/`
   - Applies its bundle's fixes
   - For Tier C bundles: runs `python -m src.{pos}.run_pipeline` for the affected position(s) and diffs `benchmark_history/` (or `{pos}/outputs/`) against `origin/main` baseline
   - Runs `pytest -m unit -q` + `ruff check . && ruff format --check .` (**foreground** — memory `feedback_background_pytest_terminates_agents`)
   - Commits to its worktree branch, **does NOT push, does NOT open a PR** (CLAUDE.md "Sub-agent contract — >10 items shape")
   - Reports back: commit SHA, branch name, files modified, findings skipped + why, any cross-bundle test-contract gaps flagged
3. **Verify worker output**: `git worktree list | grep agent-` should show one worktree per spawned worker (memory `feedback_agent_isolation_with_background` — async returns can lag). For any worker that did NOT report a commit SHA, take over the worktree directly (memory `feedback_take_over_interrupted_agent`).
4. **Cherry-pick each bundle commit onto the staging branch** in bundle-ID order. After any conflict resolution via Edit, **grep for `<<<<<<<` markers before `git add`** (memory `feedback_verify_no_conflict_markers`).
5. **Orchestrator-bridge commit (if any)** for cross-bundle test-contract gaps. Subject: `fix(audit-NNN, orchestrator, <tier>): <short summary>`.
6. **Run `.claude/hooks/pre-pr.sh`** locally. If a gate false-positives (e.g. mtime on stash-pop), surface the 3 options to the user (eat cost / authorized bypass / fix the gate) — memory `feedback_surface_gate_friction`. Do not `--no-verify`.
7. **Rebase** to ensure clean against `origin/main`: `git fetch origin main && git rebase origin/main` (memory `feedback_rebase_before_pre_pr_judge`).
8. **Invoke `pre-pr-judge` skill** (mandatory — see [CLAUDE.md](CLAUDE.md) "Before `gh pr create`").
9. **Open the tier PR** with body following the PR [#325](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/325) / [#326](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/326) structure:
   - **Summary** — tier name, bundle count, finding count, cherry-pick method
   - **Bundles** — grouped by area, each with bundle ID, commit SHA, files, one-line description
   - **Deferred** — bundles bumped to a later tier and why
   - **Risk** — what behavior changes (Tier B: none; Tier C: bounded, list the metric deltas)
   - **Test plan** — pytest / ruff / benchmark checklist
   - For Tier C: **mandatory Batch dry-run callout** if any bundle touches GPU code paths (memory `feedback_gpu_guarded_code_needs_gpu_test`)
10. **Wait green CI** (`gh pr checks <N> --watch`) before opening the next tier's PR. This is the CI-load-light cadence — sequential PRs, not all three open at once.
11. **Merge** (`gh pr merge <N> --squash`, then `git push origin --delete <branch>` separately — memory `gh_pr_merge_worktree`).
12. **Update the rolling issue** — comment with the merged PR # and the bundle list, so the next audit cycle starts from a known state. Do NOT auto-close the issue (it's rolling).

After all tier PRs land, the rolling issue should show only LEAVE findings + any UNCERTAIN→deferred items. The next `[claude-audit]` cycle will either re-flag them (real-but-deferred) or drop them (stale-after-fix).

### Mode B execution (verify-then-close)

Once the verify-then-close plan is approved:

1. **If any GAP** — run the Mode A per-tier flow above for the gap findings first (staging branch → workers → cherry-pick → tier PR → green CI → merge), then re-verify the gap before closing its issue.
2. **For each finite issue verified clean**, post a confirmation comment and close it:
   ```
   gh issue comment <N> --body "Re-verified at <sha> — N/N findings confirmed (FIXED present on main / LEAVE still valid, 0 gaps). Remediation landed via PRs <list>. Closing as done."
   gh issue close <N> --reason completed
   ```
3. **Never close the `rolling` issue** — only finite tracking issues (split / consolidated / dated per-cycle). The live rolling/cycle issues stay open for the next audit pass. (This extends, not replaces, Mode A step 12's "do not auto-close the rolling issue.")
4. **Verify the close**: `gh issue list --state open` no longer lists the closed set; the live cycle issues remain.

## What this catches that ad-hoc triage doesn't

- **Feature-drift LEAVE category** encodes the project's stop-rules from CLAUDE.md and auto-memory directly into the verification rubric. Audit suggestions that violate "no rolling into ATTN_STATIC_FEATURES" or "no training on fantasy_points" get caught at triage, not at PR review.
- **CI-friendly PR cadence** — 2–3 PRs instead of 50+ per-bug PRs cuts ~95% of `tests.yml`'s 7-shard matrix runs.
- **Plan-mode-first** — verdict list and bundling strategy are user-approved before any branches are cut or workers spawn. Workers operate on a vetted plan; nothing speculative ships.
- **Reuses established orchestration** — the per-tier worker → cherry-pick → staging-branch → one-PR flow has shipped 6+ tier PRs (audit-310, audit-318 cycles) without conflict-driven rebundles.
- **Verify-then-close (Mode B) retires remediated backlogs** — confirms a remediation actually held on `main` (not merely that PRs merged — memory `feedback_squash_merge_verify_content`) and closes the finite tracking issues, so the next `[claude-audit]` re-scan starts from a true-clean state instead of re-flagging already-fixed findings or leaving split issues open indefinitely.

## Format example — triage table excerpt

```
| id        | sev  | area   | file:line                          | verdict | category       | tier |
|-----------|------|--------|------------------------------------|---------|----------------|------|
| K-001     | HIGH | K      | tests/k/test_attn_pipeline.py:80   | FIX     | —              | B    |
| DST-003   | MED  | DST    | src/dst/data.py:358                | FIX     | —              | C    |
| DOCS-007  | MED  | Docs   | docs/method_contracts.md:217       | FIX     | —              | A    |
| QB-004    | MED  | QB     | src/qb/config.py:142               | LEAVE   | feature_drift  | —    |
| SHARED-12 | MED  | Shared | src/shared/pipeline.py:512         | LEAVE   | false_positive | —    |
```

## Format example — bundle plan excerpt

```
### Tier A — docs + tests (3 bundles, 1 PR)

| bundle    | findings           | files                            | brief                                          |
|-----------|--------------------|----------------------------------|------------------------------------------------|
| W.DOCS    | DOCS-007, DOCS-009 | docs/method_contracts.md         | Fix stale feature counts; remove deleted refs. |
| W.TESTS-K | K-001              | tests/k/test_attn_pipeline.py    | Route cfg through build_pipeline_config.       |
| W.SCRIPTS | OPS-002            | src/qb/diagnose_outliers.py      | Update _train_nn callsite to new signature.    |

File-disjointness: ✓ (no file in two bundles).
```

## Format example — verify-then-close confirmation excerpt (Mode B)

```
| id   | area    | claim             | verdict   | evidence                                            |
|------|---------|-------------------|-----------|-----------------------------------------------------|
| F63  | DST     | FIXED via #359    | CONFIRMED | src/dst/data.py:91 `away_info["spread_line"] = -…`  |
| F13  | K       | LEAVE (false_pos) | CONFIRMED | is_home guard valid — left-merge can miss schedule  |
| F37  | Serving | LEAVE (kept)      | CONFIRMED | K/DST half_ppr==ppr by design (no reception weight) |
| F108 | Data    | FIXED via #359    | GAP       | dead _NAME_OVERRIDES still imported at loader.py:44 |

Tally: 11 issues, 130/131 CONFIRMED, 1 GAP (F108 → Tier A). Close 10 issues now; #345 waits on the F108 PR.
```
