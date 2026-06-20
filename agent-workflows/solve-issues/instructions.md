# Shared solve-issues instructions

These are the provider-neutral instructions for triaging and clearing the
`claude-audit` / `codex-audit` finding backlog. Provider wrappers must read this
file, set the runtime values below, apply their own tool/worktree rules, and then
execute the shared workflow.

## Runtime contract

The wrapper must define these values before executing the workflow:

- `WORKFLOW_PROVIDER`: human-readable provider name, for example `Claude Code` or `Codex`.
- `WORKFLOW_ENTRYPOINT`: the user-facing invocation, for example `solve-issues` or `/prompts:solve-issues`.
- `WORKFLOW_WRAPPER`: the wrapper file that loaded this instruction file.
- `WORKFLOW_SHARED_INSTRUCTIONS`: `agent-workflows/solve-issues/instructions.md`.
- `WORKFLOW_ISSUES`: optional issue number, range, or comma list supplied by the user; empty means enumerate the open severity-labeled backlog.
- `WORKFLOW_DRY_RUN`: `1` means stop after classification and tier plan; do not edit files, close issues, commit, push, or open PRs.
- `WORKFLOW_PLAN_MODE`: how this provider handles plan/approval phases.
- `WORKFLOW_SUBAGENTS`: whether verification/fix workers can be delegated, and how.
- `WORKFLOW_PRE_PR_GATE`: provider deterministic pre-PR gate. For Codex this is `/prompts:pre-pr-gate`; do not execute `.codex/hooks/pre-pr.sh` directly because it is a PreToolUse hook that expects hook JSON.
- `WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT`: provider scope judge, for example `pre-pr-judge` or `/prompts:pre-pr-judge`.
- `WORKFLOW_REVIEW_TOOL`: provider PR review surface, for example `/review` or `scripts/codex-review-quiet.sh --base origin/main`.
- `WORKFLOW_MEMORY_DESTINATION`: provider memory location for referenced local recall, if any.

Provider wrappers own dispatch mechanics: how to enter/exit plan mode, how to
spawn workers, how to run hooks, and how provider-specific memory is resolved.
Do not let provider mechanics change issue parsing, FIX/LEAVE categories,
closure rules, tier definitions, regress-risk handling, signoff behavior, or
worktree-safe merge rules.

## Provider wrapper values

Claude wrapper (`.claude/skills/solve-issues/SKILL.md`):

- `WORKFLOW_PROVIDER=Claude Code`.
- `WORKFLOW_ENTRYPOINT=solve-issues`.
- `WORKFLOW_PLAN_MODE=EnterPlanMode/ExitPlanMode` when available; otherwise stop for explicit user approval.
- `WORKFLOW_SUBAGENTS=Claude Agent workers, using isolated worktrees for implementation bundles`.
- `WORKFLOW_PRE_PR_GATE=.claude/hooks/pre-pr.sh`.
- `WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT=pre-pr-judge`.
- `WORKFLOW_REVIEW_TOOL=/review`.
- `WORKFLOW_MEMORY_DESTINATION=Claude project auto-memory, plus AGENTS.md for durable cross-agent lessons`.

Codex wrapper (`.codex/prompts/solve-issues.md`):

- `WORKFLOW_PROVIDER=Codex`.
- `WORKFLOW_ENTRYPOINT=/prompts:solve-issues`.
- `WORKFLOW_ISSUES` comes from `ISSUES="<numbers or range>"` when supplied.
- `WORKFLOW_DRY_RUN=1` comes from `DRY_RUN=1`; stop after the plan with no mutations.
- `WORKFLOW_PLAN_MODE=produce a compact plan and wait for approval when explicit plan-mode tools are unavailable`.
- `WORKFLOW_SUBAGENTS=Codex subagents when available; otherwise manual tier-by-tier execution in the same file-disjoint order`.
- `WORKFLOW_PRE_PR_GATE=/prompts:pre-pr-gate`.
- `WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT=/prompts:pre-pr-judge`.
- `WORKFLOW_REVIEW_TOOL=scripts/codex-review-quiet.sh --base origin/main`.
- `WORKFLOW_MEMORY_DESTINATION=$CODEX_HOME/memories, plus AGENTS.md for durable cross-agent lessons`.

Gemini/Antigravity wrapper (`.agents/skills/solve-issues/SKILL.md`):

- `WORKFLOW_PROVIDER=Gemini CLI`.
- `WORKFLOW_ENTRYPOINT=activate_skill(name="solve-issues")`.
- `WORKFLOW_PLAN_MODE=present the plan and stop for explicit user approval before any mutation`.
- `WORKFLOW_SUBAGENTS=delegate implementation bundles to Gemini subagents if available; otherwise work them sequentially`.
- `WORKFLOW_PRE_PR_GATE=the .gemini/ BeforeTool pre-PR hook when wired; otherwise run the checks manually — ruff check . && ruff format --check . && pytest -m unit`.
- `WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT=activate_skill(name="pre-pr-judge")`.
- `WORKFLOW_REVIEW_TOOL=the @gemini-cli /review PR workflow (.github/workflows/gemini-review.yml)`.
- `WORKFLOW_MEMORY_DESTINATION=Gemini Markdown memory, plus AGENTS.md for durable cross-agent lessons`.

The wrappers stay discoverable at their existing paths. This file is the
behavioral source of truth.

## Codex parity requirements preserved from the prompt wrapper

These Codex-specific behaviors are part of the shared contract and must not be
dropped when editing this file:

- `DRY_RUN=1` stops after classification and tier planning; it performs no file edits, issue closes, commits, pushes, or PR creation.
- Explicit `ISSUES` accepts one issue, a range such as `#338-348`, or a comma list such as `#338,#341`.
- Per-finding issues prefer the final fenced JSON block with `schema=="agent-audit/v1"` or legacy `schema=="claude-audit/v1"`; strip CRLF before slicing and parsing; preserve `severity`, `regress_risk`, and `area` from labels/JSON when present.
- Legacy dated batch issues, `#NNN split` issues, and area-section issues are parsed into child finding records; parent issues close only when all child findings are confirmed.
- `out_of_scope` and unresolved `UNCERTAIN` findings stay open unless the user explicitly decides otherwise; do not label them `leave`.
- Codex workers draft commits only when subagents are available; they do not push or open PRs. Without subagents, execute the same tier-by-tier plan manually.
- Codex tier PRs run `ruff check .`, `ruff format --check .`, and `pytest -m unit -q` in the foreground, then `/prompts:pre-pr-judge` before `gh pr create`.
- Codex worktree merges use `gh pr merge <N> --squash` followed by `git push origin --delete <branch>`; never use `gh pr merge --delete-branch` from a worktree.


# Solve audit-job issues

The scheduled audit routines file **one GitHub issue per finding**, each labeled with the producer label (`claude-audit` or `codex-audit`) + one severity label (`severity-docs`/`severity-low`/`severity-medium`/`severity-high`) + one model regress-risk label (`regress-risk-docs`/`regress-risk-low`/`regress-risk-medium`/`regress-risk-high`) + an area label (`qb`/`shared`/`docs`/…). The open severity-labeled issues across both labels are the live backlog; a closed `[claude-audit] checkpoint …` or `[codex-audit] checkpoint …` issue per fire records the audited SHA (it is **not** a finding — it carries no severity label, so it never appears in the backlog query). A meaningful fraction of findings are real bugs; the rest are noise: stale claims, false positives, or suggestions that re-introduce reverted designs (rolling features into the attention static branch, training on `fantasy_points`, loss-config knobs in `tune_nn.py`, etc. — see [AGENTS.md](../../AGENTS.md) "Stop rules").

This shared workflow enters the provider's planning/approval phase, triages each open finding into **FIX** or **LEAVE** (with a category), then drafts the fix plan using the project's tier-by-risk PR consolidation pattern (AGENTS.md "Large (>10-item) parallel cleanups" + provider memory lessons where available). It produces a verdict + bundle plan for user approval — **no branches cut, no workers spawned for code changes, until the user approves**.

## Two modes

This workflow runs in one of two modes, chosen after Phase 1 fetches the target issue(s):

- **Mode A — triage-and-fix** (the default; everything in Phases 1–5 and "How to act on the approved plan" below). The open per-finding issues are fresh/unaddressed → triage each FIX/LEAVE → close the LEAVE issues → bundle the FIX set into tier-by-risk PRs (whose merges close the FIX issues via `Closes #N`). This is the mode for the live open backlog.
- **Mode B — verify-and-close** (the "## Mode B" section below). The target issue(s) **already carry a remediation record** — a comment mapping each finding to a merged fix PR or a categorized LEAVE. The job is to CONFIRM the remediation actually landed on `main` and is sound, then CLOSE the issue(s). This applies to a **legacy batch issue** (a dated multi-finding `[claude-audit] <date> — N findings` or `[codex-audit] <date> — N findings` dump or a `#NNN split — <Area>`) passed explicitly, or to re-confirming a previously-worked set.

**Detecting the mode** (do this right after Phase 1's fetch): choose **Mode B** iff —

1. a comment on the issue matches `/remediation .*complete/i`, or otherwise maps each finding to a PR # / LEAVE category; AND
2. the cited fix PRs are merged to `main` (`git log origin/main --grep=<area/PR marker>`).

Otherwise run **Mode A**. A large SHA-drift (the issue's `@<sha>` is many commits behind `main`, per Phase 1 step 4) is a strong hint to look for a remediation comment *before* assuming the findings are still open.

## When to run

- User invokes `/solve-issues` (optionally with an explicit `<issue-number>`, a range `#A-B`, or a list `#A,#B,…` to override auto-detection).
- User asks to "work down the audit findings", "triage the audit backlog", "clear the audit backlog", or similar → usually **Mode A**.
- User asks to "confirm/close an already-worked audit backlog", "verify the split issues are done", "mark the audit issues done if fixed", or points at a `#NNN split` set → **Mode B** (verify-then-close).
- After a fresh `[claude-audit]` or `[codex-audit]` cycle has added findings → **Mode A**.

Skip when:

- The user is fixing one specific bug — use a normal flow, not this skill.
- The target is a **non-audit** issue (a user-reported bug, a generic task issue) — use plain plan mode. (Finite *audit* issues — `#NNN split`, dated per-cycle, or consolidated — are **in scope**: run **Mode B**.)
- The user has indicated PRs should not be opened in this session — skip Mode A. (Mode B's verify-then-close opens no PRs unless a gap is found, so it may still apply.)

## How to run

If the provider has an explicit plan mode, enter it first. Otherwise, produce the plan in the conversation and stop for user approval before any mutation. The workflow produces a triage + bundling plan for approval — no code edits, no commits, no branches, no PRs during the plan phase.

### Phase 1 — Fetch the target issue(s)

1. Resolve the target:
   - **Default (no arg)** — the open per-finding backlog. Use the same `severity-*` filter the producers dedupe on (`routines/audit/instructions.md` Step 1):
     ```
     for label in claude-audit codex-audit; do
       gh issue list --label "$label" --state open --json number,title,labels,updatedAt --limit 400 \
         --jq '.[] | select(any(.labels[]; .name | test("^severity-(docs|low|medium|high)$")))'
     done | jq -s 'unique_by(.number) | sort_by(.number)'
     ```
     The filter drops the closed `checkpoint` issues automatically (they carry no severity label) and yields the live finding set. Each kept issue is ONE finding.
   - Explicit `/solve-issues <N>` → use `<N>` (one issue). If it carries a severity label it's a single per-finding issue; if it's a **legacy batch issue** (`[claude-audit] <date> — N findings` or `[codex-audit] <date> — N findings`) or a `#NNN split`, treat its body as multi-finding (see step 3's legacy fallback).
   - Explicit **range/list** `/solve-issues #A-B` or `#A,#B,…` → expand to the set and run per-issue.
2. Fetch full body + comments for each target issue: `gh issue view <N> --json number,title,body,labels,comments`.
3. Parse into per-finding records `{id (=issue #), severity, regress_risk, area, title, file, line, what, why_suspect, suggested_action, evidence_snippet}`:
   - **Per-finding issue (default):** one issue = one finding. Read `severity`, `regress_risk`, and `area` from the **labels** (`severity-*` / `regress-risk-*` / the area label), the rest from the body fields. **Prefer the machine-readable `json` block** (schema `agent-audit/v1`, or legacy `claude-audit/v1`, appended at the END of the body) when present — it carries `file`, `line`, `category`, `first_seen_sha`, and for new issues the producer `audit_label` plus `regress_risk`; extract it CRLF-safely the same way `routines/audit/instructions.md` Step 1 does (strip `\r` from the `gh issue view <N> --json body --jq '.body'` output, slice the fenced json block, `jq` it). If the block is absent (legacy issue) or unparseable, FALL BACK to the prose `**File**:`/`**What**:`/… body fields. Severity and area still come from LABELS either way; `category` (if present) seeds the triage hint, not the FIX/LEAVE verdict. If any issue shape lacks `regress_risk`, infer it conservatively from the files/fix shape during triage. `id` is the issue number — keep it; you'll cite `Closes #id` (FIX) or close it directly (LEAVE).
   - **Legacy fallback (explicitly-passed multi-finding issue):** if the body groups findings under area headers (`### QB`, `### Shared`, …) or it's a `#NNN split`, parse each `#### …` finding block into its own record (severity from the finding header, area from the section / split title, and `regress_risk` from labels/JSON when available or conservative inference otherwise). This drains the pre-migration batch issues without re-filing them.
4. Compare the head SHA (`@<sha>` in a finding's First-seen line, or a batch title) against `git rev-parse origin/main`. If drifted by > a handful of commits, note it ("auditor SHA is N commits behind `main`; some findings may already be stale or remediated at head").
5. **Pick the mode** (see "## Two modes"): if the target issue(s) already carry a remediation comment whose cited PRs are merged → **Mode B** (jump to "## Mode B: verify-then-close"). Otherwise → **Mode A** (continue to Phase 2).

### Phase 2 — Verify in parallel (file-disjoint workers, one per area)

Group the parsed finding records **by their `area` label** (for a legacy multi-finding issue, by the body section / split-title area). Spawn **one `WORKFLOW_SUBAGENTS` verification worker per area with open findings** (typically 6–11 workers; serialize if the provider has no parallel workers). Areas are file-disjoint by construction (each finding cites a `file:line` in its own position's tree or in `src/shared/`), so workers run truly parallel with no cross-talk.

Worker brief (template — fill the `{...}` slots, send all workers in one parallel batch):

> You are verifying audit findings for area **{AREA}** (issues **{ISSUE_NUMBERS}**). Working dir: `{worktree_path}`. Findings list (verbatim, each tagged with its issue #):
>
> {findings_verbatim}
>
> For each finding return one record:
>
> ```
> - id: #<issue>   (the GitHub issue number this finding came from; for a legacy multi-finding issue, use {area}-{N})
>   verdict: FIX | LEAVE | UNCERTAIN
>   leave_category: <if LEAVE — one of: stale | false_positive | feature_drift | out_of_scope | speculative>
>   evidence: <what you observed — quote 1–3 lines from the cited file, or test output, or grep result>
>   reasoning: <one sentence — why this verdict>
>   regress_risk: docs | low | medium | high
>   fix_tier_if_FIX: A | B | C
>   files_touched_if_FIX: [list]
>   ```
>
> **Verification rubric** — read [AGENTS.md](../../AGENTS.md) "Conventions that bite" and "Stop rules" before starting. Default verdict is **FIX**; only LEAVE with a category and reason.
>
> **Blanket scope rule:** any finding that would change a design choice, feature selection, model architecture/hyperparameters, scoring, or otherwise move model accuracy as a matter of tuning or judgment (not fixing a defect) is **LEAVE** — `feature_drift` (cite the stop-rule if one applies; else note "design/tuning choice, not a defect"). **UNLESS IT IS A CLEAR, NON-CONTROVERSIAL CORRECTNESS BUG.**
>
> The five LEAVE categories:
>
> | Category | What it means |
> |---|---|
> | `stale` | Cited `file:line` no longer matches the claim — code was already changed, or the file was deleted. |
> | `false_positive` | Code is correct as-is; auditor misread intent (e.g. flagged a deliberate guard as a bug). |
> | `feature_drift` | Suggestion violates project stop-rules: promoting rolling/L3/L5/L8/ewma/trend into `ATTN_STATIC_FEATURES`; training on `fantasy_points`; adding `HUBER_DELTAS`/`LOSS_WEIGHTS`/`head_losses`/`gated_targets` to `tune_nn.py`'s search space; adding a feature to one position's model that doesn't fit its targets; resurrecting reverted optimizations (shared-venv CI, `--preload` pre-warm). See [AGENTS.md](../../AGENTS.md) "Stop rules" for the canonical list. |
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
2. If still unresolved, batch all UNCERTAIN questions into **one** provider user-question prompt (max 4 questions). Do not spread questions across multiple turns.

For each `LEAVE` verdict with `category: feature_drift`, cross-reference the cited stop-rule in [AGENTS.md](../../AGENTS.md) by section name in the rationale — this is the part the user reviews most carefully and a precise pointer beats prose.

### Phase 4 — Bundle FIX set into tier-by-risk PRs

Apply the project's tier definitions (AGENTS.md "Large (>10-item) parallel cleanups" + memory `feedback_tier_by_risk_pr_consolidation`):

- **Tier A** — tests, docstrings, dead-symbol cleanup, operator tools (`src/qb/diagnose_outliers.py`, `src/rb/analyze_errors.py`), CLI scripts under `src/scripts/`. **No production behavior change.**
- **Tier B** — behavior-equivalent fixes: refactors, dedup, in-place → return, mechanical wiring, new validators. **May touch training-adjacent files; no MAE delta.**
- **Tier C** — bounded behavior changes: feature plumbing, data-leakage fixes, bug fixes in training/serving paths, cache fingerprint changes. A `regress-risk-high` finding requires per-position pipeline/benchmark verification by the worker; benchmark deltas are reported in the PR body.

Regress-risk ordering is ascending: `docs → low → medium → high`. Tier remains
the primary PR partition; within each tier, sort FIX findings by regress-risk,
then area, then issue # before bundling or sequencing PRs.

Within each tier, partition findings into **file-disjoint bundles** (one worker per bundle, target ~8–13 bundles per tier — the established sweet spot from the audit-318 cycle ([#323](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/323)/[#325](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/325)/[#326](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/326))). Prefer not to mix `regress-risk-high` findings with lower-risk findings when file-disjointness and the PR-count cap allow separation; if mixed, mark the bundle and PR by the highest regress-risk it contains. Each bundle:

- Lists its `finding_id`s
- Lists its max `regress_risk`
- Lists its `files_touched` (file-disjointness verified across all bundles in the tier — write the table in the plan)
- Has a one-line task summary for the worker brief

**PR count target: 2–3 PRs** (one per non-empty tier). If a tier must split, split and open PRs in regress-risk ascending order before using file area as the tiebreaker (for example, Tier C low/medium before Tier C high) — but **max 4 PRs total** to keep `tests.yml`'s 7-shard matrix CI load light. If a tier is empty after triage, skip it entirely.

For shared-code signature changes (a worker bundle modifies a function's signature in `src/shared/`), add **"grep every caller of any function whose signature you change"** to that worker's brief (AGENTS.md "File-disjointness is for parallelism, not correctness"). If the grep finds callers in other bundles, the orchestrator either re-bundles to combine them or plans an **orchestrator-bridge commit** on the staging branch (memory `feedback_tier_by_risk_pr_consolidation` — orchestrator-bridge pattern).

### Phase 5 — Write the plan + exit

Write the plan to the plan file with these sections:

0. **Master backlog (severity-ordered)** — the at-a-glance ranked view of everything open: all findings sorted by **severity (HIGH → MEDIUM → LOW → DOCS)**, then regress-risk ascending (`docs → low → medium → high`), then area, then issue #: `severity | regress-risk | area | #issue | title | verdict | tier-if-FIX`. This answers "what's open, by severity"; sections 3–5 answer "what to fix first / together". (The same view is available any time without the skill by listing severity-labeled issues under both `claude-audit` and `codex-audit`.)
1. **Triage table** — one row per finding: `#issue | severity | regress-risk | area | file:line | verdict | category-if-LEAVE | tier-if-FIX`
2. **LEAVE rationale + issues to close** — grouped by category, one line per finding pointing to the stop-rule or evidence. These issues are closed (with a reason comment) on approval — list the `#issue`s explicitly.
3. **FIX bundles per tier** — Tier A → B → C, and within each tier regress-risk ascending, listing bundles with `bundle_id | max_regress_risk | finding_ids (#issues) | files | worker_brief_summary`
4. **File-disjointness verification** — per-tier table proving no file appears in two bundles
5. **Execution sequence** — Tier A PR(s) from lower to higher regress-risk → wait green/merge → Tier B PR(s) lower to higher regress-risk → … (the CI-light cadence). Each PR body cites `Closes #N` for every finding-issue it fixes, so merging auto-closes them.
6. **Open questions for user** — anything UNCERTAIN that needs the user's call before workers spawn

Then exit the provider's planning/approval phase. If the provider has no explicit plan-mode tool, stop after presenting the plan and wait for user approval.

## Mode B: verify-then-close (alternate to Phases 2–5)

Use this when Phase 1 selected Mode B: the findings are already claimed-remediated and the issue(s) are finite. The goal is **confirm, then close** — Phases 2–4's FIX bundling does not run unless verification turns up a gap.

### Phase 2V — Verify in parallel (one `WORKFLOW_SUBAGENTS` worker per area)

Spawn **one `WORKFLOW_SUBAGENTS` worker per area / per split issue**, all in one parallel batch (or sequentially if the provider has no parallel workers). Worker brief (fill the `{...}` slots):

> You are VERIFYING already-claimed remediation of audit findings for area **{AREA}**, issue(s) **#{N}**. Working dir: `{worktree_path}` — confirm it is at `origin/main` HEAD; read files with **absolute paths** (memory `feedback_edit_tool_worktree_path`). Each issue body lists findings (id + `file:line` + what + why) and carries a remediation comment mapping each finding to a FIX (with PR #) or a LEAVE (category + reason); the fix PRs are already merged to `main`. CONFIRM the remediation is real and sound — do **not** re-fix, edit, commit, or push.
>
> Fetch `gh issue view <N> --json body,comments`. For each finding:
>
> - **claimed FIXED** → open the cited `file:line` and confirm the described bug is gone / the fix is present; quote 1–3 lines as evidence. Docs findings: confirm the doc now matches the code it describes (the stale claim is gone **and** the new text is accurate).
> - **claimed LEAVE** → re-validate the category against current code (`false_positive` = genuinely correct as-is; `stale` = genuinely resolved elsewhere; `feature_drift` = the suggestion would violate an AGENTS.md stop-rule). **A LEAVE that is actually a real, unfixed bug is the critical thing to catch → flag it GAP.**
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

Write the plan with: (1) the **confirmation table**; (2) the **per-issue close list** (which issues close, with the fix-PR list to cite in each close comment); (3) any **gap → tier-PR sub-plan** (reusing Phase 4's bundling). Then exit the provider's planning/approval phase. If the provider has no explicit plan-mode tool, stop after presenting the plan and wait for user approval.

## How to act on the approved plan

Once the user approves the plan:

**First, retire the LEAVE issues.** For each finding triaged LEAVE (`false_positive` / `feature_drift` / `stale` / `speculative`), label it `leave` and close it as **not planned** with a one-line reason (for `feature_drift`, cite the AGENTS.md stop-rule by section name). The `leave` label + `not planned` state distinguish a noise issue from a genuinely-fixed one (which closes `completed`), so each audit producer can compute a per-area real-vs-noise **yield** to weight its worker budgets ([shared audit instructions](../../routines/audit/instructions.md) Step 1):
```
gh label create leave --color CCCCCC --description "Audit finding triaged as noise (false positive / stale / stop-rule drift)" 2>/dev/null || true
gh issue edit <#> --add-label leave
gh issue close <#> --reason "not planned" --comment "Triaged LEAVE (<category>): <reason / stop-rule section>. Not a fix target."
```
`out_of_scope` and UNCERTAIN-deferred issues stay **open** — they remain the visible backlog (they are NOT noise; do not label them `leave`). Then execute the FIX tiers one at a time, and execute split PRs within a tier in regress-risk ascending order. For each tier or split tier slice:

1. **Staging branch from `origin/main`**, with a unique branch per tier slice:
   ```
   git fetch origin main --quiet
   git checkout -b audit-NNN/tier-X origin/main
   ```
   Use `audit-NNN/tier-X` only for an unsplit tier. If a tier is split by regress-risk, include the split key in the branch name, for example `audit-NNN/tier-C-low`, `audit-NNN/tier-C-medium`, or `audit-NNN/tier-C-high`, so each slice gets an isolated PR.
2. **Spawn all bundle workers in parallel** using the provider worker mechanism described by the injected `WORKFLOW_SUBAGENTS` (each provider's wrapper supplies its own — e.g. Claude `Agent` workers in isolated worktrees, Codex subagents, or sequential tier-by-tier execution where no parallel worker exists). Each worker:
   - Symlinks data dirs in its worktree (memory `feedback_worktree_data_symlink`): `main_root="$(dirname "$(git rev-parse --git-common-dir)")"; ln -sf "$main_root/data/"{splits,raw} data/` (portable — derives the parent checkout from git, no hardcoded path)
   - Applies its bundle's fixes
   - For bundles whose max regress-risk is `high`: runs `python -m src.{pos}.run_pipeline` for the affected position(s) and diffs `benchmark_history/` (or `{pos}/outputs/`) against `origin/main` baseline
   - Runs `pytest -m unit -q` + `ruff check . && ruff format --check .` (**foreground** — memory `feedback_background_pytest_terminates_agents`)
   - Commits to its worktree branch, **does NOT push, does NOT open a PR** (AGENTS.md "Large (>10-item) parallel cleanups")
   - Reports back: commit SHA, branch name, files modified, findings skipped + why, any cross-bundle test-contract gaps flagged
3. **Verify worker output**: `git worktree list | grep agent-` should show one worktree per spawned worker (memory `feedback_agent_isolation_with_background` — async returns can lag). For any worker that did NOT report a commit SHA, take over the worktree directly (memory `feedback_take_over_interrupted_agent`).
4. **Cherry-pick each bundle commit onto the staging branch** in the planned regress-risk ascending bundle order. After any conflict resolution via Edit, **grep for `<<<<<<<` markers before `git add`** (memory `feedback_verify_no_conflict_markers`).
5. **Orchestrator-bridge commit (if any)** for cross-bundle test-contract gaps. Subject: `fix(audit-NNN, orchestrator, <tier>): <short summary>`.
6. **Run the provider pre-PR gate** locally (`WORKFLOW_PRE_PR_GATE`; Codex uses `/prompts:pre-pr-gate`, not `.codex/hooks/pre-pr.sh` directly). If a gate false-positives (e.g. mtime on stash-pop), surface the 3 options to the user (eat cost / authorized bypass / fix the gate) — memory `feedback_surface_gate_friction`. Do not `--no-verify`.
7. **Rebase** to ensure clean against `origin/main`: `git fetch origin main && git rebase origin/main` (memory `feedback_rebase_before_pre_pr_judge`).
8. **Invoke the provider pre-pr judge entrypoint (`WORKFLOW_PRE_PR_JUDGE_ENTRYPOINT`)** (mandatory — see [AGENTS.md](../../AGENTS.md) "When making changes").
9. **Open the tier PR** with body following the PR [#325](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/325) / [#326](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/326) structure:
   - **Summary** — tier name, max regress-risk, bundle count, finding count, cherry-pick method
   - **Closes** — `Closes #N` for every finding-issue fixed in this tier (so merge auto-closes them)
   - **Bundles** — grouped by regress-risk then area, each with bundle ID, commit SHA, files, one-line description
   - **Deferred** — bundles bumped to a later tier/regress-risk split and why
   - **Risk** — behavior risk plus model regress-risk; for `regress-risk-high`, list the metric deltas
   - **Test plan** — pytest / ruff / benchmark checklist
   - For Tier C: **mandatory Batch dry-run callout** if any bundle touches GPU code paths (memory `feedback_gpu_guarded_code_needs_gpu_test`)
10. **Wait green CI** (`gh pr checks <N> --watch`) before opening the next tier's PR. This is the CI-load-light cadence — sequential PRs, not all three open at once.
11. **Get explicit user merge sign-off, then merge.** After green CI, show the user the PR diff (`gh pr diff <N>`) and — for `regress-risk-high` fixes — the benchmark deltas from the PR body, and ask for explicit approval through the provider's user-question mechanism. Only after the user approves: `gh pr merge <N> --squash`, then `git push origin --delete <branch>` separately. Never auto-merge a solve-issues PR on green CI alone; provider post-PR hooks preserve the same stop for `audit-*/tier-*` branches.
12. **Confirm closure** — the merged PR's `Closes #N` auto-closes each finding-issue it fixed. Spot-check with `gh issue view #N --json state` (CLOSED); manually `gh issue close #N` any that GitHub didn't auto-close (wording mismatch, etc.). LEAVE issues were already closed at the top of this section.

After all tier PRs land, the open `severity-*`-labeled backlog should show only `out_of_scope` + UNCERTAIN→deferred findings. The next `[claude-audit]` or `[codex-audit]` cycle won't re-file the fixed/closed ones — producer dedupe spans **closed** issues from both labels too.

### Mode B execution (verify-then-close)

Once the verify-then-close plan is approved:

1. **If any GAP** — run the Mode A per-tier flow above for the gap findings first (staging branch → workers → cherry-pick → tier PR → green CI → merge), then re-verify the gap before closing its issue.
2. **For each confirmed FIXED per-finding issue**, post a confirmation comment and close it as completed:
   ```
   gh issue comment <N> --body "Re-verified at <sha> — FIXED present on main, 0 gaps. Remediation landed via PRs <list>. Closing as completed."
   gh issue close <N> --reason completed
   ```
3. **For each confirmed noise LEAVE per-finding issue**, preserve audit-yield accounting by labeling it `leave` and closing it as not planned:
   ```
   gh label create leave --color CCCCCC --description "Audit finding triaged as noise (false positive / stale / stop-rule drift)" 2>/dev/null || true
   gh issue edit <N> --add-label leave
   gh issue close <N> --reason "not planned" --comment "Re-verified at <sha> — confirmed LEAVE (<category>): <reason / stop-rule section>. Not a fix target."
   ```
4. **For confirmed legacy batch / `#NNN split` parent issues**, close only after every child finding is confirmed. Summarize the child outcomes in the close comment, including FIXED PRs and LEAVE categories; close the parent as completed unless the parent itself is a pure noise issue.
5. **Close only what's confirmed.** A per-finding issue closes when its fix is confirmed on `main` (or its LEAVE re-validated); a legacy batch / `#NNN split` issue closes only when ALL its findings are confirmed. Gap-bearing issues stay open until their fix PR merges and re-verify is clean.
6. **Verify the close**: `gh issue list --state open` no longer lists the closed set; the live cycle issues remain.

## What this catches that ad-hoc triage doesn't

- **Feature-drift LEAVE category** encodes the project's stop-rules from AGENTS.md and auto-memory directly into the verification rubric. Audit suggestions that violate "no rolling into ATTN_STATIC_FEATURES" or "no training on fantasy_points" get caught at triage, not at PR review.
- **CI-friendly PR cadence** — 2–3 PRs instead of 50+ per-bug PRs cuts ~95% of `tests.yml`'s 7-shard matrix runs.
- **Plan-mode-first + merge sign-off** — verdict list and bundling strategy are user-approved before any branches are cut or workers spawn. Workers operate on a vetted plan; nothing speculative ships. And beyond plan approval, each tier PR stops for **explicit user merge sign-off** (the diff + any `regress-risk-high` benchmark deltas) — a solve-issues PR is never auto-merged on green CI alone (enforced by `post-pr-create.sh` for `audit-*/tier-*` branches).
- **Reuses established orchestration** — the per-tier worker → cherry-pick → staging-branch → one-PR flow has shipped 6+ tier PRs (the code-review remediation rollup #312/#314/#315, audit-318 cycles) without conflict-driven rebundles.
- **Verify-then-close (Mode B) retires remediated backlogs** — confirms a remediation actually held on `main` (not merely that PRs merged — memory `feedback_squash_merge_verify_content`) and closes the finite tracking issues, so the next `[claude-audit]` or `[codex-audit]` re-scan starts from a true-clean state instead of re-flagging already-fixed findings or leaving split issues open indefinitely.

## Format example — triage table excerpt

```
| #issue | sev    | regress-risk | area   | file:line                          | verdict | category       | tier |
|--------|--------|--------------|--------|------------------------------------|---------|----------------|------|
| #412   | HIGH   | low          | k      | tests/k/test_attn_pipeline.py:80   | FIX     | —              | B    |
| #418   | MEDIUM | high         | dst    | src/dst/data.py:358                | FIX     | —              | C    |
| #421   | DOCS   | docs         | docs   | docs/method_contracts.md:217       | FIX     | —              | A    |
| #409   | LOW    | medium       | qb     | src/qb/config.py:142               | LEAVE   | feature_drift  | —    |
| #415   | MEDIUM | low          | shared | src/shared/pipeline.py:512         | LEAVE   | false_positive | —    |
```

## Format example — bundle plan excerpt

```
### Tier A — docs + tests (3 bundles, 1 PR)

| bundle    | max regress-risk | findings   | files                            | brief                                          |
|-----------|------------------|------------|----------------------------------|------------------------------------------------|
| W.DOCS    | docs             | #421, #423 | docs/method_contracts.md         | Fix stale feature counts; remove deleted refs. |
| W.TESTS-K | low              | #412       | tests/k/test_attn_pipeline.py    | Route cfg through build_pipeline_config.       |
| W.SCRIPTS | low              | #407       | src/qb/diagnose_outliers.py      | Update _train_nn callsite to new signature.    |

File-disjointness: ✓ (no file in two bundles).
```

## Format example — verify-then-close confirmation excerpt (Mode B)

```
| id      | area    | claim             | verdict   | evidence                                            |
|---------|---------|-------------------|-----------|-----------------------------------------------------|
| DST-002 | DST     | FIXED via #PR     | CONFIRMED | src/dst/data.py away-row spread_line sign-flipped   |
| K-005   | K       | LEAVE (false_pos) | CONFIRMED | is_home guard valid — left-merge can miss schedule  |
| SRV-001 | Serving | LEAVE (kept)      | CONFIRMED | K/DST half_ppr==ppr by design (no reception weight) |
| DATA-09 | Data    | FIXED via #PR     | GAP       | claimed-removed helper still imported at loader.py  |

Tally (illustrative): 11 issues, 130/131 CONFIRMED, 1 GAP (DATA-09 → Tier A). Close the 10 clean issues now; the gap-bearing issue waits on its fix PR.
```
