=== MISSION ===
Review my code in parallel for bugs/quirks — spawn as many Opus 4.8 1M Max subagents as you can, each emitting HIGH/MED-only findings with verbatim evidence. The orchestrator verifies each cited line, dedupes against open and closed GitHub issues, drops anything already in CLAUDE.md Stop rules or TODO.md Fixed archive, consolidates partial/full duplicates within this run's worker output, and creates ONE NEW GitHub issue per fire containing all that run's findings in full long-form. Watch for artifacts of unfinished (yet merged) PRs, semantic merge conflicts from concurrent PRs blind to each other's changes, cross-position inconsistencies, training-vs-inference drift, and orphan code under live test coverage.

=== IMPLEMENTATION ===
You are the orchestrator for a scheduled code-audit run on the Fantasy_Football_ML_AWS repo. Working dir is the repo root. Time budget: 30 minutes wall-clock (skipped runs exit in <1 min). You are READ-ONLY on repo files. The ONLY write actions permitted are `gh issue create`, `gh issue close` (only on issues YOU create in THIS same run), and `gh issue comment` (only on issues YOU create in THIS same run). Writing to /tmp/* is fine.

=== FINDING FORMAT (canonical, full long-form) ===
    #### F<N> <SEV>: <title>
    - **File**: `<path>:<line>`
    - **First seen**: @<short_sha> on <YYYY-MM-DD>
    - **What**: <2-3 sentences>
    - **Why suspect**: <2-3 sentences>
    - **Suggested action**: <one sentence>
    - **Evidence**: `<verbatim line from the file>`
    - **Related**: F<other_id> [optional — only if cross-referenced in STEP 3b PASS 3]

F-numbers are PER ISSUE (each new issue starts at F1; F-numbers do not carry across issues). First seen is the SHORT_SHA / DATE of the current run.

STEP 0 — Skip check (sequential, ~30 sec):
  0a. HEAD_SHA=$(git rev-parse HEAD); SHORT_SHA=$(git rev-parse --short HEAD)
  0b. gh issue list --label claude-audit --state all --limit 10 --json number --jq '.[].number' \
        | while read N; do gh issue view $N --json body,comments --jq '.body, (.comments[].body // empty)'; done > /tmp/audit_history.txt
  0c. If /tmp/audit_history.txt contains a line exactly matching `HEAD-SHA: ${HEAD_SHA}`: print SKIPPED message and exit 0.
  0d. Else proceed.
  SAFETY: If gh fails, do NOT skip.

STEP 1 — Prep (sequential, ~2 min):
  1a. Read CLAUDE.md "Stop rules" section verbatim. Hold it.
  1b. grep "^### \[FIXED\]" TODO.md — capture every title line. Hold the list.
  1c. Build the dedupe pool. Fetch bodies + comments of ALL recent claude-audit issues (open AND closed, last 30):
        gh issue list --label claude-audit --state all --limit 30 --json number --jq '.[].number' \
          | while read N; do gh issue view $N --json body,comments --jq '.body, (.comments[].body // empty)'; done > /tmp/dedupe_pool.txt
        grep -E "^#### (F[0-9]+ )?(HIGH|MED): " /tmp/dedupe_pool.txt \
          | sed -E 's/^#### (F[0-9]+ )?(HIGH|MED): //' > /tmp/known_titles.txt
        # Also extract one-line compact entries from older issues (legacy compressed format) for dedupe
        grep -E "^- F[0-9]+ \[(HIGH|MED)\]" /tmp/dedupe_pool.txt >> /tmp/known_titles.txt

STEP 2 — Fanout (parallel, ~13 min):
  Spawn 12 general-purpose subagents IN A SINGLE MESSAGE (Agent tool calls run concurrently). Each Agent call sets model="opus". Worker scopes:
    #1  QB auditor          — src/qb/, tests/qb/
    #2  RB auditor          — src/rb/, tests/rb/
    #3  WR auditor          — src/wr/, tests/wr/
    #4  TE auditor          — src/te/, tests/te/
    #5  K auditor           — src/k/, tests/k/
    #6  DST auditor         — src/dst/, tests/dst/
    #7  Shared auditor      — src/shared/, src/models/, src/training/, src/evaluation/, tests/shared/
    #8  Data+features       — src/data/, src/features/
    #9  Serving auditor     — src/serving/, tests/serving/ (extra focus: training-vs-inference drift)
    #10 Batch+CI auditor    — src/batch/, .github/workflows/, .claude/hooks/
    #11 Docs consistency    — CLAUDE.md, README.md, SETUP.md, TODO.md, docs/
    #12 Within-position invariant + broken-reference auditor — iterate over QB → DST one at a time. WITHIN-position invariant violations (LOSS_WEIGHTS ≈ 2.0/HUBER_DELTAS within one position's config; target-naming consistent within a position) AND broken-reference drift (a file in position X references a key/value X's own config.py doesn't define). DOES NOT FLAG per-position differences ACROSS positions — those are by design.

  Per-worker template (substitute {N}, {SCOPE}, {FOCUS}):
    ROLE: Auditor #{N} of 12. Scope: {SCOPE}. 12-minute budget.

    PRIMARY FOCUS:
      (a) Artifacts of unfinished-but-merged PRs (dead code, half-renamed symbols, orphan imports, commented-out blocks, TODO/FIXME, feature-list/test-fixture mismatches).
      (b) Semantic merge conflicts from concurrent PRs blind to each other's changes (check `git log -p -n 30 -- <file>` for competing recent edits).
      (c) Orphan code under live test coverage (production-side functions with zero callers in `src/` but exercised by tests).
      (d) {FOCUS}
      (e) Anything that would silently produce wrong results.

    NOT FINDINGS (by design — do not report):
      - "Position X has feature F that position Y doesn't" → per-position whitelists are intentional
      - "Position X's loss weight value differs from Y's" → only the within-position ratio invariant matters
      - "Position X uses head_hidden_overrides, Y doesn't" → per-position tuning
      - "Add F to all positions for parity" / "harmonize X across positions" → feature engineering
      - "Position X has CONFIG_TINY, Y doesn't" → optional convention
      - "Position X's NN hidden dim differs from Y's" → per-position tuning
      When in doubt, drop the finding.

    STOP RULES — drop anything overlapping:
    <inline CLAUDE.md Stop rules + every TODO.md FIXED title from STEP 1>

    SEVERITY: HIGH (wrong result / silent loss / security / benchmark-changing) or MED (unfinished-PR artifact / within-position invariant violation / broken-reference drift / semantic merge conflict / orphan code under live test coverage). NO LOW.

    OUTPUT: JSON array only. Each: {"file": "<path>", "line": <int>, "severity": "HIGH"|"MED", "title": "<<80 chars>", "what": "<2-3 sentences>", "why_suspect": "<2-3 sentences>", "suggested_action": "<one sentence>", "evidence_quote": "<verbatim line from file>"}
    Workers do NOT assign F-numbers — the orchestrator does.

STEP 3 — Verify new findings (sequential, ~5 min):
  For each new finding from each worker:
    3a. Read file at cited line. Confirm evidence_quote matches (whitespace-normalized). DROP if mismatch.
    3b. grep CLAUDE.md + TODO.md for 2-3 distinctive title keywords. DROP if matched.
    3c. Check title against /tmp/known_titles.txt (substring either direction). DROP if found — already reported in a recent claude-audit issue.
    3d. For "unfinished PR" / "semantic merge conflict" claims: `git log -p -n 20 -- <file>` and confirm consistency. DROP if completed elsewhere.
    3e. If finding matches NOT FINDINGS patterns: DROP.
  Hold /tmp/new_findings.jsonl (survivors). Hold N_NEW, N_NEW_HIGH, N_NEW_MED.

STEP 3b — Consolidate duplicates within THIS run's findings (sequential, ~2 min):
  Operates only on the new findings from this run (no rolling state). Use tentative F1..F${N_NEW} IDs.
  PASS 1 — Full duplicates: same file AND same line AND whitespace-normalized evidence match. Keep lowest tentative-F; drop the others. Record canonical ← merged.
  PASS 2 — Same-file partial duplicates: lines within ±10 AND ≥2 distinctive title keywords AND `what`/`why_suspect` describe semantically the same defect. Canonical = lowest tentative-F; merge any UNIQUE one-sentence fragments from others into canonical's what/why_suspect (no bloat); drop the others.
  PASS 3 — Cross-file related: different files, ≥3 shared title keywords, what/why_suspect of one references the other's file/symbol. Keep BOTH and inject `- **Related**: F<other>` into each.
  CONSERVATIVE BIAS: when in doubt, do not merge.
  HARD CAP: never drop more than 30% of N_NEW via consolidation. If exceeded, skip consolidation and post unconsolidated.
  Output: /tmp/consolidated_new.jsonl (final set, post-consolidation) + /tmp/consolidations.jsonl (log of merged clusters + cross-references).

STEP 4 — Compile + post (~3 min):
  HEAD_SHA=$(git rev-parse HEAD); SHORT_SHA=$(git rev-parse --short HEAD); DATE=$(date -u +"%Y-%m-%d %H:%M UTC"); DATE_ONLY=$(date -u +"%Y-%m-%d")

  ASSIGN F-NUMBERS: number the consolidated findings F1, F2, ... in worker-output order. All First seen = SHORT_SHA on DATE_ONLY.

  Case A — N_NEW == 0:
    → CLEAN CHECKPOINT. Create + immediately close an issue with title "[claude-audit] ${DATE} — 0 findings (clean) @${SHORT_SHA}" and body containing HEAD-SHA + HEAD-SHORT + Date. Use --label claude-audit. Close with comment "Clean-state checkpoint, auto-closed by audit routine."

  Case B — N_NEW > 0:
    → CREATE ONE NEW ISSUE for this run. Build /tmp/new_body.md:

      # [claude-audit] ${DATE} — ${N_OPEN_AFTER} findings (${N_HIGH} HIGH, ${N_MED} MED) @${SHORT_SHA}

      HEAD-SHA: ${HEAD_SHA}

      ## TL;DR
      | Area | HIGH | MED |
      |------|------|-----|
      ...rows by area, then a Total row...

      ## Consolidations + cross-references this run
      (only include if any from STEP 3b)
      - Consolidated: M clusters, X entries dropped
          - F<canonical> ← F<merged>, ... (`<file>` — reason)
      - Cross-referenced: K pairings
          - F<A> ↔ F<B> (`<file_a>` ↔ `<file_b>` — reason)

      ## Findings
      ### <area>
      #### F<N> HIGH: <title>
      - **File**: `path:line`
      - **First seen**: @<short_sha> on <YYYY-MM-DD>
      - **What**: ...
      - **Why suspect**: ...
      - **Suggested action**: ...
      - **Evidence**: `<evidence_quote>`
      - **Related**: F<other> [if any]
      ...grouped by area, sorted by F-number ascending within area...

      ---
      *Per-run audit issue. Each fire creates a new one. F-numbers are scoped to THIS issue. Close when triaged; dedupe against open AND closed issues prevents re-flagging.*

    HARD BOUND on body size: target ≤55k chars (safety margin under GitHub's 65k cap). If estimated body would exceed this, split into MULTIPLE issues with titles "[claude-audit] ${DATE} part 1/N — ...", "part 2/N — ...", etc. Split on area boundaries; if a single area exceeds the cap, sub-split within that area at finding boundaries (F1-F30, F31-F60, etc.). Each part has its own HEAD-SHA header + TL;DR for the findings in that part. Cross-reference the parts: each part's body footer says "This is part k of N for the ${DATE} @${SHORT_SHA} run; sibling parts: #<num1>, #<num2>, ...". The orchestrator creates them in order, captures each new issue number, then edits each part's body to inject the full sibling list at the end.

    Run:
      gh issue create --title "[claude-audit] ${DATE} — ${N_TOTAL} findings (${N_HIGH} HIGH, ${N_MED} MED) @${SHORT_SHA}" \
                      --label claude-audit \
                      --label <area-labels-comma-separated> \
                      --body-file /tmp/new_body.md
      Print the issue URL.

    If gh write fails, print full body to stdout so the run isn't lost.

CONSTRAINTS:
  - Read-only on repo files. Writing to /tmp/* is fine.
  - Do NOT push branches or open PRs.
  - You MAY close clean-checkpoint issues YOU create in this run. Do NOT close or modify any other issues, and do NOT use `gh issue edit` to modify any issue's title/body.
  - F-numbers reset to F1 in each new issue.
  - First seen on each finding = the SHORT_SHA / DATE_ONLY of THIS run (since findings are fresh in this issue).
  - Never re-flag anything in the Stop rules block, TODO.md Fixed archive, or the dedupe pool (open+closed claude-audit issue bodies and comments).
  - Never propose cross-position harmonization — feature engineering, not audit.
  - Empty runs (N_NEW == 0) post a closed clean checkpoint so HEAD-SHA is recorded for STEP 0 skip-check.
  - HARD BOUND: target ≤55k chars per issue body. Split into parts at area boundaries if exceeded.
