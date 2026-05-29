=== MISSION ===
Review my code in parallel for bugs/quirks — spawn as many Opus 4.8 1M Max subagents as you can, each emitting HIGH/MED-only findings with verbatim evidence. The orchestrator verifies each cited line, dedupes against open and closed GitHub issues, drops anything already in CLAUDE.md Stop rules or TODO.md Fixed archive, consolidates partial/full duplicates within this run's worker output, and files ONE GITHUB ISSUE PER SURVIVING FINDING — labeled by severity (`severity-high`/`severity-medium`) and area — plus one closed checkpoint issue recording the audited SHA. Watch for artifacts of unfinished (yet merged) PRs, semantic merge conflicts from concurrent PRs blind to each other's changes, cross-position inconsistencies, training-vs-inference drift, and orphan code under live test coverage.

=== IMPLEMENTATION ===
You are the orchestrator for a scheduled code-audit run on the Fantasy_Football_ML_AWS repo. Working dir is the repo root. Time budget: 30 minutes wall-clock (skipped runs exit in <1 min). You are READ-ONLY on repo files. The ONLY write actions permitted are `gh label create` (the severity labels, idempotent), `gh issue create` (per-finding issues + the checkpoint), `gh issue comment` (only on issues YOU create in THIS same run), and `gh issue close` (only on the checkpoint issue YOU create in THIS run). Writing to /tmp/* is fine.

=== ISSUE MODEL (one issue per finding) ===
Each surviving finding becomes ONE GitHub issue:
  - **Title**: `[claude-audit] <area>: <title>` — `<area>` ∈ qb|rb|wr|te|k|dst|shared|data|serving|batch|ci|docs|cross-position; `<title>` is the finding title (<80 chars). Severity is NEVER in the title — it is a LABEL, so a HIGH↔MED reclassification doesn't mint a "new" title and break dedup.
  - **Labels**: `claude-audit`, the severity label (`severity-high` or `severity-medium`), and the area label.
  - **Area** is derived from the finding's `file` path / the worker scope that produced it: `src/qb/*`→qb … `src/dst/*`→dst, `src/shared|models|training|evaluation/*`→shared, `src/data|features/*`→data, `src/serving/*`→serving, `src/batch/*`→batch, `.github/workflows|.claude/hooks/*`→ci, docs/CLAUDE.md/README.md/SETUP.md/TODO.md→docs. Worker #12 invariant/broken-ref findings take the area of the position they pertain to.
  - **Body** (canonical long-form):
        - **File**: `<path>:<line>`
        - **Severity**: HIGH | MED
        - **Area**: <area>
        - **First seen**: @<short_sha> on <YYYY-MM-DD>
        - **What**: <2-3 sentences>
        - **Why suspect**: <2-3 sentences>
        - **Suggested action**: <one sentence>
        - **Evidence**: `<verbatim line from the file>`
        - **Related**: #<other_issue> [optional — added by a post-create comment if cross-referenced in STEP 3b PASS 3]

There are NO F-numbers. Findings are identified by their GitHub issue number. Because dedup (STEP 1c / 3c) suppresses refiling, each issue persists across fires — so its **First seen** SHA/date and `createdAt` give the finding's true age. You only ever CREATE new issues for findings not already filed; never edit or reset an existing issue.

Ensure the severity labels exist (idempotent — safe to run every fire; area labels already exist in the repo):
    gh label create severity-high   --color B60205 --description "Audit: wrong result / silent loss / security / benchmark-changing" 2>/dev/null || true
    gh label create severity-medium --color FBCA04 --description "Audit: unfinished-PR artifact / invariant / drift / orphan code" 2>/dev/null || true

STEP 0 — Skip check (sequential, ~30 sec):
  0a. HEAD_SHA=$(git rev-parse HEAD); SHORT_SHA=$(git rev-parse --short HEAD)
  0b. Fetch bodies of recent CHECKPOINT issues (each records one audited SHA):
        gh issue list --label claude-audit --state all --limit 60 --json number,title \
          --jq '.[] | select(.title | test("checkpoint")) | .number' \
          | while read N; do gh issue view $N --json body --jq '.body'; done > /tmp/audit_history.txt
  0c. If /tmp/audit_history.txt contains a line exactly matching `HEAD-SHA: ${HEAD_SHA}`: print SKIPPED message and exit 0.
  0d. Else proceed.
  SAFETY: If gh fails, do NOT skip.

STEP 1 — Prep (sequential, ~2 min):
  1a. Read CLAUDE.md "Stop rules" section verbatim. Hold it.
  1b. grep "^### \[FIXED\]" TODO.md — capture every title line. Hold the list.
  1c. Build the dedupe pool from existing PER-FINDING issues — open AND closed, those carrying a `severity-*` label (this naturally EXCLUDES checkpoint issues, which have no severity label):
        gh issue list --label claude-audit --state all --limit 400 --json number,title,labels \
          --jq '.[] | select(any(.labels[]; .name=="severity-high" or .name=="severity-medium")) | "\(.number)\t\(.title)"' > /tmp/known_issues.tsv
        # For file-aware dedup, capture each known issue's cited File: path (strip the :line):
        : > /tmp/known_files.tsv
        cut -f1 /tmp/known_issues.tsv | while read N; do
          F=$(gh issue view "$N" --json body --jq '.body' | grep -m1 -oE '`[^`]+:[0-9]+`' | tr -d '`' | sed -E 's/:[0-9]+$//')
          printf '%s\t%s\n' "$N" "$F" >> /tmp/known_files.tsv
        done
      Dedupe KEY = (area + cited file-path + ≥2 distinctive title keywords). The title prefix `[claude-audit] <area>:` carries area; `/tmp/known_files.tsv` carries the file.

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
    Workers do NOT create issues and do NOT assign issue numbers — the orchestrator does.

STEP 3 — Verify new findings (sequential, ~5 min):
  For each new finding from each worker:
    3a. Read file at cited line. Confirm evidence_quote matches (whitespace-normalized). DROP if mismatch.
    3b. grep CLAUDE.md + TODO.md for 2-3 distinctive title keywords. DROP if matched.
    3c. Dedupe against existing issues: a new finding is a DUPLICATE (DROP it) when some entry in /tmp/known_issues.tsv has the SAME area (its title prefix `[claude-audit] <area>:`) AND the SAME cited file (per /tmp/known_files.tsv) AND ≥2 shared distinctive title keywords. The pool already spans open AND closed — so a finding already filed, already triaged-closed, or already fixed is NOT re-filed.
    3d. For "unfinished PR" / "semantic merge conflict" claims: `git log -p -n 20 -- <file>` and confirm consistency. DROP if completed elsewhere.
    3e. If finding matches NOT FINDINGS patterns: DROP.
  Hold /tmp/new_findings.jsonl (survivors). Hold N_NEW, N_NEW_HIGH, N_NEW_MED.

STEP 3b — Consolidate duplicates within THIS run's findings (sequential, ~2 min):
  Operates only on the new findings from this run (no rolling state). Use tentative local IDs t1..t${N_NEW}.
  PASS 1 — Full duplicates: same file AND same line AND whitespace-normalized evidence match. Keep one; drop the others.
  PASS 2 — Same-file partial duplicates: lines within ±10 AND ≥2 distinctive title keywords AND `what`/`why_suspect` describe semantically the same defect. Keep one; merge any UNIQUE one-sentence fragments from the others into its what/why_suspect (no bloat); drop the others.
  PASS 3 — Cross-file related: different files, ≥3 shared title keywords, what/why_suspect of one references the other's file/symbol. Keep BOTH; record the pairing (tentative IDs + one-line reason) in /tmp/related.tsv so the orchestrator can link the two ISSUES with a comment AFTER both are created (issue numbers aren't known until then).
  CONSERVATIVE BIAS: when in doubt, do not merge.
  HARD CAP: never drop more than 30% of N_NEW via consolidation. If exceeded, skip consolidation and file unconsolidated.
  Output: /tmp/consolidated_new.jsonl (final set, each tagged with its tentative ID + resolved area) + /tmp/related.tsv (tA<TAB>tB<TAB>reason).

STEP 4 — File issues (~3 min):
  HEAD_SHA=$(git rev-parse HEAD); SHORT_SHA=$(git rev-parse --short HEAD); DATE=$(date -u +"%Y-%m-%d %H:%M UTC"); DATE_ONLY=$(date -u +"%Y-%m-%d")
  All First seen = SHORT_SHA on DATE_ONLY.

  4a. PER-FINDING ISSUES (only if N_NEW > 0). For each finding in /tmp/consolidated_new.jsonl: write the canonical body (see ISSUE MODEL) to /tmp/body.md, resolve its area + severity label, create the issue, and record the new number keyed by tentative ID:
        SEV_LABEL=$( [ "$severity" = "HIGH" ] && echo severity-high || echo severity-medium )
        URL=$(gh issue create \
                --title "[claude-audit] ${area}: ${title}" \
                --label claude-audit --label "$SEV_LABEL" --label "$area" \
                --body-file /tmp/body.md)
        NUM=${URL##*/}                      # issue number from the URL
        printf '%s\t%s\t%s\n' "$tID" "$NUM" "$URL" >> /tmp/filed.tsv
      If a `gh issue create` fails (e.g. an unexpected area label), retry once WITHOUT the area label; if it still fails, print the full body to stdout so the finding isn't lost.

  4b. RELATED LINKS. For each pairing tA<TAB>tB<TAB>reason in /tmp/related.tsv, resolve to issue numbers via /tmp/filed.tsv and cross-link with comments on the issues you just created:
        gh issue comment "$NUM_A" --body "Related: #${NUM_B} — ${reason}"
        gh issue comment "$NUM_B" --body "Related: #${NUM_A} — ${reason}"

  4c. CHECKPOINT ISSUE (ALWAYS — every run, including clean N_NEW==0 runs). Records the audited SHA for STEP 0's skip-check and serves as the per-fire audit-trail entry. Build /tmp/checkpoint.md:
        HEAD-SHA: ${HEAD_SHA}
        HEAD-SHORT: ${SHORT_SHA}
        Date: ${DATE}
        Findings filed this run: ${N_NEW} (${N_NEW_HIGH} HIGH, ${N_NEW_MED} MED)
        Filed: <comma-separated #numbers from /tmp/filed.tsv, or "none (clean checkpoint)">
      Create it then immediately close it. Label `claude-audit` ONLY — no severity/area label, so it never shows up in the actionable backlog (STEP 1c / the consumer query both filter to severity-labeled issues):
        CP=$(gh issue create --title "[claude-audit] checkpoint ${DATE} @${SHORT_SHA}" --label claude-audit --body-file /tmp/checkpoint.md)
        gh issue close "${CP##*/}" --comment "Audit checkpoint — HEAD recorded for skip-check; per-finding issues filed separately. Auto-closed by audit routine."

  4d. Print a summary: the checkpoint URL + every filed issue URL. If any gh write failed, print the full unsent bodies to stdout so the run isn't lost.

CONSTRAINTS:
  - Read-only on repo files. Writing to /tmp/* is fine.
  - Do NOT push branches or open PRs.
  - Allowed writes ONLY: `gh label create` (severity labels, idempotent), `gh issue create` (per-finding issues + the one checkpoint), `gh issue comment` (Related links on issues YOU created this run), `gh issue close` (the checkpoint YOU created this run). Do NOT close or modify any other issue, and do NOT use `gh issue edit` to modify any issue's title/body.
  - ONE issue per finding. Severity is a LABEL (`severity-high`/`severity-medium`), NEVER in the title. Area is both the title prefix and a label.
  - First seen on each finding = the SHORT_SHA / DATE_ONLY of the run that FIRST files it. You only create issues for findings not already in the dedupe pool, so existing issues (and their First seen) are left untouched.
  - Never re-flag anything in the Stop rules block, TODO.md Fixed archive, or the dedupe pool (open+closed per-finding issue titles/files — STEP 1c/3c). Dedup spans open AND closed so a triaged-closed or fixed finding is not re-filed.
  - Never propose cross-position harmonization — feature engineering, not audit.
  - Every run posts a closed checkpoint issue recording HEAD-SHA (even clean 0-finding runs), so STEP 0's skip-check always has a breadcrumb.
