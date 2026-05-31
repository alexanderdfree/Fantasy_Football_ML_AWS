=== MISSION ===
Review my code in parallel for bugs/quirks — spawn a panel of Opus subagents — per-area location auditors plus standing cross-cutting lenses — each emitting HIGH/MED-only findings with verbatim evidence. The orchestrator verifies each cited line, dedupes against open and closed GitHub issues, drops anything already in CLAUDE.md Stop rules or the `todo/fixed-archive.md` Fixed archive, consolidates partial/full duplicates within this run's worker output, and files ONE GITHUB ISSUE PER SURVIVING FINDING — labeled by severity (`severity-high`/`severity-medium`) and area — plus one closed checkpoint issue recording the audited SHA. Watch for artifacts of unfinished (yet merged) PRs, semantic merge conflicts from concurrent PRs blind to each other's changes, cross-position inconsistencies, training-vs-inference drift, and orphan code under live test coverage.

=== IMPLEMENTATION ===
You are the orchestrator for a scheduled code-audit run on the Fantasy_Football_ML_AWS repo. Working dir is the repo root. Time budget: 2 hours wall-clock. You are READ-ONLY on repo files. The ONLY write actions permitted are `gh label create` (the severity labels, idempotent), `gh issue create` (per-finding issues + the checkpoint), `gh issue comment` (only on issues YOU create in THIS same run), and `gh issue close` (only on the checkpoint issue YOU create in THIS run). Writing to /tmp/* is fine.

=== ISSUE MODEL (one issue per finding) ===
Each surviving finding becomes ONE GitHub issue:
  - **Title**: `[claude-audit] <area>: <title>` — `<area>` ∈ qb|rb|wr|te|k|dst|shared|data|serving|batch|ci|docs|cross-position; `<title>` is the finding title (<80 chars). Severity is NEVER in the title — it is a LABEL, so a HIGH↔MED reclassification doesn't mint a "new" title and break dedup.
  - **Labels**: `claude-audit`, the severity label (`severity-high` or `severity-medium`), and the area label.
  - **Area** is derived from the finding's `file` path / the worker scope that produced it: `src/qb/*`→qb … `src/dst/*`→dst, `src/shared|models|training|evaluation/*`→shared, `src/data|features/*`→data, `src/serving/*`→serving, `src/batch/*`→batch, `.github/workflows|.claude/hooks|.codex/hooks/*`→ci, and `AGENTS.md|CLAUDE.md|CODEX.md|README.md|SETUP.md|TODO.md|docs/*|.claude/skills/*|.codex/prompts/*|.claude/routines/audit/*|scripts/bootstrap-codex-local.sh`→docs. The config-invariant/broken-ref lens (L4) findings take the area of the position they pertain to.
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
        - **Machine-readable finding** (appended verbatim at the END of the body; deterministically parseable — STEP 1c dedup and the solve-issues consumer read THIS; prose fields above are for humans + backward-compat):
          ```json
          {"schema":"claude-audit/v1","file":"<path>","line":<int>,"severity":"HIGH|MED","area":"<area>","category":"<category>","first_seen_sha":"<short_sha>"}
          ```
          category ∈ unfinished_pr | merge_conflict | orphan_code | invariant | broken_reference | train_serve_drift | wrong_result | security | other. `file` is the path WITHOUT :line; `line` is the int. area/severity mirror the labels; first_seen_sha = the SHORT_SHA of the filing run.

There are NO F-numbers. Findings are identified by their GitHub issue number. Because dedup (STEP 1c / 3c) suppresses refiling, each issue persists across fires — so its **First seen** SHA/date and `createdAt` give the finding's true age. You only ever CREATE new issues for findings not already filed; never edit or reset an existing issue.

Ensure the severity labels exist (idempotent — safe to run every fire; area labels already exist in the repo):
    gh label create severity-high   --color B60205 --description "Audit: wrong result / silent loss / security / benchmark-changing" 2>/dev/null || true
    gh label create severity-medium --color FBCA04 --description "Audit: unfinished-PR artifact / invariant / drift / orphan code" 2>/dev/null || true

STEP 1 — Prep (sequential, ~2 min):
  1a. Read CLAUDE.md "Stop rules" section verbatim. Hold it.
  1b. grep "^### \[FIXED\]" TODO.md — capture every title line. Hold the list.
  1c. Build the dedupe pool from existing PER-FINDING issues — open AND closed, those carrying a `severity-*` label (this naturally EXCLUDES checkpoint issues, which have no severity label):
        gh issue list --label claude-audit --state all --limit 400 --json number,title,labels \
          --jq '.[] | select(any(.labels[]; .name=="severity-high" or .name=="severity-medium")) | "\(.number)\t\(.title)"' > /tmp/known_issues.tsv
        # For file-aware dedup, capture each known issue's cited File: path (strip the :line):
        : > /tmp/known_files.tsv
        cut -f1 /tmp/known_issues.tsv | while read N; do
          B=$(gh issue view "$N" --json body --jq '.body' | tr -d '\r')
          # Prefer the v1 machine-readable block; .file is the path WITHOUT :line — exactly what dedup keys on.
          F=$(printf '%s\n' "$B" | sed -n '/```json/,/```/p' | sed '1d;$d' | jq -r 'select(.schema=="claude-audit/v1") | .file' 2>/dev/null | head -n1)
          # Legacy fallback for issues filed before the block existed:
          [ -z "$F" ] && F=$(printf '%s\n' "$B" | grep -m1 -oE '`[^`]+:[0-9]+`' | tr -d '`' | sed -E 's/:[0-9]+$//')
          printf '%s\t%s\n' "$N" "$F" >> /tmp/known_files.tsv
        done
      Dedupe KEY = (area + cited file-path + ≥2 distinctive title keywords). The title prefix `[claude-audit] <area>:` carries area; `/tmp/known_files.tsv` carries the file.

  1d. Compute per-area YIELD (real-vs-noise track record) to weight worker budgets in STEP 2. A finding triaged NOISE carries the `leave` label (applied by the solve-issues skill, closed "not planned"); a genuinely-fixed finding closes WITHOUT `leave` (as "completed"). Bucket CLOSED audit issues by the `[claude-audit] <area>:` title prefix (this excludes checkpoints + cross-position):
        gh issue list --label claude-audit --state closed --limit 800 --json title,labels > /tmp/closed.json
        for area in qb rb wr te k dst shared data serving batch ci docs; do
          sample=$(jq --arg a "$area" '[.[] | select(.title|startswith("[claude-audit] "+$a+":"))] | length' /tmp/closed.json)
          noise=$(jq --arg a "$area" '[.[] | select(.title|startswith("[claude-audit] "+$a+":")) | select(any(.labels[];.name=="leave"))] | length' /tmp/closed.json)
          printf '%s\t%s\t%s\n' "$area" "$sample" "$noise"
        done > /tmp/area_yield.tsv   # area<TAB>sample<TAB>noise; yield = (sample-noise)/sample
      Until the `leave` backlog accrues, most areas read sample<5 → STEP 2's min-sample guard keeps their base tier (correct cold-start). Historical LEAVE issues closed before this signal existed lack the label and miscount as "real" until re-triaged — the min-sample guard bounds that.

STEP 2 — Fanout (parallel): two layers, ALL workers spawned IN A SINGLE MESSAGE (Agent tool calls run concurrently), each Agent call model="opus". LAYER A = the per-area location auditors #1–#11 below; LAYER B = the five standing cross-cutting lenses L1–L5 below (always spawned, every fire). Total ≈ 16. If the platform caps concurrent spawns below that, spawn LAYER A then LAYER B in two back-to-back messages (each concurrent within itself). Each worker's {BUDGET} comes from the tier table after the lens list. LAYER A location-auditor scopes:
    #1  QB auditor          — src/qb/, tests/qb/
    #2  RB auditor          — src/rb/, tests/rb/
    #3  WR auditor          — src/wr/, tests/wr/
    #4  TE auditor          — src/te/, tests/te/
    #5  K auditor           — src/k/, tests/k/
    #6  DST auditor         — src/dst/, tests/dst/
    #7  Shared auditor      — src/shared/, src/models/, src/training/, src/evaluation/, tests/shared/
    #8  Data+features       — src/data/, src/features/
    #9  Serving auditor     — src/serving/, tests/serving/ (focus: serving-internal correctness — request/feature handling, scaler & artifact loading; the train/serve PARITY comparison is lens L1's job)
    #10 Batch+CI auditor    — src/batch/, .github/workflows/, .claude/hooks/, .codex/hooks/
    #11 Docs/tooling consistency — AGENTS.md, CLAUDE.md, CODEX.md, README.md, SETUP.md, TODO.md, docs/, .claude/skills/, .codex/prompts/, .claude/routines/audit/, scripts/bootstrap-codex-local.sh (FOCUS: substantive doc-vs-code mismatches — wrong module/symbol attribution, a documented feature/decision/count that doesn't exist or is miscounted, a stated invariant the code violates, a dead cross-ref to a deleted file, or checked-in agent-tooling instructions that no longer match the files/hooks/prompts they describe. Do NOT report a doc/comment whose ONLY error is a stale `file:line`/`file:lines X-Y` citation when the cited target is otherwise correct — line numbers drift as code is inserted above; that is cosmetic, not a finding.)

  LAYER B — standing cross-cutting lenses (by failure-MODE, not location; ALWAYS spawned, every fire):
    L1  Train/serve parity — SCOPE: src/serving/ vs src/shared/pipeline.py + each src/{pos}/features.py. FOCUS: a feature / scaler / merge (weather, Vegas, red-zone, clip) present in the TRAINING feature-build but absent or different in the SERVING path (or vice-versa). Read BOTH callsites; only an actual expression/column DIFFERENCE is a finding (category train_serve_drift).
    L2  Cross-position consistency — SCOPE: src/qb|rb|wr|te|k|dst/ side-by-side. FOCUS: ONLY *unintended* divergence — a fix / rename / guard applied to 5 of 6 positions but missing in the 6th with NO per-position rationale. Intentional per-position differences (features, hyperparams, loss weights, NN dims, CONFIG_TINY) are NOT findings (see NOT FINDINGS). This is the most false-positive-prone lens: when a difference COULD be deliberate tuning/whitelist, DROP.
    L3  Recent-PR archaeology — SCOPE: the last ~30 commits / recently-merged PRs, tree-wide (`git log -p -n 30`, `gh pr list --state merged --limit 20`). FOCUS: unfinished-but-merged-PR artifacts and semantic merge conflicts between concurrent PRs blind to each other (categories unfinished_pr, merge_conflict).
    L4  Config-invariant + broken-reference — SCOPE: all six src/{pos}/config.py + their consumers. FOCUS: WITHIN-position invariant violations (LOSS_WEIGHTS ≈ 2.0/HUBER_DELTAS within one position's config; target-naming consistent within a position) AND broken-reference drift (a file in position X references a key/value X's own config.py doesn't define). DOES NOT FLAG per-position differences ACROSS positions — those are by design.
    L5  Agent tooling parity — SCOPE: AGENTS.md, CLAUDE.md, CODEX.md, .claude/settings.json, .claude/hooks/, .claude/skills/, .claude/routines/audit/, .codex/hooks.json, .codex/hooks/, .codex/prompts/, scripts/bootstrap-codex-local.sh, scripts/bootstrap-claude-wsl.sh, scripts/claude-memory-sync.sh, SETUP.md. FOCUS: Claude/Codex equivalents promised in AGENTS.md or CODEX.md that are missing, stale, or point to the wrong provider's entrypoint; hook wrappers whose delegated messages no longer match the caller; prompt/skill workflows whose Codex version omits required decisions from the Claude source. NOT FINDINGS: intentional platform differences documented in AGENTS.md/CODEX.md, including Codex SessionStart being context-only, Codex memory sync being N/A by design, and Codex having no scheduled routine equivalent while consuming the `claude-audit` backlog.

  BUDGET — set each worker's {BUDGET} (minutes; SOFT pacing only — the run's 2h wall-clock is the sole HARD limit) from a base leverage tier, adjusted by per-area yield (/tmp/area_yield.tsv from STEP 1d; yield = (sample−noise)/sample):
    Base tier: HEAVY 30 → shared, serving, data, and all five lenses L1–L5. MEDIUM 20 → qb rb wr te k dst, batch/CI (#10). LIGHT 10 → docs.
    Yield adjustment (ONLY for an area with sample ≥ 5): yield ≥ 0.60 → bump up one tier (10→20→30); yield < 0.30 → drop one tier (30→20→10). FLOOR: shared, serving, and L1–L5 never drop below HEAVY. sample < 5 (or no data) → keep base tier.
    Before spawning, print the chosen {worker → tier, budget, yield, sample} table to stdout — no silent weighting.

  Per-worker template (substitute {ROLE_ID}, {SCOPE}, {FOCUS}, {BUDGET}; {ROLE_ID} = the worker's name, e.g. "QB auditor" or "Lens L1 — train/serve parity"):
    ROLE: {ROLE_ID}. Scope: {SCOPE}. {BUDGET}-minute soft budget (pacing only; the run's 2h wall-clock is the sole hard limit).

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
      - "Doc/comment cites `file:lineN` or `file:lines X-Y` but the entity is actually at a different line/range" → PURELY COSMETIC. Source line numbers drift whenever code is inserted above; a stale line-NUMBER or line-RANGE in prose/docstrings is NOT a finding. DROP it.
        STILL REPORT (substantive doc errors, NOT line-number drift): wrong MODULE/symbol attribution (doc says feature X is in module A but it's in B); a documented feature/decision/COUNT that doesn't exist or is wrong (e.g. "ADR has D1–D15" when a real D16 exists); a stated INVARIANT the code violates; a dead cross-ref to a DELETED file; a config KEY/VALUE a file references that isn't defined (that is broken_reference, lens L4). The line-number being off is only noise when the cited TARGET is otherwise correct.
      - ANY finding that would change a design choice, feature selection, model architecture/hyperparameters, scoring, or otherwise alter model accuracy as a matter of tuning or judgment rather than fixing a defect → DROP. UNLESS IT IS A CLEAR, NON-CONTROVERSIAL CORRECTNESS BUG.
      When in doubt, drop the finding.

    STOP RULES — drop anything overlapping:
    <inline CLAUDE.md Stop rules + every TODO.md FIXED title from STEP 1>

    SELF-VERIFY (mandatory — for EVERY candidate BEFORE you emit it; first-line filter, not the orchestrator's job):
      1. Re-open the cited file at the cited line. Confirm evidence_quote is VERBATIM-present (whitespace-normalized: collapse space/tab runs, ignore leading/trailing indent). If absent at (or within ±3 lines of) the cited line, DROP.
      2. Actively re-confirm the defect STILL HOLDS via the cheapest decisive check for its category — do not infer from the quote alone:
           - orphan_code: `grep -rn "<symbol>" src/` — if any prod caller exists outside tests, DROP.
           - broken_reference: open the referenced key/value; confirm it truly is NOT defined (`grep -rn "<KEY>" src/<pos>/config.py`). If it resolves, DROP.
           - unfinished_pr / merge_conflict: `git log -p -n 30 -- <file>`; confirm the half-finished/competing edit is still on HEAD and not completed by a later commit. If reconciled, DROP.
           - train_serve_drift: read both callsites; confirm train-path and serve-path expressions actually differ. If they match, DROP.
           - invariant: re-read the cited values; confirm the WITHIN-position invariant is actually violated (a per-position difference is NOT a finding). If satisfied, DROP.
           - wrong_result / security / other: name the concrete trigger (input/condition) reaching the wrong/unsafe path; if you cannot, DROP.
      3. Only survivors of both checks may be emitted. When in doubt, DROP.
      The orchestrator (STEP 3) re-verifies as a BACKSTOP; self-verify is your first-line filter and must not be skipped.

    SEVERITY: HIGH (wrong result / silent loss / security / benchmark-changing — where "benchmark-changing" means a clear correctness bug whose fix happens to move the metric, NOT a tuning/design change; see NOT FINDINGS) or MED (unfinished-PR artifact / within-position invariant violation / broken-reference drift / semantic merge conflict / orphan code under live test coverage). NO LOW.

    OUTPUT: JSON array only. Each: {"file": "<path>", "line": <int>, "severity": "HIGH"|"MED", "category": "unfinished_pr|merge_conflict|orphan_code|invariant|broken_reference|train_serve_drift|wrong_result|security|other", "title": "<<80 chars>", "what": "<2-3 sentences>", "why_suspect": "<2-3 sentences>", "suggested_action": "<one sentence>", "evidence_quote": "<verbatim line from file>", "verification": "<one-line note of what you checked + result>"}
    Workers do NOT create issues and do NOT assign issue numbers — the orchestrator does.
    Pick category to match the defect type (drives downstream triage + dedup): unfinished-PR artifact→unfinished_pr; semantic merge conflict→merge_conflict; orphan-code-under-test→orphan_code; within-position invariant→invariant; missing config key/value reference→broken_reference; training-vs-inference mismatch→train_serve_drift; silently-wrong output→wrong_result; security→security; else→other.

STEP 3 — Verify new findings (sequential, ~5 min):
  For each new finding from each worker:
    (Workers already self-verified per the per-worker SELF-VERIFY block; STEP 3 is the orchestrator BACKSTOP — re-run the cheap checks below; do not trust the worker's verification note blindly.)
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
        category=${category:-other}
        # Append the machine-readable block to the prose body already in /tmp/body.md:
        {
          printf '\n```json\n'
          jq -nc \
            --arg file "$file" --argjson line "$line" \
            --arg sev "$severity" --arg area "$area" \
            --arg cat "$category" --arg sha "$SHORT_SHA" \
            '{schema:"claude-audit/v1",file:$file,line:$line,severity:$sev,area:$area,category:$cat,first_seen_sha:$sha}'
          printf '\n```\n'
        } >> /tmp/body.md
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

  4c. CHECKPOINT ISSUE (ALWAYS — every run, including clean N_NEW==0 runs). The per-fire audit-trail entry — records the audited SHA + findings filed this run. Build /tmp/checkpoint.md:
        HEAD-SHA: ${HEAD_SHA}
        HEAD-SHORT: ${SHORT_SHA}
        Date: ${DATE}
        Findings filed this run: ${N_NEW} (${N_NEW_HIGH} HIGH, ${N_NEW_MED} MED)
        Filed: <comma-separated #numbers from /tmp/filed.tsv, or "none (clean checkpoint)">
      Create it then immediately close it. Label `claude-audit` ONLY — no severity/area label, so it never shows up in the actionable backlog (STEP 1c / the consumer query both filter to severity-labeled issues):
        CP=$(gh issue create --title "[claude-audit] checkpoint ${DATE} @${SHORT_SHA}" --label claude-audit --body-file /tmp/checkpoint.md)
        gh issue close "${CP##*/}" --comment "Audit checkpoint — HEAD + finding counts recorded for the audit trail; per-finding issues filed separately. Auto-closed by audit routine."

  4d. Print a summary: the checkpoint URL + every filed issue URL. If any gh write failed, print the full unsent bodies to stdout so the run isn't lost.

CONSTRAINTS:
  - Read-only on repo files. Writing to /tmp/* is fine.
  - Do NOT push branches or open PRs.
  - Allowed writes ONLY: `gh label create` (severity labels, idempotent), `gh issue create` (per-finding issues + the one checkpoint), `gh issue comment` (Related links on issues YOU created this run), `gh issue close` (the checkpoint YOU created this run). Do NOT close or modify any other issue, and do NOT use `gh issue edit` to modify any issue's title/body.
  - ONE issue per finding. Severity is a LABEL (`severity-high`/`severity-medium`), NEVER in the title. Area is both the title prefix and a label.
  - First seen on each finding = the SHORT_SHA / DATE_ONLY of the run that FIRST files it. You only create issues for findings not already in the dedupe pool, so existing issues (and their First seen) are left untouched.
  - Never re-flag anything in the Stop rules block, the `todo/fixed-archive.md` Fixed archive, or the dedupe pool (open+closed per-finding issue titles/files — STEP 1c/3c). Dedup spans open AND closed so a triaged-closed or fixed finding is not re-filed.
  - Never propose cross-position harmonization — feature engineering, not audit.
  - Never file design / tuning / accuracy-judgment changes — only clear, non-controversial correctness bugs (the blanket NOT FINDINGS rule).
  - Every run posts a closed checkpoint issue recording HEAD-SHA (even clean 0-finding runs) as the per-fire audit trail.
