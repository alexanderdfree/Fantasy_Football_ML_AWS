# Shared code audit instructions

These are the provider-neutral instructions for the scheduled code audit routine.
Provider wrappers must read this file, set the runtime variables below, apply their
own tool/worktree rules, and then execute the shared workflow.

## Runtime contract

The wrapper must define these values before executing the workflow:

- `AUDIT_PROVIDER`: human-readable provider name, for example `Claude` or `Codex`.
- `AUDIT_LABEL`: the GitHub issue label this provider files under, for example
  `claude-audit` or `codex-audit`.
- `DEDUPE_AUDIT_LABELS`: all audit labels that must suppress duplicates. This
  repo currently uses `claude-audit codex-audit`.

The provider wrapper owns dispatch mechanics: which local tools are allowed, how
workers are launched, whether parallelism is available, and where the routine is
configured. Do not let provider mechanics change the issue schema, dedupe rules,
verification depth, stop rules, or checkpoint behavior below.

## Mission

Review the Fantasy_Football_ML_AWS repo for bugs and quirks in parallel when the
provider supports it. Use per-area location auditors plus standing cross-cutting
lenses. Each worker emits severity-labeled findings with verbatim evidence.

The orchestrator verifies every cited line, dedupes against open and closed audit
issues from all labels in `DEDUPE_AUDIT_LABELS`, drops anything already covered by
AGENTS.md stop rules or the `todo/fixed-archive.md` fixed archive, consolidates
partial/full duplicates within this run, and files one GitHub issue per surviving
finding under `AUDIT_LABEL`. It also files one closed checkpoint issue under
`AUDIT_LABEL` recording the audited SHA.

Watch especially for artifacts of unfinished merged PRs, semantic merge conflicts
from concurrent PRs blind to each other's changes, cross-position inconsistencies,
training-vs-inference drift, and orphan code under live test coverage.

## Operating boundaries

Working directory is the repo root. Time budget is 2 hours wall-clock.

The routine is read-only on repo files. Writing to `/tmp/*` is allowed. The only
GitHub writes permitted are:

- `gh label create` for audit/severity/regress-risk labels, idempotently.
- `gh issue create` for per-finding issues and this run's checkpoint.
- `gh issue comment` only on issues created in this same run, for related links.
- `gh issue close` only on the checkpoint issue created in this same run.

Do not edit repo files, commit, push, create branches, open PRs, close existing
finding issues, or edit existing issue titles/bodies.

## Issue model

Each surviving finding becomes one GitHub issue:

- **Title**: `[<AUDIT_LABEL>] <area>: <title>`
  - `<area>` is one of `qb`, `rb`, `wr`, `te`, `k`, `dst`, `shared`, `data`,
    `serving`, `batch`, `ci`, `docs`, or `cross-position`.
  - `<title>` is the finding title, under 80 characters.
  - Severity and regress-risk are never in the title; they are labels, so
    reclassification does not mint a new title and break dedupe.
- **Labels**: `<AUDIT_LABEL>`, exactly one severity label, exactly one
  regress-risk label, and the area label when the area label exists.
- **Area mapping**:
  - `src/qb/*` -> `qb`; `src/rb/*` -> `rb`; `src/wr/*` -> `wr`;
    `src/te/*` -> `te`; `src/k/*` -> `k`; `src/dst/*` -> `dst`.
  - `src/shared/*`, `models/*`, `training/*`, `evaluation/*`, `src/config.py`,
    and `src/analysis/*` -> `shared`.
  - `src/data/*` and `src/features/*` -> `data`.
  - `src/serving/*` -> `serving`.
  - `src/batch/*`, `src/tuning/*`, and `src/benchmarking/*` -> `batch`.
  - `src/scripts/*`, `.github/workflows/*`, `.claude/hooks/*`, and
    `.codex/hooks/*` -> `ci`.
  - `AGENTS.md`, `CLAUDE.md`, `CODEX.md`, `README.md`, `SETUP.md`, `TODO.md`,
    `docs/*`, `.claude/skills/*`, `.codex/prompts/*`,
    `.claude/routines/audit/*`, `.codex/automations/audit/*`,
    `routines/audit/*`, and `scripts/bootstrap-codex-local.sh` -> `docs`.
  - Config-invariant/broken-reference lens findings take the area of the
    position they pertain to.

## Severity and model regress-risk

Severity is bug impact:

- `severity-docs`: docs/comment-only findings, including stale prose, dead
  links, wrong documented counts, or wrong checked-in agent instructions.
- `severity-low`: latent, no-op, unreachable, or unlikely-to-fire defects.
- `severity-medium`: plausible correctness bug between low and high impact.
- `severity-high`: likely regressing error metrics, silently producing wrong
  results, security-sensitive, or causing live production problems now.

Model regress-risk is the likelihood that the fix changes model error metrics
or trained artifacts. It is not about serving UI/API/display behavior unless the
fix changes training, feature values, targets, scoring, evaluation, artifacts, or
model inputs.

- `regress-risk-docs`: docs/comment-only fix.
- `regress-risk-low`: model change is not possible.
- `regress-risk-medium`: model change is possible but unlikely, or the issue is
  currently no-op/latent for model metrics.
- `regress-risk-high`: model change is likely; fixes should be urged to rerun
  the relevant pipeline or benchmark.

Docs/comment-only findings always use `severity-docs` and `regress-risk-docs`.

Canonical body:

~~~markdown
- **File**: `<path>:<line>`
- **Severity**: DOCS | LOW | MEDIUM | HIGH
- **Model regress-risk**: docs | low | medium | high
- **Area**: <area>
- **First seen**: @<short_sha> on <YYYY-MM-DD>
- **What**: <2-3 sentences>
- **Why suspect**: <2-3 sentences>
- **Suggested action**: <one sentence>
- **Evidence**: `<verbatim line from the file>`
- **Related**: #<other_issue> [optional, added by comment after creation]

```json
{"schema":"agent-audit/v1","audit_label":"<AUDIT_LABEL>","provider":"<AUDIT_PROVIDER>","file":"<path>","line":<int>,"severity":"DOCS|LOW|MEDIUM|HIGH","regress_risk":"docs|low|medium|high","area":"<area>","category":"<category>","first_seen_sha":"<short_sha>"}
```
~~~

`category` is one of `unfinished_pr`, `merge_conflict`, `orphan_code`,
`invariant`, `broken_reference`, `train_serve_drift`, `wrong_result`,
`security`, or `other`. `file` is the path without `:line`; `line` is an int.

Legacy audit issues may contain `{"schema":"claude-audit/v1",...}`. Treat that
schema as parse-compatible for dedupe and solve-issues consumers, but new issues
must use `agent-audit/v1`.

There are no F-numbers. Findings are identified by GitHub issue number. Dedup
suppresses refiling, so an issue's first-seen SHA/date and `createdAt` provide
the finding age. Only create issues for findings not already filed; never edit or
reset an existing finding issue.

Ensure labels exist at the start of every run:

```bash
gh label create "$AUDIT_LABEL" --color 5319E7 --description "Automated code audit finding/checkpoint" 2>/dev/null || true
gh label create severity-docs --color 6F42C1 --description "Audit: docs/comment-only issue" 2>/dev/null || true
gh label create severity-low --color 0E8A16 --description "Audit: latent / no-op / unlikely-to-fire defect" 2>/dev/null || true
gh label create severity-medium --color FBCA04 --description "Audit: plausible correctness bug between low and high impact" 2>/dev/null || true
gh label create severity-high --color B60205 --description "Audit: likely metric regression / live prod issue / security" 2>/dev/null || true
gh label create regress-risk-docs --color 6F42C1 --description "Audit fix: docs/comment-only, no model change" 2>/dev/null || true
gh label create regress-risk-low --color C2E0C6 --description "Audit fix: model change not possible" 2>/dev/null || true
gh label create regress-risk-medium --color FEF2C0 --description "Audit fix: model change possible but unlikely or currently no-op" 2>/dev/null || true
gh label create regress-risk-high --color D93F0B --description "Audit fix: model change likely; rerun pipeline/benchmark" 2>/dev/null || true
```

## Step 1: Prep

1. Read AGENTS.md stop rules verbatim. Hold them for worker prompts and final
   verification.
2. Run `grep "^### \[FIXED\]" todo/fixed-archive.md` and capture every title
   line. Hold the list.
3. Build the dedupe pool from existing per-finding issues across all labels in
   `DEDUPE_AUDIT_LABELS`. Include open and closed issues carrying a severity
   label; this excludes checkpoint issues because they carry no severity label.

```bash
: > /tmp/known_issues.tsv
for label in $DEDUPE_AUDIT_LABELS; do
  gh issue list --label "$label" --state all --limit 400 --json number,title,labels \
    --jq '.[] | select(any(.labels[]; .name | test("^severity-(docs|low|medium|high)$"))) | "\(.number)\t\(.title)"' \
    >> /tmp/known_issues.tsv
done
sort -u /tmp/known_issues.tsv -o /tmp/known_issues.tsv

: > /tmp/known_files.tsv
cut -f1 /tmp/known_issues.tsv | while read -r N; do
  B=$(gh issue view "$N" --json body --jq '.body' | tr -d '\r')
  F=$(printf '%s\n' "$B" | sed -n '/```json/,/```/p' | sed '1d;$d' \
    | jq -r 'select(.schema=="agent-audit/v1" or .schema=="claude-audit/v1") | .file' 2>/dev/null \
    | head -n1)
  [ -z "$F" ] && F=$(printf '%s\n' "$B" | grep -m1 -oE '`[^`]+:[0-9]+`' | tr -d '`' | sed -E 's/:[0-9]+$//')
  printf '%s\t%s\n' "$N" "$F" >> /tmp/known_files.tsv
done
```

Dedupe key: area + cited file path + at least two distinctive title keywords.
Area comes from the title prefix (`[claude-audit] <area>:` or
`[codex-audit] <area>:`); `/tmp/known_files.tsv` carries the file.

4. Compute per-area yield from this provider's own closed audit issues. Dedup
   spans both providers, but yield/history is provider-local by `AUDIT_LABEL`.
   A finding triaged as noise carries the `leave` label. A genuinely fixed
   finding closes without `leave`.

```bash
gh issue list --label "$AUDIT_LABEL" --state closed --limit 800 --json title,labels > /tmp/closed.json
for area in qb rb wr te k dst shared data serving batch ci docs; do
  sample=$(jq --arg label "$AUDIT_LABEL" --arg a "$area" '[.[] | select(.title|startswith("["+$label+"] "+$a+":"))] | length' /tmp/closed.json)
  noise=$(jq --arg label "$AUDIT_LABEL" --arg a "$area" '[.[] | select(.title|startswith("["+$label+"] "+$a+":")) | select(any(.labels[];.name=="leave"))] | length' /tmp/closed.json)
  printf '%s\t%s\t%s\n' "$area" "$sample" "$noise"
done > /tmp/area_yield.tsv
```

Until the `leave` backlog accrues, most areas read sample<5, so the min-sample
guard keeps their base tier.

## Step 2: Fanout

Use the provider's available parallelism. Spawn all workers in one batch if the
provider can do so. If not, run Layer A and then Layer B, or run the workers
sequentially while preserving the same scopes and output contract.

Layer A location-auditor scopes:

1. QB auditor: `src/qb/`, `tests/qb/`
2. RB auditor: `src/rb/`, `tests/rb/`
3. WR auditor: `src/wr/`, `tests/wr/`
4. TE auditor: `src/te/`, `tests/te/`
5. K auditor: `src/k/`, `tests/k/`
6. DST auditor: `src/dst/`, `tests/dst/`
7. Shared auditor: `src/shared/`, `tests/shared/`
8. Data+features auditor: `src/data/`, `src/features/`
9. Serving auditor: `src/serving/`, `tests/test_app*.py`
10. Batch+CI auditor: `src/batch/`, `.github/workflows/`, `.claude/hooks/`,
    `.codex/hooks/`
11. Docs/tooling consistency auditor: `AGENTS.md`, `CLAUDE.md`, `CODEX.md`,
    `README.md`, `SETUP.md`, `TODO.md`, `docs/`, `.claude/skills/`,
    `.codex/prompts/`, `.claude/routines/audit/`,
    `.codex/automations/audit/`, `routines/audit/`,
    `scripts/bootstrap-codex-local.sh`

Layer B standing cross-cutting lenses:

- L1 Train/serve parity: `src/serving/` vs `src/shared/pipeline.py` and each
  `src/{pos}/features.py`. Only an actual feature/scaler/merge/expression
  difference is a finding.
- L2 Cross-position consistency: `src/qb|rb|wr|te|k|dst/` side by side. Only
  unintended divergence is a finding; deliberate tuning/whitelist differences
  are not findings.
- L3 Recent-PR archaeology: last about 30 commits and recently merged PRs. Look
  for unfinished merged-PR artifacts and semantic merge conflicts.
- L4 Config-invariant + broken-reference: all six `src/{pos}/config.py` files
  and their consumers. Focus on within-position invariant violations and
  broken references to keys/values absent from that same position config.
- L5 Agent tooling parity: `AGENTS.md`, `CLAUDE.md`, `CODEX.md`,
  `.claude/settings.json`, `.claude/hooks/`, `.claude/skills/`,
  `.claude/routines/audit/`, `.codex/hooks.json`, `.codex/hooks/`,
  `.codex/prompts/`, `.codex/automations/audit/`, `routines/audit/`,
  `scripts/bootstrap-codex-local.sh`, `scripts/bootstrap-claude-wsl.sh`,
  `scripts/claude-memory-sync.sh`, `scripts/codex-memory-sync.sh`,
  `scripts/agent-memory-sync.sh`, and `SETUP.md`. Report substantive provider
  parity breaks promised by docs; do not report intentional provider differences.

Budget is soft pacing only; the 2-hour run wall-clock is the only hard limit.
Set each worker's budget from a base leverage tier, adjusted by `/tmp/area_yield.tsv`:

- HEAVY 30 minutes: shared, serving, data, L1-L5.
- MEDIUM 20 minutes: qb, rb, wr, te, k, dst, batch/CI.
- LIGHT 10 minutes: docs.
- If sample >= 5 and yield >= 0.60, bump up one tier.
- If sample >= 5 and yield < 0.30, drop one tier.
- shared, serving, and L1-L5 never drop below HEAVY.
- sample < 5 or no data keeps the base tier.

Before spawning, print the chosen worker -> tier, budget, yield, sample table.

## Per-worker prompt template

Substitute `{ROLE_ID}`, `{SCOPE}`, `{FOCUS}`, and `{BUDGET}`.

```text
ROLE: {ROLE_ID}. Scope: {SCOPE}. {BUDGET}-minute soft budget. The run's
2-hour wall-clock is the sole hard limit.

PRIMARY FOCUS:
(a) Artifacts of unfinished-but-merged PRs: dead code, half-renamed symbols,
    orphan imports, commented-out blocks, TODO/FIXME, feature-list/test-fixture
    mismatches.
(b) Semantic merge conflicts from concurrent PRs blind to each other's changes:
    check `git log -p -n 30 -- <file>` for competing recent edits.
(c) Orphan code under live test coverage: production-side functions with zero
    callers in `src/` but exercised by tests.
(d) {FOCUS}
(e) Anything that would silently produce wrong results.

NOT FINDINGS:
- "Position X has feature F that position Y doesn't" -> per-position whitelists
  are intentional.
- "Position X's loss weight value differs from Y's" -> only the within-position
  ratio invariant matters.
- "Position X uses head_hidden_overrides, Y doesn't" -> per-position tuning.
- "Add F to all positions for parity" / "harmonize X across positions" ->
  feature engineering.
- "Position X has CONFIG_TINY, Y doesn't" -> optional convention.
- "Position X's NN hidden dim differs from Y's" -> per-position tuning.
- A doc/comment has only a stale `file:lineN` or `file:lines X-Y` citation while
  the cited target is otherwise correct -> cosmetic line-number drift.
- Any finding that would change a design choice, feature selection, model
  architecture/hyperparameters, scoring, or model accuracy as tuning/judgment
  rather than fixing a defect -> DROP unless it is a clear, non-controversial
  correctness bug.

STILL REPORT substantive doc errors: wrong module/symbol attribution, a
documented feature/decision/count that does not exist or is wrong, a stated
invariant the code violates, a dead cross-ref to a deleted file, or a config
key/value reference that is not defined.

When in doubt, drop the finding.

STOP RULES:
<inline AGENTS.md stop rules + every todo/fixed-archive.md FIXED title from Step 1>

SELF-VERIFY every candidate before emitting it:
1. Re-open the cited file at the cited line. Confirm evidence_quote is
   verbatim-present, whitespace-normalized, at or within +/-3 lines. If absent,
   DROP.
2. Actively re-confirm the defect still holds via the cheapest decisive check:
   - orphan_code: `grep -rn "<symbol>" src/`; if any production caller exists
     outside tests, DROP.
   - broken_reference: open the referenced config/key/value; confirm it is not
     defined. If it resolves, DROP.
   - unfinished_pr / merge_conflict: `git log -p -n 30 -- <file>`; confirm the
     issue is still on HEAD and not completed by a later commit. If reconciled,
     DROP.
   - train_serve_drift: read both callsites; confirm expressions actually
     differ. If they match, DROP.
   - invariant: re-read cited values; confirm the within-position invariant is
     violated. If satisfied, DROP.
   - wrong_result / security / other: name the concrete trigger reaching the
     wrong/unsafe path. If you cannot, DROP.
3. Only survivors may be emitted. When in doubt, DROP.

SEVERITY:
DOCS = docs/comment-only findings. LOW = latent, no-op, unreachable, or
unlikely-to-fire defects. MEDIUM = plausible correctness bug between low and
high impact. HIGH = likely regressing error metrics, silently producing wrong
results, security-sensitive, or causing live production problems now.

MODEL REGRESS-RISK:
docs = docs/comment-only fix. low = model change is not possible. medium =
model change is possible but unlikely, or currently no-op/latent for model
metrics. high = model change is likely; fixes should be urged to rerun the
relevant pipeline or benchmark. This is about MSE/MAE/FP-MAE and trained
artifacts, not serving UI/API/display behavior by itself.

OUTPUT: JSON array only. Each object:
{"file":"<path>","line":<int>,"severity":"DOCS|LOW|MEDIUM|HIGH","regress_risk":"docs|low|medium|high","category":"unfinished_pr|merge_conflict|orphan_code|invariant|broken_reference|train_serve_drift|wrong_result|security|other","title":"<<80 chars>","what":"<2-3 sentences>","why_suspect":"<2-3 sentences>","suggested_action":"<one sentence>","evidence_quote":"<verbatim line from file>","verification":"<one-line note of what you checked + result>"}

Workers do not create issues and do not assign issue numbers.
```

## Step 3: Verify new findings

For each new worker finding, the orchestrator re-verifies as a backstop:

1. Read file at cited line. Confirm `evidence_quote` matches,
   whitespace-normalized. Drop on mismatch.
2. Grep AGENTS.md and `todo/fixed-archive.md` for 2-3 distinctive title
   keywords. Drop if matched.
3. Dedupe against `/tmp/known_issues.tsv` and `/tmp/known_files.tsv`: duplicate
   when an existing issue has the same area, same cited file, and at least two
   shared distinctive title keywords. The pool spans open and closed
   `claude-audit` and `codex-audit` per-finding issues.
4. For unfinished-PR or semantic-merge-conflict claims, run
   `git log -p -n 20 -- <file>` and confirm consistency. Drop if completed
   elsewhere.
5. Drop anything matching NOT FINDINGS patterns.

Hold `/tmp/new_findings.jsonl` with survivors and counters for total plus
DOCS/LOW/MEDIUM/HIGH severities and docs/low/medium/high regress-risk values.

## Step 3b: Consolidate duplicates within this run

Operate only on new findings from this run. Use tentative local IDs `t1..tN`.

- PASS 1 full duplicates: same file, same line, and whitespace-normalized
  evidence match. Keep one.
- PASS 2 same-file partial duplicates: lines within +/-10, at least two
  distinctive title keywords, and semantically same `what`/`why_suspect`. Keep
  one and merge only unique one-sentence fragments.
- PASS 3 cross-file related: different files, at least three shared title
  keywords, and one finding references the other's file/symbol. Keep both and
  record pairing in `/tmp/related.tsv` for post-create comments.

Conservative bias: when in doubt, do not merge. Hard cap: never drop more than
30% of new findings via consolidation; if exceeded, skip consolidation and file
unconsolidated.

Output `/tmp/consolidated_new.jsonl` with tentative ID + resolved area, and
`/tmp/related.tsv` with `tA<TAB>tB<TAB>reason`.

## Step 4: File issues

Set run metadata:

```bash
HEAD_SHA=$(git rev-parse HEAD)
SHORT_SHA=$(git rev-parse --short HEAD)
DATE=$(date -u +"%Y-%m-%d %H:%M UTC")
DATE_ONLY=$(date -u +"%Y-%m-%d")
```

All first-seen values are `SHORT_SHA` on `DATE_ONLY`.

For each finding in `/tmp/consolidated_new.jsonl`, build the canonical body and
append the machine-readable block:

```bash
{
  printf '\n```json\n'
  jq -nc \
    --arg schema "agent-audit/v1" \
    --arg audit_label "$AUDIT_LABEL" \
    --arg provider "$AUDIT_PROVIDER" \
    --arg file "$file" --argjson line "$line" \
    --arg sev "$severity" --arg area "$area" \
    --arg regress_risk "$regress_risk" \
    --arg cat "${category:-other}" --arg sha "$SHORT_SHA" \
    '{schema:$schema,audit_label:$audit_label,provider:$provider,file:$file,line:$line,severity:$sev,regress_risk:$regress_risk,area:$area,category:$cat,first_seen_sha:$sha}'
  printf '\n```\n'
} >> /tmp/body.md
```

Create the issue:

```bash
case "$severity" in
  DOCS) SEV_LABEL=severity-docs ;;
  LOW) SEV_LABEL=severity-low ;;
  MEDIUM|MED) SEV_LABEL=severity-medium ;;
  HIGH) SEV_LABEL=severity-high ;;
  *) SEV_LABEL=severity-medium ;;
esac
case "$regress_risk" in
  docs|low|medium|high) RISK_LABEL="regress-risk-$regress_risk" ;;
  *) RISK_LABEL=regress-risk-medium ;;
esac
URL=$(gh issue create \
  --title "[$AUDIT_LABEL] ${area}: ${title}" \
  --label "$AUDIT_LABEL" --label "$SEV_LABEL" --label "$RISK_LABEL" --label "$area" \
  --body-file /tmp/body.md)
NUM=${URL##*/}
printf '%s\t%s\t%s\n' "$tID" "$NUM" "$URL" >> /tmp/filed.tsv
```

If `gh issue create` fails because of an unexpected area label, retry once
without the area label. If it still fails, print the full body to stdout so the
finding is not lost.

For each pairing in `/tmp/related.tsv`, resolve issue numbers via
`/tmp/filed.tsv` and add symmetric comments on issues created in this run:

```bash
gh issue comment "$NUM_A" --body "Related: #${NUM_B} - ${reason}"
gh issue comment "$NUM_B" --body "Related: #${NUM_A} - ${reason}"
```

Always create a checkpoint issue, including clean runs:

```bash
cat > /tmp/checkpoint.md <<EOF
HEAD-SHA: ${HEAD_SHA}
HEAD-SHORT: ${SHORT_SHA}
Date: ${DATE}
Provider: ${AUDIT_PROVIDER}
Audit label: ${AUDIT_LABEL}
Findings filed this run: ${N_NEW}
Severity counts: ${N_NEW_DOCS} DOCS, ${N_NEW_LOW} LOW, ${N_NEW_MEDIUM} MEDIUM, ${N_NEW_HIGH} HIGH
Regress-risk counts: ${N_RISK_DOCS} docs, ${N_RISK_LOW} low, ${N_RISK_MEDIUM} medium, ${N_RISK_HIGH} high
Filed: <comma-separated #numbers from /tmp/filed.tsv, or "none (clean checkpoint)">
EOF

CP=$(gh issue create --title "[$AUDIT_LABEL] checkpoint ${DATE} @${SHORT_SHA}" --label "$AUDIT_LABEL" --body-file /tmp/checkpoint.md)
gh issue close "${CP##*/}" --comment "Audit checkpoint - HEAD + finding counts recorded for the audit trail; per-finding issues filed separately. Auto-closed by audit routine."
```

Print a summary with the checkpoint URL and every filed issue URL. If any GitHub
write failed, print the unsent bodies to stdout.

## Final constraints

- One issue per finding.
- Severity and regress-risk are labels, never title tokens.
- Area is both the title prefix and an area label where that label exists.
- First seen is the SHA/date of the run that first files the finding.
- Existing issues are left untouched.
- Dedup spans open and closed severity-labeled `claude-audit` and `codex-audit`
  issues, so triaged-closed or fixed findings are not re-filed.
- Never re-flag anything in AGENTS.md stop rules or `todo/fixed-archive.md`.
- Never propose cross-position harmonization; that is feature engineering.
- Never file design, tuning, or accuracy-judgment changes unless they are clear,
  non-controversial correctness bugs.
- Every run posts one closed checkpoint issue under its own `AUDIT_LABEL`.
