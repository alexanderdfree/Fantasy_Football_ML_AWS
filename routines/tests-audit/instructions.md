# Shared tests-audit instructions

These are the provider-neutral instructions for the scheduled **tests-audit**
routine — a deeper, test-suite-scoped sibling of the general code audit
(`routines/audit/instructions.md`). Provider wrappers must read this file, set
the runtime variables below, apply their own tool/worktree rules, and then
execute the shared workflow.

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

This routine **reuses the same `claude-audit`/`codex-audit` labels, the same
`agent-audit/v1` schema, and the same severity/regress-risk/area labels** as the
general audit. It does not introduce its own label. Its findings therefore
dedupe against — and flow into the same `solve-issues` backlog as — the general
audit's findings.

## Mission

Review the Fantasy_Football_ML_AWS **test suite** for defects in parallel when
the provider supports it. Use per-shard location auditors plus standing
test-specific cross-cutting lenses. Each worker emits severity-labeled findings
with verbatim evidence.

This is a **deeper, scoped pass than the general audit**, not a replacement for
it. The general audit (`routines/audit/instructions.md`) sweeps `tests/{pos}/`
shallowly alongside production code; this routine goes deep on the test suite
itself. Because both file under the same `claude-audit`/`codex-audit` labels and
this routine dedupes against the **same shared pool**, any finding the general
audit (or a prior tests-audit / infrastructure-audit run) already filed is
suppressed automatically. The value here is depth in the test domain, not
breadth.

The orchestrator verifies every cited line, dedupes against open and closed
audit issues from all labels in `DEDUPE_AUDIT_LABELS`, drops anything already
covered by AGENTS.md stop rules or the `todo/fixed-archive.md` fixed archive,
consolidates partial/full duplicates within this run, and files one GitHub issue
per surviving finding under `AUDIT_LABEL`. It also files one closed checkpoint
issue under `AUDIT_LABEL` recording the audited SHA.

Watch especially for tests that assert the wrong invariant or pass without
exercising the behavior they name, fixtures that have drifted from the
production config/whitelist they stand in for, coverage gaps on metric-affecting
code, flaky/non-isolated tests, marker mistakes, and stale tests referencing
removed code.

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
- **Area mapping** (test paths map to the same area set as the general audit, by
  the production area each test exercises):
  - `tests/qb/*` -> `qb`; `tests/rb/*` -> `rb`; `tests/wr/*` -> `wr`;
    `tests/te/*` -> `te`; `tests/k/*` -> `k`; `tests/dst/*` -> `dst`.
  - `tests/shared/*`, `tests/conftest.py`, `tests/_skip_helpers.py`,
    `tests/_pipeline_e2e_utils.py`, `tests/integration/*`, `tests/analysis/*`,
    and other root-level non-position test modules -> `shared`.
  - `tests/test_app*.py` and other serving-API tests -> `serving`.
  - `tests/batch/*` -> `batch`.
  - `tests/scripts/*`, `pyproject.toml` (`[tool.pytest]`/coverage config),
    `codecov.yml`, and `.github/workflows/tests.yml` -> `ci`.
  - A finding about a test/fixture's divergence from a position's production
    config takes the area of that position.

## Severity and model regress-risk

Severity is bug impact:

- `severity-docs`: docs/comment-only findings, including stale prose, dead
  links, wrong documented counts, or wrong checked-in agent instructions.
- `severity-low`: latent, no-op, unreachable, or unlikely-to-fire defects.
- `severity-medium`: plausible correctness bug between low and high impact.
- `severity-high`: likely regressing error metrics, silently producing wrong
  results, security-sensitive, or causing live production problems now.

For test-suite findings, read severity as the consequence of the test defect:
a test that would let a real metric regression or wrong-result bug ship green is
`severity-high`; a flaky/non-isolated test that intermittently fails CI without
masking a real bug is typically `severity-medium`; a coverage gap on a
metric-affecting path is usually `severity-low` to `severity-medium` by how
likely an untested change is to slip; a stale assertion that can never fire is
`severity-low`.

Model regress-risk is the likelihood that the fix changes model error metrics
or trained artifacts. It is not about serving UI/API/display behavior unless the
fix changes training, feature values, targets, scoring, evaluation, artifacts, or
model inputs. A pure test/fixture/CI fix that does not touch production code is
almost always `regress-risk-low`; the exception is a fixture or contract test
whose correction implies a production whitelist/target is also wrong (flag those
`regress-risk-medium` and name the production file to recheck).

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
For test-suite findings, the closest categories are usually `wrong_result` (a
test asserting/letting through the wrong thing), `invariant` (a fixture/config
drift), `orphan_code` (a test exercising dead production code), or `other`
(flakiness/isolation/coverage/marker); pick the nearest fit.

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

1. Read AGENTS.md stop rules verbatim, plus the testing conventions in AGENTS.md
   (`CONFIG_TINY` is the test fixture not production; feature whitelist is
   explicit; attention static-feature whitelist is separate and non-temporal;
   `non_negative_targets` is per-head; always diff training vs inference) and the
   CI & training section (`tests.yml` shard matrix, `pytest -m unit`, the 80%
   per-flag Codecov target, the `[docs-only]` opt-in). Hold them for worker
   prompts and final verification.
2. Run `grep "^### \[FIXED\]" todo/fixed-archive.md` and capture every title
   line. Hold the list.
3. Build the dedupe pool from existing per-finding issues across all labels in
   `DEDUPE_AUDIT_LABELS`. Include open and closed issues carrying a severity
   label; this excludes checkpoint issues because they carry no severity label.

```bash
: > /tmp/known_issues.tsv
for label in $DEDUPE_AUDIT_LABELS; do
  gh issue list --label "$label" --state all --limit 800 --json number,title,labels \
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

Use parallel workers where the provider supports them. How many workers to
spawn, how scopes are grouped into workers, and how batches are sequenced is up
to the runner — a worker may cover several scopes, and over-spawning is worse
than fewer, deeper workers. Every scope below must be covered; run sequentially
where parallelism is unavailable, preserving the same scopes and output
contract.

Layer A test-shard auditor scopes (each per-position scope pairs the tests with
the production config/fixtures they stand in for):

1. QB tests auditor: `tests/qb/`, cross-checked against `src/qb/config.py`
   (`CONFIG_TINY` vs `POSITION_CONFIG`), `src/qb/features.py`, `src/qb/targets.py`
2. RB tests auditor: `tests/rb/`, cross-checked against `src/rb/{config,features,targets}.py`
3. WR tests auditor: `tests/wr/`, cross-checked against `src/wr/{config,features,targets}.py`
4. TE tests auditor: `tests/te/`, cross-checked against `src/te/{config,features,targets}.py`
5. K tests auditor: `tests/k/`, cross-checked against `src/k/{config,features,targets}.py`
6. DST tests auditor: `tests/dst/`, cross-checked against `src/dst/{config,features,targets}.py`
7. Shared tests auditor: `tests/shared/`, `tests/conftest.py`, and root-level
   non-position test modules (e.g. `tests/test_aggregate_targets*.py`)
8. Serving tests auditor: `tests/test_app*.py` and any other serving-API tests
9. Batch/scripts/integration tests auditor: `tests/batch/`, `tests/scripts/`,
   `tests/integration/`, `tests/analysis/`
10. Test-config + CI auditor: `pyproject.toml` (`[tool.pytest.ini_options]`,
    markers, `addopts`, coverage `core`), `codecov.yml` (flags, components,
    `ignore`, 80% targets), `tests/_skip_helpers.py`, `tests/_pipeline_e2e_utils.py`,
    and `.github/workflows/tests.yml` (shard matrix, `[docs-only]` detect,
    `COVERAGE_CORE`, faulthandler timeouts)

Layer B standing test cross-cutting lenses:

- T1 Coverage gap: production `src/` paths that affect metrics/serving with no
  test exercising them. Honor `codecov.yml` `ignore`/component excludes —
  diagnostic CLIs (`diagnose_outliers.py`, `analyze_errors.py`,
  `benchmark_ridge_variants.py`), `src/benchmarking/`, `src/tuning/`,
  `src/analysis/`, and `src/scripts/` are excluded from the denominator by
  design and are NOT coverage-gap findings. Only a genuinely untested
  metric-affecting path is a finding.
- T2 Fixture/config drift: a `CONFIG_TINY` value or a synthetic-frame fixture
  that diverges from the `POSITION_CONFIG` (or feature whitelist / target set)
  it stands in for in a way that makes a test assert the wrong production
  behavior. Read `POSITION_CONFIG` (kwarg form, lower in the file), not the
  `CONFIG_TINY` dict literal, for production truth. A *deliberately* shrunken
  `CONFIG_TINY` (fewer epochs, LightGBM off, attention off) is intended and is
  NOT a finding.
- T3 Test/production parity: feature-whitelist contract tests that no longer
  match `src/features/engineer.py` output or the position `include_features` /
  `attn_static_features` / `attn_history_stats` lists; an `attn_static_features`
  non-temporal invariant the suite claims to guard but does not; train-vs-serve
  feature build coverage gaps.
- T4 Flakiness/isolation: xdist races (a from-import of a loader bypassing the
  `conftest` schedule stub; a partial-parquet `ArrowInvalid` from a worker
  reading another's mid-write cache), E2E/pipeline tests that write into a
  position's served `{pos}/outputs/` instead of an isolated tmp dir (see
  `tests/_pipeline_e2e_utils.py`), and missing/incorrect skip guards
  (`require_splits()` / `ALLOW_SKIP_E2E`) on data-dependent tests.
- T5 Assertion quality: a test that passes without asserting the behavior its
  name claims; a `regression`-marked test with no real threshold; GPU-guarded
  production code (`if torch.cuda.is_available()`) presented as covered when it
  cannot execute on CPU CI (report as known-untested, graded LOW/MEDIUM).
- T6 Stale/orphan tests: a test referencing a removed feature/column/symbol/
  marker; a pytest marker used but not registered in `pyproject.toml`
  (`--strict-markers` would error); a `tests.yml` shard matrix out of sync with
  the `tests/{pos}/` layout (e.g. a new position not wired into a shard or the
  Codecov flag set).

Budget is soft pacing only; the 2-hour run wall-clock is the only hard limit.
Set each scope's soft budget from a base leverage tier, adjusted by
`/tmp/area_yield.tsv` (a worker covering several scopes gets the sum of its
scopes' budgets):

- HEAVY 30 minutes: shared tests, serving tests, test-config + CI, T1-T6.
- MEDIUM 20 minutes: qb, rb, wr, te, k, dst tests; batch/scripts/integration tests.
- If sample >= 5 and yield >= 0.60, bump up one tier.
- If sample >= 5 and yield < 0.30, drop one tier.
- shared tests, serving tests, test-config + CI, and T1-T6 never drop below HEAVY.
- sample < 5 or no data keeps the base tier.

Before spawning, print the scope -> tier, budget, yield, sample table and the
scope -> worker grouping.

## Per-worker prompt template

Substitute `{ROLE_ID}`, `{SCOPE}`, `{FOCUS}`, and `{BUDGET}`.

```text
ROLE: {ROLE_ID}. Scope: {SCOPE}. {BUDGET}-minute soft budget. The run's
2-hour wall-clock is the sole hard limit. You are auditing the TEST SUITE.

PRIMARY FOCUS:
(a) Artifacts of unfinished-but-merged PRs in tests: dead test code, half-renamed
    fixtures/symbols, orphan imports, commented-out test blocks, TODO/FIXME,
    feature-list/test-fixture mismatches.
(b) Semantic merge conflicts from concurrent PRs blind to each other's changes:
    check `git log -p -n 30 -- <file>` for competing recent edits.
(c) Orphan code under live test coverage: production-side functions with zero
    callers in `src/` but exercised by tests.
(d) {FOCUS}
(e) Anything that would let a real metric regression or wrong-result bug ship
    green — a test that asserts the wrong invariant, passes without exercising
    the behavior it names, or guards a production invariant it does not actually
    check.

NOT FINDINGS:
- "Position X has a test/fixture Y that position Z doesn't" -> per-position test
  suites and whitelists are intentional.
- "Add tests to position X for parity with Y" / "harmonize fixtures across
  positions" -> not a defect; feature/test engineering.
- "Position X has CONFIG_TINY, Y doesn't" or a CONFIG_TINY value that is smaller
  than POSITION_CONFIG (fewer epochs, LightGBM off, attention off) -> CONFIG_TINY
  is a deliberately shrunken test fixture, not production.
- Sub-target coverage on codecov-`ignore`d paths (diagnostic CLIs, src/tuning,
  src/benchmarking, src/analysis, src/scripts) -> excluded from the denominator
  by design.
- A GPU-guarded production path uncovered by CPU CI -> known-untested by design;
  report only the actual incorrectness if any, graded LOW/MEDIUM, never HIGH for
  "no GPU coverage" alone.
- Style-only test nits (naming, ordering, redundant asserts) unless they cause a
  wrong or absent assertion.
- A doc/comment with only a stale `file:lineN` citation while the cited target is
  otherwise correct -> cosmetic line-number drift.
- Any finding that would change a design choice, feature selection, model
  architecture/hyperparameters, scoring, or model accuracy as tuning/judgment
  rather than fixing a defect -> DROP unless it is a clear, non-controversial
  correctness bug.

STILL REPORT substantive doc errors: wrong module/symbol attribution in test
docs, a documented testing convention/count that is wrong, a stated test
invariant the suite violates, a dead cross-ref to a deleted fixture/file, or a
referenced marker/fixture that is not defined.

When in doubt, drop the finding.

STOP RULES:
<inline AGENTS.md stop rules + every todo/fixed-archive.md FIXED title from Step 1>

SELF-VERIFY every candidate before emitting it:
1. Re-open the cited file at the cited line. Confirm evidence_quote is
   verbatim-present, whitespace-normalized, at or within +/-3 lines. If absent,
   DROP.
2. Actively re-confirm the defect still holds via the cheapest decisive check:
   - coverage_gap (category other/orphan_code): confirm no test references the
     symbol AND the path is not in codecov `ignore`/excluded components. If
     covered or excluded, DROP.
   - fixture/config drift (category invariant): open POSITION_CONFIG (kwarg form)
     and the fixture; confirm the values actually diverge in a way that changes
     asserted behavior. If they match or the divergence is the intended TINY
     shrink, DROP.
   - test/prod parity (category train_serve_drift): read the test's expected
     columns/values and the production whitelist/engineer output; confirm they
     differ. If they match, DROP.
   - flakiness/isolation (category other): name the concrete race/clobber (the
     from-import, the shared path, the served outputs/ write). If you cannot,
     DROP.
   - stale/orphan (category orphan_code/broken_reference): `grep -rn "<symbol>"
     src/ tests/`; confirm the referenced feature/column/symbol/marker is truly
     gone or unregistered. If it resolves, DROP.
   - unfinished_pr / merge_conflict: `git log -p -n 30 -- <file>`; confirm the
     issue is still on HEAD and not completed by a later commit. If reconciled,
     DROP.
   - wrong_result / security / other: name the concrete trigger reaching the
     wrong/unsafe path. If you cannot, DROP.
3. Only survivors may be emitted. When in doubt, DROP.

SEVERITY:
DOCS = docs/comment-only findings. LOW = latent, no-op, unreachable, or
unlikely-to-fire defects (e.g. a stale assertion that can never fire). MEDIUM =
plausible correctness bug between low and high impact (e.g. a flaky/non-isolated
test, a coverage gap on a likely-to-change metric path). HIGH = a test defect
likely to let a metric regression or wrong-result bug ship green, or a test that
silently produces wrong results.

MODEL REGRESS-RISK:
docs = docs/comment-only fix. low = model change is not possible (pure
test/fixture/CI fix). medium = model change is possible but unlikely, or the fix
implies a production whitelist/target is also wrong. high = model change is
likely; fixes should be urged to rerun the relevant pipeline or benchmark. This
is about MSE/MAE/FP-MAE and trained artifacts, not serving UI/API/display
behavior by itself.

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
4. For coverage-gap/orphan claims, `grep -rn "<symbol>" src/ tests/` and confirm
   no test exercises it and it is not codecov-excluded. For fixture/parity
   claims, open both the fixture and the production config/whitelist and confirm
   they actually diverge. For unfinished-PR or semantic-merge-conflict claims,
   run `git log -p -n 20 -- <file>` and confirm consistency. Drop if resolved.
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

Always create a checkpoint issue, including clean runs. The checkpoint title is
routine-distinct so the tests-audit trail is separable from the general audit
and the infrastructure-audit:

```bash
cat > /tmp/checkpoint.md <<EOF
HEAD-SHA: ${HEAD_SHA}
HEAD-SHORT: ${SHORT_SHA}
Date: ${DATE}
Routine: tests-audit
Provider: ${AUDIT_PROVIDER}
Audit label: ${AUDIT_LABEL}
Findings filed this run: ${N_NEW}
Severity counts: ${N_NEW_DOCS} DOCS, ${N_NEW_LOW} LOW, ${N_NEW_MEDIUM} MEDIUM, ${N_NEW_HIGH} HIGH
Regress-risk counts: ${N_RISK_DOCS} docs, ${N_RISK_LOW} low, ${N_RISK_MEDIUM} medium, ${N_RISK_HIGH} high
Filed: <comma-separated #numbers from /tmp/filed.tsv, or "none (clean checkpoint)">
EOF

CP=$(gh issue create --title "[$AUDIT_LABEL] tests-audit checkpoint ${DATE} @${SHORT_SHA}" --label "$AUDIT_LABEL" --body-file /tmp/checkpoint.md)
gh issue close "${CP##*/}" --comment "tests-audit checkpoint - HEAD + finding counts recorded for the audit trail; per-finding issues filed separately. Auto-closed by tests-audit routine."
```

Print a summary with the checkpoint URL and every filed issue URL. If any GitHub
write failed, print the unsent bodies to stdout.

## Final constraints

- One issue per finding.
- Severity and regress-risk are labels, never title tokens.
- Area is both the title prefix and an area label where that label exists.
- First seen is the SHA/date of the run that first files the finding.
- Existing issues are left untouched.
- Reuse the `claude-audit`/`codex-audit` labels and `agent-audit/v1` schema; this
  routine introduces no new label.
- Dedup spans open and closed severity-labeled `claude-audit` and `codex-audit`
  issues, so findings already filed by the general audit, a prior tests-audit, or
  the infrastructure-audit are not re-filed.
- Never re-flag anything in AGENTS.md stop rules or `todo/fixed-archive.md`.
- Never propose cross-position test/fixture harmonization; that is test
  engineering.
- Never file design, tuning, or accuracy-judgment changes unless they are clear,
  non-controversial correctness bugs.
- Every run posts one closed `tests-audit checkpoint` issue under its own
  `AUDIT_LABEL`.
