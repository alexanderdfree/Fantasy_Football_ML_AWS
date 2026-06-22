# Shared infrastructure-audit instructions

These are the provider-neutral instructions for the scheduled
**infrastructure-audit** routine — a deeper, infra-scoped sibling of the general
code audit (`routines/audit/instructions.md`). It covers CI/CD, training
orchestration (AWS Batch / EC2), Docker images, serving/ECS deploy, IAM, and the
model-artifact lifecycle. Provider wrappers must read this file, set the runtime
variables below, apply their own tool/worktree rules, and then execute the shared
workflow.

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

Review the Fantasy_Football_ML_AWS **infrastructure** — CI/CD workflows, training
orchestration (Batch/EC2), Docker images, serving/ECS deploy, IAM, and the
model-artifact lifecycle — for correctness and operational-safety defects in
parallel when the provider supports it. Use per-area location auditors plus
standing infra-specific cross-cutting lenses. Each worker emits severity-labeled
findings with verbatim evidence.

This is a **deeper, scoped pass than the general audit**, not a replacement for
it. The general audit (`routines/audit/instructions.md`) sweeps `src/batch/` and
`.github/workflows/` shallowly through its Batch+CI auditor; this routine goes
deep on the whole deploy/orchestration surface, including `infra/` and the
serving deploy path. Because both file under the same `claude-audit`/`codex-audit`
labels and this routine dedupes against the **same shared pool**, any finding the
general audit (or a prior infrastructure-audit / tests-audit run) already filed is
suppressed automatically. The value here is depth in the infrastructure domain,
not breadth.

The orchestrator verifies every cited line, dedupes against open and closed
audit issues from all labels in `DEDUPE_AUDIT_LABELS`, drops anything already
covered by AGENTS.md stop rules or the `todo/fixed-archive.md` fixed archive,
consolidates partial/full duplicates within this run, and files one GitHub issue
per surviving finding under `AUDIT_LABEL`. It also files one closed checkpoint
issue under `AUDIT_LABEL` recording the audited SHA.

Watch especially for training-path routing flags mishandled across workflows,
artifact races / missing job-definition pins, stale-artifact serving risks,
Dockerfile/wheel/platform-detection bugs, `[docs-only]`/`paths:`-filter gaps,
over-broad IAM or committed secrets, and quota/auto-shutdown gaps.

## Operating boundaries

Working directory is the repo root. Time budget is 2 hours wall-clock.

The routine is read-only on repo files. Writing to `/tmp/*` is allowed. The only
GitHub writes permitted are:

- `gh label create` for audit/severity/regress-risk labels, idempotently.
- `gh issue create` for per-finding issues and this run's checkpoint.
- `gh issue comment` only on issues created in this same run, for related links.
- `gh issue close` only on the checkpoint issue created in this same run.

Do not edit repo files, commit, push, create branches, open PRs, close existing
finding issues, or edit existing issue titles/bodies. Do not run AWS CLI / `gh
api` mutations, dispatch workflows, or touch live cloud resources — this routine
reads repo-tracked infra definitions only.

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
- **Area mapping** (infra paths reuse the existing area set — no new area label):
  - `.github/workflows/*`, `.claude/hooks/*`, `.codex/hooks/*`, `.gemini/hooks/*`,
    `src/scripts/*` (e.g. `scope_positions.py`), `requirements*.txt` -> `ci`.
  - `src/batch/*` (Dockerfile.train, launch.py, train.py, benchmark.py,
    build_and_push.sh), `infra/batch/*`, `infra/ec2/*` -> `batch`.
  - `infra/aws/*` (ECS task definition + IAM + bootstrap/seed/teardown) and the
    deploy-relevant parts of `src/serving/*` (entrypoint, gunicorn/pre-warm,
    `/health`, S3 artifact sync) -> `serving`.
  - A finding spanning a workflow and the per-position training it dispatches
    takes `batch` if it is about training/artifacts, `ci` if about the workflow
    wiring itself.
  - The file-issue step already retries without the area label if a label does
    not exist, so an `infra`-style label is unnecessary; map to the nearest
    existing area above.

## Severity and model regress-risk

Severity is bug impact:

- `severity-docs`: docs/comment-only findings, including stale prose, dead
  links, wrong documented counts, or wrong checked-in agent instructions.
- `severity-low`: latent, no-op, unreachable, or unlikely-to-fire defects.
- `severity-medium`: plausible correctness bug between low and high impact.
- `severity-high`: likely regressing error metrics, silently producing wrong
  results, security-sensitive, or causing live production problems now.

For infrastructure findings, read severity as operational consequence: a defect
that would serve stale/NaN predictions, train the wrong code, leak a secret, or
break a deploy is `severity-high`; a flag/path-filter gap or artifact race that
could intermittently misfire is typically `severity-medium`; a latent or
unreachable orchestration defect is `severity-low`.

Model regress-risk is the likelihood that the fix changes model error metrics
or trained artifacts. It is not about serving UI/API/display behavior unless the
fix changes training, feature values, targets, scoring, evaluation, artifacts, or
model inputs. An infra fix is `regress-risk-high` when it changes which code or
data trains a model (a `scope_positions` mis-scope, a wrong job-definition
revision, a stale-splits training path) — those should be urged to rerun the
relevant pipeline/benchmark. A pure CI-wiring, deploy, IAM, or cost fix that does
not alter the trained artifact is `regress-risk-low`.

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
For infra findings, common fits are `broken_reference` (a workflow referencing a
removed script/env var/job-def), `invariant` (a flag/scope mismatch across
workflows), `train_serve_drift` (a serving deploy that diverges from how training
publishes artifacts), `security` (over-broad IAM, committed secret), or
`wrong_result` (an artifact race / stale-artifact serve); pick the nearest fit.

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

1. Read AGENTS.md stop rules verbatim, plus the "Platform & hardware targets"
   stop-rules (FP32+TF32 default / FP16+BF16 opt-in / CUDA-graph autodetect-ON
   sm_80+ / MPS opt-in / Windows OPENBLAS), the "CI & training" section (`BATCH_ACTIVE` /
   `BATCH_SPLIT_ACTIVE`, `train-batch.yml` vs `train-ec2.yml`, the shared
   `detect` job + `scope_positions`, the `[docs-only]` opt-in contract,
   `deploy.yml` being paths-gated), and the infra-relevant Stop rules
   (shared-venv CI, `--preload` pre-warm, in-container artifact build). Hold them
   for worker prompts and final verification.
2. Run `grep "^### \[FIXED\]" todo/fixed-archive.md` and capture every title
   line. Hold the list. Also skim `docs/batch_design.md`, `docs/ec2_design.md`,
   and the infra-relevant ADRs (job-def pinning ADR-0020, split training
   ADR-0019, upcoming-week ADR-0018) so a documented-and-intended design is not
   re-flagged as a defect.
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

Use parallel workers where the provider supports them. How many workers to
spawn, how scopes are grouped into workers, and how batches are sequenced is up
to the runner — a worker may cover several scopes, and over-spawning is worse
than fewer, deeper workers. Every scope below must be covered; run sequentially
where parallelism is unavailable, preserving the same scopes and output
contract.

Layer A location-auditor scopes:

1. CI-workflows auditor: `.github/workflows/*.yml` — `tests.yml`,
   `batch-image.yml`, `train-batch.yml`, `train-ec2.yml`, `deploy.yml`,
   `_detect-positions.yml`, `refresh-splits.yml`, `refresh-upcoming-week.yml`,
   `ab-batch.yml`, `retune-nn-batch.yml`, `retune-lgbm.yml`, `ablate-rb-gate.yml`,
   `ablate-scheduler.yml`, `benchmark-batch.yml`, `skip-sentinel.yml`,
   `codeql.yml`
2. Batch-orchestration auditor: `src/batch/` — `Dockerfile.train` +
   `Dockerfile.train.dockerignore`, `launch.py`, `train.py`, `benchmark.py`,
   `build_and_push.sh`, `requirements.txt`
3. AWS Batch/EC2 infra auditor: `infra/batch/` (setup/teardown, IAM trust/job
   policies) and `infra/ec2/` (launch-instance, user-data, auto-shutdown
   service/timer, cloudwatch-agent, IAM)
4. Serving/ECS infra auditor: `infra/aws/` (task-definition.json, IAM trust/role
   policies, bootstrap/seed/teardown, bucket versioning) and the **deploy-relevant**
   parts of `src/serving/` (entrypoint, gunicorn config + post-fork pre-warm,
   `/health`, S3 artifact sync). Do NOT audit serving feature-build logic here —
   that belongs to the general audit / tests-audit.
5. Position-scope + hooks auditor: `src/scripts/scope_positions.py` and its
   contract test `tests/scripts/test_scope_positions.py`,
   `.github/workflows/_detect-positions.yml`, `.claude/hooks/`, `.codex/hooks/`, and
   `.gemini/hooks/` (the pre-pr / post-pr / freshness gates that gate merges and training)

Layer B standing infra cross-cutting lenses:

- I1 Training-path routing: `BATCH_ACTIVE` / `BATCH_SPLIT_ACTIVE` handled
  consistently across `batch-image.yml` -> `train-batch.yml` / `train-ec2.yml`;
  `src/scripts/scope_positions.py` <-> `src/shared/registry.py` position-list
  parity (contract-tested), the global-trigger list (`src/shared/`, `src/data/`,
  `src/features/`, `src/batch/` minus `tune`/`ablate`/`launch.py`/`benchmark.py`,
  `src/config.py`, `requirements.txt`), the tuner-must-live-in-`src/tuning/` rule
  (#280 burned GPU jobs on a tuner placed in `src/batch/`), and the
  `launch.py`/`benchmark.py` exclusions. Only a real routing/scope defect is a
  finding.
- I2 Artifact lifecycle: job-definition revision pinning (ADR-0020; a
  concurrent-run divergence when two images register in the same window),
  atomic manifest write, versioned `history/{ts}-{sha}` artifact paths, and the
  split-mode `nn` + `cpu` + `merge` dependency graph (validate-before-promote;
  a merge-job gap leaving stale artifacts). Only a real race/lifecycle defect is
  a finding.
- I3 Serving deploy safety: ECS service rollover after an **architecture-changing**
  retrain (a weight-only hot-swap NaNs the new shape — the 2026-06-15 staleness
  incident); no heavy `load_raw_data` / `build_features` / inference in the
  serving container (#1069 -> #1076 OOM, build in CI then download); `/health`
  distinguishing cold-start 200 from affirmative 503; no module-level pre-warm
  under gunicorn `--preload` (#148/#149). Only an actual divergence from these
  decided patterns is a finding.
- I4 Docker / platform: cu126 (T4/A10G/L4 Batch image) vs cu130 (Blackwell
  sm_120 local) wheel correctness per build target; `.dockerignore` excludes
  data/large binaries; AMP/CUDA-graph/`torch.compile` autodetect-and-branch (not
  hardcoded for one arch); the Windows `OPENBLAS_NUM_THREADS=1` correctness guard.
  Re-flag a hardcoded-for-one-box assumption, NOT an intentional per-arch default.
- I5 Workflow correctness: `[docs-only]` gate consistency across `tests.yml`
  (`detect`), `batch-image.yml` (`check-docs-only`), `_detect-positions.yml`, and
  the paths-gated `deploy.yml`; `paths:` / `paths-ignore:` filters; concurrency
  groups; retry/backoff on S3/Batch polling; least-privilege IAM in `infra/*`
  policy JSON; and no committed secrets/credentials in workflows or infra
  scripts.
- I6 Cost / quota safety: AWS quota assumptions (24 vCPU G+VT OD, 64 vCPU Spot;
  `maxvCpus=64`), Spot CE ordering (g6 primary -> g5 fallback), and the EC2
  auto-shutdown service/timer being present and wired (idle-cost guard). Only a
  correctness/safety gap is a finding — not a cost-tuning suggestion.

Budget is soft pacing only; the 2-hour run wall-clock is the only hard limit.
Set each scope's soft budget from a base leverage tier, adjusted by
`/tmp/area_yield.tsv` (a worker covering several scopes gets the sum of its
scopes' budgets):

- HEAVY 30 minutes: CI-workflows, Batch-orchestration, serving/ECS infra, I1-I6.
- MEDIUM 20 minutes: AWS Batch/EC2 infra, position-scope + hooks.
- If sample >= 5 and yield >= 0.60, bump up one tier.
- If sample >= 5 and yield < 0.30, drop one tier.
- CI-workflows, Batch-orchestration, serving/ECS infra, and I1-I6 never drop
  below HEAVY.
- sample < 5 or no data keeps the base tier.

Before spawning, print the scope -> tier, budget, yield, sample table and the
scope -> worker grouping.

## Per-worker prompt template

Substitute `{ROLE_ID}`, `{SCOPE}`, `{FOCUS}`, and `{BUDGET}`.

```text
ROLE: {ROLE_ID}. Scope: {SCOPE}. {BUDGET}-minute soft budget. The run's
2-hour wall-clock is the sole hard limit. You are auditing INFRASTRUCTURE
(CI/CD, Batch/EC2 training, Docker, serving/ECS deploy, IAM, artifact lifecycle).

PRIMARY FOCUS:
(a) Artifacts of unfinished-but-merged PRs in infra: dead workflow steps,
    half-renamed jobs/env vars, references to removed scripts/job-defs/secrets,
    commented-out blocks, TODO/FIXME.
(b) Semantic merge conflicts from concurrent PRs blind to each other's changes:
    check `git log -p -n 30 -- <file>` for competing recent edits to a workflow
    or infra script.
(c) Broken references: a workflow or infra script naming a script, env var,
    repo variable, job definition, S3 path, or IAM resource that does not exist
    or has been renamed.
(d) {FOCUS}
(e) Anything that would silently train the wrong code/data, serve a stale or
    NaN artifact, leak a secret, or break a deploy.

NOT FINDINGS:
- Intentional per-arch defaults: FP16 on every CUDA GPU with BF16 opt-in only,
  CUDA-graph autodetect-ON for sm_80+, MPS opt-in (never the Mac default), the
  T4/sm_75 lowest-common-denominator EC2 rollback target -> decided stop-rules,
  not bugs. "The GPU supports BF16, enable it" is NOT a finding.
- Cost/perf tuning suggestions (cheaper instance, different parallelism, fewer
  Spot CEs) that are not an actual correctness or safety defect.
- "train-ec2.yml is slower than train-batch.yml" -> the EC2 path is the
  intentional warm rollback; speed is by design.
- Anything in AGENTS.md Stop rules / todo/fixed-archive (shared-venv CI
  optimization, gunicorn `--preload` module pre-warm, building the upcoming-week
  artifact inside the serving container) -> already tried and reverted.
- A doc/comment with only a stale `file:lineN` citation while the cited target is
  otherwise correct -> cosmetic line-number drift.
- Any finding that would change a design choice, instance selection, or
  cost/perf tradeoff as judgment rather than fixing a defect -> DROP unless it is
  a clear, non-controversial correctness/safety bug.

STILL REPORT substantive doc errors: a workflow/infra doc that attributes the
wrong file/job/flag, a documented training-path/quota/decision that is wrong, a
stated invariant the workflow violates, a dead cross-ref to a deleted
script/job-def, or a referenced env var / repo variable / job definition that is
not defined.

When in doubt, drop the finding.

STOP RULES:
<inline AGENTS.md stop rules + every todo/fixed-archive.md FIXED title from Step 1>

SELF-VERIFY every candidate before emitting it:
1. Re-open the cited file at the cited line. Confirm evidence_quote is
   verbatim-present, whitespace-normalized, at or within +/-3 lines. If absent,
   DROP.
2. Actively re-confirm the defect still holds via the cheapest decisive check:
   - broken_reference: open the referenced script/env var/repo variable/job-def/
     S3 path/IAM resource; confirm it is not defined or is renamed. If it
     resolves, DROP.
   - invariant (flag/scope): read every workflow/file that consumes the flag or
     scope mapping; confirm they actually disagree. If consistent, DROP. For a
     `scope_positions` claim, check `tests/scripts/test_scope_positions.py` and
     `src/shared/registry.py` — if the contract test would catch it, it is not a
     latent defect.
   - train_serve_drift (deploy vs training-publish): read both sides; confirm the
     serving download/shape path actually diverges from how training writes the
     artifact. If they match, DROP.
   - security: name the concrete over-permission or committed secret and why it
     is reachable. A documented placeholder/example is not a secret. If you
     cannot, DROP.
   - unfinished_pr / merge_conflict: `git log -p -n 30 -- <file>`; confirm the
     issue is still on HEAD and not completed by a later commit. If reconciled,
     DROP.
   - wrong_result / other: name the concrete trigger reaching the wrong/unsafe
     path (the artifact race, the stale-artifact serve, the mis-scoped retrain).
     If you cannot, DROP.
3. Only survivors may be emitted. When in doubt, DROP.

SEVERITY:
DOCS = docs/comment-only findings. LOW = latent, no-op, unreachable, or
unlikely-to-fire orchestration defects. MEDIUM = plausible correctness bug
between low and high impact (a flag/path-filter gap, an artifact race that could
intermittently misfire). HIGH = would serve stale/NaN predictions, train the
wrong code/data, leak a secret, or break a deploy now.

MODEL REGRESS-RISK:
docs = docs/comment-only fix. low = model change is not possible (pure CI/deploy/
IAM/cost fix). medium = model change is possible but unlikely, or currently
no-op/latent for model metrics. high = the fix changes which code or data trains
a model (a scope mis-map, a wrong job-def revision, a stale-splits training
path); urge a pipeline/benchmark rerun. This is about MSE/MAE/FP-MAE and trained
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
   keywords. Drop if matched. Also confirm the finding is not a
   documented-and-intended design in `docs/batch_design.md`, `docs/ec2_design.md`,
   or the infra ADRs.
3. Dedupe against `/tmp/known_issues.tsv` and `/tmp/known_files.tsv`: duplicate
   when an existing issue has the same area, same cited file, and at least two
   shared distinctive title keywords. The pool spans open and closed
   `claude-audit` and `codex-audit` per-finding issues.
4. For broken-reference claims, open the referenced script/env var/job-def/IAM
   resource and confirm it is absent. For flag/scope claims, read every consumer
   and confirm they disagree (and that the `scope_positions` contract test would
   not catch it). For unfinished-PR or semantic-merge-conflict claims, run
   `git log -p -n 20 -- <file>` and confirm consistency. Drop if resolved.
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
routine-distinct so the infrastructure-audit trail is separable from the general
audit and the tests-audit:

```bash
cat > /tmp/checkpoint.md <<EOF
HEAD-SHA: ${HEAD_SHA}
HEAD-SHORT: ${SHORT_SHA}
Date: ${DATE}
Routine: infrastructure-audit
Provider: ${AUDIT_PROVIDER}
Audit label: ${AUDIT_LABEL}
Findings filed this run: ${N_NEW}
Severity counts: ${N_NEW_DOCS} DOCS, ${N_NEW_LOW} LOW, ${N_NEW_MEDIUM} MEDIUM, ${N_NEW_HIGH} HIGH
Regress-risk counts: ${N_RISK_DOCS} docs, ${N_RISK_LOW} low, ${N_RISK_MEDIUM} medium, ${N_RISK_HIGH} high
Filed: <comma-separated #numbers from /tmp/filed.tsv, or "none (clean checkpoint)">
EOF

CP=$(gh issue create --title "[$AUDIT_LABEL] infrastructure-audit checkpoint ${DATE} @${SHORT_SHA}" --label "$AUDIT_LABEL" --body-file /tmp/checkpoint.md)
gh issue close "${CP##*/}" --comment "infrastructure-audit checkpoint - HEAD + finding counts recorded for the audit trail; per-finding issues filed separately. Auto-closed by infrastructure-audit routine."
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
  issues, so findings already filed by the general audit, a prior
  infrastructure-audit, or the tests-audit are not re-filed.
- Never re-flag anything in AGENTS.md stop rules or `todo/fixed-archive.md`.
- Never propose an intentional per-arch / cost / instance-choice change as a
  defect; those are tuning/judgment.
- Never file design, tuning, or accuracy-judgment changes unless they are clear,
  non-controversial correctness/safety bugs.
- Every run posts one closed `infrastructure-audit checkpoint` issue under its
  own `AUDIT_LABEL`.
