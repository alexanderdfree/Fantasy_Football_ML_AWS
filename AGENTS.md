# AGENTS.md

Shared instructions for Claude Code, Codex and Gemini. Keep this entrypoint under
8 KiB; detailed guidance belongs in the topic files below. Read the relevant
sections before changing that subsystem. Do not load every linked document.

## Start with the task

- Verify the active worktree and `git status` before editing. Use paths inside
  this checkout; never silently edit the parent checkout. Fetch `origin/main`
  and inspect recent commits before planning and again before a PR. Check open
  PRs for overlap in shared files. Current remote state takes precedence over
  an old worktree or recalled result.
- Follow the user's current scope and prior authorization. A tentative mechanism
  is a hypothesis: validate its fit before building a new subsystem. If scope
  proves infeasible or a gate blocks the intended work, explain the evidence and
  options; do not silently drop scope or bypass the gate.
- Use current code/configuration for implemented behavior, the relevant ADR for
  intended decisions, and live artifacts for shipped behavior. Reconcile any
  disagreement. Memories and dated incident reports are retrieval aids; they
  cannot establish current versions, quotas, flags, settings or deployment state.

## Read only what applies

| Task | Guidance to read |
|---|---|
| Locate a subsystem or add a position | [Project layout](agent-guides/project.md) |
| Features, targets, losses or NN wiring | [Model contracts](agent-guides/modeling.md), [modeling stop rules](agent-guides/stop-rules.md#modeling-and-features) |
| Investigate accuracy, data effects or a claimed invariant | [Production validation](agent-guides/validation.md#production-path), [investigation](agent-guides/investigation.md) |
| Run tests, trains, tunes, benchmarks or A/Bs | [Entry points](agent-guides/experiments.md), relevant [environment](agent-guides/environment.md) section; commands in [SETUP.md](SETUP.md) |
| Device, dtype, performance or GPU changes | [Device/dtype policy](agent-guides/platform.md#device-and-dtype-policy), [GPU stop rules](agent-guides/stop-rules.md#gpu-execution) |
| CI, AWS, serving or artifact lifecycle | [Operations](agent-guides/operations.md), [CI/serving stop rules](agent-guides/stop-rules.md#ci-and-serving); [ADRs](docs/ARCHITECTURE.md) |
| Edit, review, open or merge a PR | [Delivery](agent-guides/delivery.md); provider workflow in [CODEX.md](CODEX.md), [CLAUDE.md](CLAUDE.md) or [GEMINI.md](GEMINI.md) |
| Existing plans or a previously fixed bug | Search [TODO.md](TODO.md) or [fixed-issue index](todo/fixed-archive.md), then open only matching entries |
| Maintain instructions or memory | [Context maintenance](agent-guides/context-maintenance.md) |

Human overview: [README.md](README.md); read only when needed.

## Invariants to preserve

- All six positions (`qb/rb/wr/te/k/dst`) predict raw NFL stats. Compute fantasy
  points afterwards with `predictions_to_fantasy_points`; never train on totals.
  Shared changes must account for all six positions and every relevant caller.
- Production configuration is `POSITION_CONFIG`; `CONFIG_TINY` is a test fixture.
  Feature whitelists are explicit. Updating a feature requires its configuration
  allowlist and fixtures. Attention has a separate static/history allowlist:
  static features are non-temporal; history tokens are raw per-game signals.
  Do not add rolling/ewma/windowed aggregates to either as redundant history.
- **Evaluation cohorts (ADR-0024):** compare identical regular-season player-weeks
  using the same projected scoring components in forecasts and actuals; missing
  components are unavailable, never a full-fantasy fallback. Use
  `src/shared/comparison_scoring.py` and the [cohort rules](agent-guides/validation.md#evaluation-cohorts).
  `weekly_reference_top24` uses the archived pregame reference; `elite_top24`
  retains prior-season importance. Actual weekly leaders are for ranking;
  seasonal leaders are retrospective. Never use a model's own top-N pool for
  cross-source MAE, restore static expert summaries as live accuracy, or silently
  omit unavailable cohort data from Batch/local results.
- Preserve training/inference feature parity, per-head non-negativity and coupled
  loss scales/weights. Keep NN forward/loss/aggregation operations in `torch` so
  gradients survive. See the model contracts before changing any of these.
- Validate with the production loader, NaN handling, configuration and actual
  pipeline. Identical Ridge MAE is a diagnostic clue, not proof of identical
  data; verify inputs/configuration directly. Check activation preconditions and
  positive controls before claiming an effect. GPU-only paths need GPU evidence.
- Use existing parallelized harnesses before any compute-bearing run; start with
  a targeted unit subset or one real smoke cell. A/Bs must isolate outputs and
  compare the same regime. Production training stays eager; stacked tuning is
  not seed-by-seed comparable with eager training. Read platform rules first.
- Platform behavior autodetects through existing primitives and supports explicit
  overrides. CUDA defaults to FP32+TF32 where supported; FP16/BF16 and Mac MPS are
  opt-in. Native Windows requires `OPENBLAS_NUM_THREADS=1`. Do not infer a dtype
  default or speedup from hardware support alone.
- Heavy feature building/inference belongs in CI-built artifacts, not the serving
  container. Edit dashboard sources in `src/serving/frontend/`, then rebuild the
  committed bundle with `npm run build`; never edit `static/js/app.js` directly.
- Do not commit datasets, model weights or large media. Tuning/ablation code goes
  in `src/tuning/`; diagnostics in `src/analysis/`. Shared/training paths can
  trigger six-position retraining: check `src/scripts/scope_positions.py`.

## Delivery and evidence

- Follow the full delivery workflow: feature branch → appropriate local checks →
  pre-PR scope judge → PR → current green CI/review → merge when authorized.
  Preserve explicit approval gates, including `solve-issues` owner sign-off.
  Never use `--no-verify`, `--admin`, or a silent gate bypass. The documented
  silent-stop CI exception requires local validation; see delivery guidance.
- NN/feature/loss/target changes require an actual affected-position pipeline
  comparison before merging. Unit tests and green CI do not establish metric
  neutrality. Use relevant subgroup metrics and multi-seed evidence.
- Run a gate separately from dependent mutations. Verify checkout before rebase,
  resolve all conflict markers, and verify MERGED state and latest squash content
  before remote branch deletion. In worktrees use `gh pr merge --squash` without
  `--delete-branch`; the parent may hold `main`.
- Keep durable decisions in the relevant ADR and its changelog; record non-trivial
  fixes once in the fixed-issue archive. Update existing guidance where the lesson
  belongs, linking to evidence. Trivial edits need neither a new ADR nor incident.
  A docs-only CI opt-out is permitted only for wholly non-behavioral changes;
  rendered response strings and changes to the opt-out machinery do not qualify.
- Keep output bounded: enumerate/size before reading, save large logs/payloads to
  temporary files, preserve the command's exit status, and return selected fields.
  Never dump session JSONL, minified bundles or full endpoint payloads into chat.
  Re-read truncated/garbled results in a smaller query before relying on them.
