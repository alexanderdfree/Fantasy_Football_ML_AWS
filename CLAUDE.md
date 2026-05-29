# CLAUDE.md

Orientation file for Claude Code. Human-facing docs live elsewhere — this file exists to surface the conventions, gotchas, and "before you touch X, read Y" rules that aren't obvious from a first pass through the tree.

## Orient yourself first
- **[README.md](README.md)** — overview, architecture diagram, eval results.
- **[SETUP.md](SETUP.md)** — install, first-time data pull, how to run everything locally. If you need a command, it's probably here.
- **[TODO.md](TODO.md)** — open issues and a **Fixed archive** with root-cause + lesson for every non-trivial bug ever squashed. **Read this before proposing changes near anything it mentions** — most "obvious" fixes have been tried and the archive explains why they were wrong. **Update it as you ship**: move Open → Fixed archive (or add a fresh archive entry) using the existing `### [FIXED] Title` + **File(s)/What/Fix/Lesson** format.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the project's ADR (decisions D1–D15 + a dated `Update history`), with rejected alternatives. **Living doc** — update it whenever a non-trivial change touches or adds an architectural decision (see "When making changes").

## Project shape (six-position symmetry)
Each of `src/qb/ src/rb/ src/wr/ src/te/ src/k/ src/dst/` follows the same template:

```
src/{pos}/
  config.py        # hyperparams (Ridge alpha grids, NN dims, loss weights, Huber deltas, LightGBM params)
  data.py          # loading + temporal split specifics
  features.py      # position-specific feature engineering
  targets.py       # raw-stat target definitions
  run_pipeline.py  # exposes run() and (for QB/RB/WR) run_cv()
```

Tests for each position live under `tests/{pos}/`.

Shared plumbing is in [src/shared/](src/shared/): `pipeline.py` (train/eval loop), `models.py` (single-target building blocks `RidgeModel` / `ElasticNetModel` / `SeasonAverageBaseline` / `LastWeekBaseline` plus the multi-target wrappers `RidgeMultiTarget`, `ElasticNetMultiTarget`, `LightGBMMultiTarget`, plus `TwoStageRidge` and the gated TD heads), `neural_net.py` (attention), `aggregate_targets.py` (raw-stat → fantasy-point scoring), `training.py`, `evaluation.py`, `backtest.py`. The root-level `models/` directory is a separate placeholder for trained-model artifacts that load from S3 — different beast.

The rest of `src/` groups by purpose:
- `src/data/` — cross-position data loading + temporal split (per-position `data.py` files wrap these): `loader.py`, `nflcom_loader.py`, `preprocessing.py`, `redzone_pbp.py`, `split.py`.
- `src/features/engineer.py` — cross-position feature engineering coordinator.
- `src/shared/evaluation.py` — position-aware visualization/aggregation layer plus the `compute_metrics(y_true, y_pred)` helper used by backtest and pipeline.
- `src/serving/` — Flask app + assets.
- `src/batch/` — training orchestration (AWS Batch path). New tuner/ablation files go in `src/tuning/`, **never** here — files under `src/batch/` (except those whose name contains `tune` or `ablate`) trigger a full 6-position retrain via [src/scripts/scope_positions.py](src/scripts/scope_positions.py). PR [#280](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/280) burned ~4 GPU-jobs from a tuner-only change accidentally placed here.
- `src/benchmarking/`, `src/tuning/` — Optuna + ablations.
- `src/analysis/` — post-hoc analyses.
- `src/scripts/` — operator CLIs.
- `src/config.py` — global constants (`SEASONS`, `POSITIONS`, scoring dicts, `TOP_K_RANKING`). Distinct from per-position `src/{pos}/config.py`, which holds model hyperparams.

All six positions train an attention NN (DST landed via `cc0c627`, K via `801b61a`). There is no "skill-positions-only" carve-out anymore — if you're adding an NN-related knob, wire it through every position.

**Adding a new position**: copy an existing folder under `src/`, rename files/constants, wire it into `src/batch/train.py` and the position list in `.github/workflows/_detect-positions.yml` (shared by `train-batch.yml` (active path) and `train-ec2.yml` (rollback path)). Also update [src/scripts/scope_positions.py](src/scripts/scope_positions.py) — the canonical path → positions mapping (contract-tested by [tests/scripts/test_scope_positions.py](tests/scripts/test_scope_positions.py)) used by both training workflows' `detect` job. Add tests under `tests/{pos}/`.

## Conventions that bite if ignored

### Raw-stat targets, never fantasy-point targets
Every position predicts raw NFL stats (yards, TDs, receptions, etc.). Fantasy points are computed *after* prediction via `src.shared.aggregate_targets.predictions_to_fantasy_points(pos, preds)`. If you find yourself training a model directly on `fantasy_points`, stop — you'll break scoring-format flexibility and regress the ~1.9 pt/game double-count fix documented in TODO.md's archive.

### Feature whitelist is explicit, not inferred
`POSITION_CONFIG.include_features` in each `src/{pos}/config.py` (a kwarg on the `PositionConfig` dataclass, typically backed by a module-level `_INCLUDE_FEATURES` dict) is an opt-in list. New columns must be added explicitly — the training code will *not* pick them up automatically. This prevents silent feature leakage. When you add a feature, update both the feature-engineering file *and* the include dict, then update the test fixture (`tests/conftest.py` or `tests/{pos}/conftest.py`).

### `CONFIG_TINY` is the test fixture, not production
Each `src/{pos}/config.py` exports **two** config shapes that look identical at a glance and have opposite values for the same toggle:

- `CONFIG_TINY = {...}` — a small dict literal near the module top with shrunken `nn_epochs`, no LightGBM, attention often disabled. Used by `tests/{pos}/` for fast unit runs. Dict-literal syntax (`"train_lightgbm": False`).
- `POSITION_CONFIG = PositionConfig(...)` — the production config object consumed by AWS Batch via `build_pipeline_config(pos, POSITION_CONFIG)` in `run_pipeline.py`. Kwarg syntax (`train_lightgbm=True`).

`grep "train_lightgbm" src/k/config.py` returns **both** entries with opposite booleans. When checking what production actually runs, always read `POSITION_CONFIG` (kwarg form, lower in the file) — never the dict-literal form.

### Attention static-feature whitelist is separate per position
The attention NN's static branch reads a *second*, smaller allowlist: `POSITION_CONFIG.attn_static_features` (commit `2500ecc`), a kwarg on the per-position `PositionConfig` dataclass. It is defined per position (QB/RB/WR/TE derive it from an `ATTN_STATIC_CATEGORIES` subset of `_INCLUDE_FEATURES`; DST/K enumerate it directly). The static branch is **deliberately non-temporal**.

**Never propose adding rolling / ewma / trend / L3 / L5 / L8 (or any windowed) features to `ATTN_STATIC_FEATURES`.** Temporal signal already feeds the NN through `ATTN_HISTORY_STATS` via the per-game attention sequence. Mixing windowed features into the static branch re-creates the double-counting this design exists to prevent. If you see attention NN losing to ridge/LightGBM on some target and you're tempted to "promote the rolling stats LGBM uses" into the static branch — stop. That's the wrong reach. The architectures differ, not the input availability: LGBM consumes rolling stats as flat columns split by trees; the attention NN consumes the *same underlying signal* as a 17-game sequence through attention. Eligible reaches for closing such a gap:

1. Add **non-temporal** features to `ATTN_STATIC_FEATURES` — prior-season aggregates, matchup, contextual, weather/Vegas, role/depth, season-to-date rates, interactions.
2. Add new **per-game** stats to `ATTN_HISTORY_STATS` — red-zone splits, share-style measures, game-script not already in the 17-game sequence.
3. Retune the loss head — Huber δ + matching `LOSS_WEIGHTS = 2.0 / δ` (see next section).
4. Change a head's parametric form — gated/two-stage for sparse counts (but note `hurdle_poisson` was tried and reverted for RB sparse counts in PR [#219](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/219)).
5. NN architecture — `d_model`, `n_heads`, dropout. Larger has been tried and regressed on 15K-sample positions; verify against benchmark before merging.

Adding a feature to `INCLUDE_FEATURES` does **not** feed it into attention — add it to `ATTN_STATIC_FEATURES` (if non-temporal) or `ATTN_HISTORY_STATS` (if per-game) too if that's what you want.

### Loss weights are tuned inverse-to-Huber-delta
`LOSS_WEIGHTS` ≈ `2.0 / HUBER_DELTAS[target]`. The rationale is baked into QB's config comment ([src/qb/config.py](src/qb/config.py)): without this rebalance, yards targets (δ=15–25) dominated count heads (δ=0.5) ~2500× per sample and the count heads collapsed to the mean. If you retune a Huber delta, re-derive the matching loss weight — don't change one without the other.

### `non_negative_targets` is per-head, not global
The NN clamps outputs to ≥ 0 per head. **All six positions set `nn_non_negative_targets=set(_TARGETS)` explicitly** (clamp every head) in their `POSITION_CONFIG`; the `PositionConfig` field default is `field(default_factory=set)` — empty (no clamp) — so a future position that forgets to set it would silently disable non-negativity. The `MultiHeadNet`-level default of `None` (which clamps every head) is the *fallback* shape; production never hits it. If a position ever adds a signed head (e.g. a bonus that can go negative), pass a set that *excludes* that head rather than flipping the behaviour globally. If you construct `MultiHeadNet(...)` anywhere outside `src/shared/pipeline.py::_train_nn`, mirror the `non_negative_targets=cfg.get("nn_non_negative_targets")` kwarg — the CV path was missed once (see TODO.md archive).

### Always diff training vs inference paths
The training pipeline in `src/shared/pipeline.py` and the serving code in `src/serving/app.py` both build features. They have drifted silently in the past (weather/Vegas merge in training but not serving; scaler clip in one path but not the other). If you touch feature building in either, check the other.

### Use `torch` ops inside NN training paths, not `numpy`
Anything that runs inside the forward pass, loss, or an `aggregate_fn` callback must stay in `torch` to preserve gradients. `np.digitize`/`np.clip`/`np.where` on tensors silently breaks autograd — call `torch.bucketize`/`torch.clamp`/`torch.where` instead. Note that `torch.bucketize(..., right=False)` and `np.digitize(..., right=False)` use opposite edge-inclusion conventions; verify boundaries when porting.

### Don't commit data or large binaries
Datasets (`*.parquet`, `*.csv`), model weights, and demo media (`.mov`/`.mp4`) never live in git. Training data loads via `nfl_data_py` at workflow runtime; demo videos go to YouTube and are linked from [README.md](README.md). For new CI data dependencies, fetch in the workflow step — do not stash a file in the repo to "make CI green."

### Stop rules — things that have been tried and reverted
These have all been attempted, shipped, and reverted. Re-proposing them costs a round-trip; don't.

- **Shared-venv CI optimization** — reverted in PRs [#110](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/110) / [#111](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/111) (2026-04-23). Artifact download (~25s/shard) is slower than the warm `uv` install (~10s). Wall-clock is the metric, not compute.
- **Module-level pre-warm under gunicorn `--preload`** — reverted in PRs [#148](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/148) / [#149](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/149) (2026-04-27). The bind happens *after* preload import; a slow pre-warm causes ALB TCP-refused → unhealthy. Use a `post_fork` hook or a background thread instead.
- **Training models directly on `fantasy_points`** — cross-link to "Raw-stat targets" above. Re-introducing this regresses the ~1.9 pt/game double-count fix in TODO.md's archive.
- **Promoting rolling / L3 / L5 / L8 / ewma / trend features into `ATTN_STATIC_FEATURES`** — cross-link to "Attention static-feature whitelist" above. The static branch is deliberately non-temporal; temporal signal already feeds attention through `ATTN_HISTORY_STATS`. Do not pitch this as a way to "close the gap to LightGBM" — the input availability is not the gap; the architecture is.
- **Adding loss-config knobs (`HUBER_DELTAS`, `LOSS_WEIGHTS`, `head_losses`, `gated_targets`) to [src/tuning/tune_nn.py](src/tuning/tune_nn.py)'s search space** — cross-link to "Loss weights are tuned inverse-to-Huber-delta" above. `LOSS_WEIGHTS ≈ 2.0 / HUBER_DELTAS` is a coupling, not two independent axes; Optuna sampling them independently produces inconsistent pairs and searching deltas + deriving weights also blows up dimensionality past what ~30 trials resolve. Hand-tune loss-config via the [src/tuning/ablate_rb_gate.py](src/tuning/ablate_rb_gate.py) pattern instead (hardcoded variants, side-by-side decision table).

## Running code

Commands live in [SETUP.md](SETUP.md). Shortcuts:
- `python -m src.benchmarking.benchmark [POS ...]` — benchmark & refresh artifacts (writes a `{run_id}.json` file under `benchmark_history/`).
- `python -m src.{pos}.run_pipeline` — single position, full local run.
- `pytest -m unit` — fast subset, runs in seconds. `pytest` for the full suite (requires `data/splits/*.parquet`).
- `ruff check . && ruff format --check .` — lint/format gate used by CI.

## CI & training

- `tests.yml` — ruff + pytest on push/PR. Installs via `uv` (migrated in `3c897d8`) and shards pytest across `QB/RB/WR/TE/K/DST/shared` matrix jobs (per-position paths under `tests/{pos}/`; the `shared` shard runs `tests/` excluding the per-position dirs). Each shard uploads coverage to Codecov under a matching flag; the project target is **80% per component/flag** (see [codecov.yml](codecov.yml)). Diagnostic CLIs (`src/qb/diagnose_outliers.py`, `src/rb/analyze_errors.py`) are excluded from the coverage denominator. If `Run Tests` silently stops firing on rapid force-push cadence (occasional GitHub Actions bug), run `pytest` locally and merge with `gh pr merge --squash`.
- `batch-image.yml` → `train-batch.yml` OR `train-ec2.yml` — image build triggers training; which workflow fires is controlled by the `BATCH_ACTIVE` repo variable, currently `true` (default since 2026-05-20). `true` → parallel Spot fan-out via `train-batch.yml` (six g4dn.xlarge Spot instances, one per position; **measured 2026-05-21: ~10 min "Submit Batch jobs and wait" + ~3 min "Refresh ECS service" after ALB target-group tuning (was ~8 min before) = ~15 min total train job wall-clock**, vs. the original ~25–30 min design estimate. See [docs/batch_design.md](docs/batch_design.md), D13). `false` (rollback) → warm-EC2 path via `train-ec2.yml` (~120 min sequential; see [docs/ec2_design.md](docs/ec2_design.md), D7/D9). `workflow_dispatch` on either workflow bypasses the gate (break-glass). Both paths use the same `detect` job (diff the merge commit, retrain only changed positions); the path → positions mapping is centralized in [src/scripts/scope_positions.py](src/scripts/scope_positions.py) and contract-tested by [tests/scripts/test_scope_positions.py](tests/scripts/test_scope_positions.py) — touch both when changing the global-trigger list. AWS quotas: g4dn.xlarge OD = 4 vCPU (one instance, EC2 path); Spot G+VT = 24 vCPU (six instances, Batch path) — sized exactly for the fan-out.
- `deploy.yml` — ECS Flask deploy.

AWS-side operational notes live in auto-memory (GPU quota, training path, CI anomaly) — Claude loads those automatically, so this file doesn't duplicate them.

## Scheduled routines

A Claude Code cloud routine (a claude.ai scheduled remote agent) runs a **codebase audit** every 2h: it fans out parallel auditor subagents, dedupes against open+closed `claude-audit` GitHub issues, and files **one issue per finding** — labeled `claude-audit` + severity (`severity-high`/`severity-medium`) + area (EXCEPT `docs` findings, which all collapse into one persistent consolidated issue, `[claude-audit] docs: consolidated documentation findings`) — plus one closed `[claude-audit] checkpoint` issue per fire recording the audited SHA (HIGH/MED findings only; severity lives in the label, not the title). Those per-finding issues are what the [`solve-issues`](.claude/skills/solve-issues/SKILL.md) skill triages, orders by severity, and bundles into tier-by-risk PRs — this is the producer side of that loop. The actionable backlog is `gh issue list --label claude-audit --label severity-high` (checkpoints carry no severity label, so they're excluded).

The routine's prompt is version-controlled here, not only in the dashboard. **[.claude/routines/audit/](.claude/routines/audit/) is the source of truth:**
- [`prompt.md`](.claude/routines/audit/prompt.md) — the full audit prompt (what actually runs).
- [`shim.md`](.claude/routines/audit/shim.md) — the thin pointer deployed to the dashboard; at run time it reads `prompt.md` from the `main` checkout and executes it verbatim.
- [`config.json`](.claude/routines/audit/config.json) — deploy params (trigger id, model, cron, environment, allowed tools).
- [`README.md`](.claude/routines/audit/README.md) — the edit/push recipe and gotchas.

**To change what the audit does:** edit `prompt.md`, merge to `main` — the next fire picks it up, no dashboard push needed. **To change a deploy param or the shim itself:** edit `config.json` / `shim.md`, then invoke `/schedule` and say "push the audit routine" (see the README). `prompt.md` must stay at exactly that path on `main`, or the shim stops without auditing rather than improvising.

## Worktree workflow

This repo is regularly worked from `.claude/worktrees/<name>` clones where the parent worktree holds `main`. Three quirks:

- **Edit/Write paths must carry the worktree prefix.** Sub-agents (Explore/Plan) and plan files report repo-relative or *parent*-absolute paths (`/…/Final-Project/src/foo.py`). Using those verbatim for Edit/Write silently writes to the parent (`main`'s checkout, or whatever branch it holds) instead of this feature branch — `git status` in the worktree stays clean and any benchmark re-run uses the *unchanged* code (MAE Δ=0.0000 on every target is the late smell). **Before the first Edit/Write, re-prefix the path to `/…/Final-Project/.claude/worktrees/<name>/…`, then `grep` the new symbol in the worktree file to confirm the edit landed there.** Recovery: `cp <parent>/<file> <worktree>/<file>` (if committed baselines match) or `git -C <parent> diff -- <file> | git -C <worktree> apply`, then `git -C <parent> checkout -- <file>`. Burned ~30 min GPU time on PR #284; recurred on #354 and #370.
- **`gh pr merge --delete-branch` fails** in a worktree (it tries to `git checkout main`, which is held by the parent). Use `gh pr merge <N> --squash` then `git push origin --delete <branch>` separately. Local feature branch can stay.
- **"Is X on `main`?" / dead-link checks** must read `origin/main:<path>` via `git fetch origin main --quiet && git show origin/main:<path>` — never `cat <path>` in the worktree, which lags `main`.

## When making changes
- **Before `gh pr create`, invoke the `pre-pr-judge` skill.** The deterministic hook at [.claude/hooks/pre-pr.sh](.claude/hooks/pre-pr.sh) gates ruff / pytest / benchmark freshness. The [pre-pr-judge skill](.claude/skills/pre-pr-judge/SKILL.md) spawns a worker subagent to diff the branch against `origin/main`, compare against the original task, and flag scope creep — the "agent did more than I asked" failure mode behind the reverted shared-venv ([#110](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/110) / [#111](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/111)) and gunicorn `--preload` ([#148](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/148) / [#149](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/149)) PRs. Skip only for trivial changes — see the skill for the skip list.
- **After `gh pr create`, the [.claude/hooks/post-pr-create.sh](.claude/hooks/post-pr-create.sh) PostToolUse hook injects an autonomous follow-up workflow.** Claude's next turn (1) runs `git fetch origin main --quiet && git rebase origin/main` to fold in any newly merged PRs and force-pushes if the rebase changed anything (aborts + surfaces to user on conflict), (2) invokes the `/review` skill on the PR, (3) auto-applies **every** finding via Edit/Write **except** architectural / design-level ones — nit/minor/style, naming, formatting, dead imports, local logic fixes, security fixes with localized edits, missing-test additions, missing edge-case handlers, doc/comment tweaks, etc., (4) commits any auto-applied fixes as `review: address /review nits` and pushes, (5) if any architectural / design-level finding was surfaced (rethink-the-approach, multi-file refactor outside scope, new abstraction, design-judgment question), stops and lists them for the user, (6) otherwise, waits for green CI via `gh pr checks <N> --watch` then auto-merges with `gh pr merge <N> --squash` and deletes the remote branch via `git push origin --delete <branch>` (worktree-safe; see the worktree section). CI failure → stop and surface; never `--admin` bypass. The hook itself only injects the instruction — Claude executes the workflow. Pairs with the `pre-pr-judge` bullet above (pre/post around `gh pr create`).
- **Open a PR, wait for green CI, then merge.** Push to a feature branch, open the PR with `gh pr create`, then `gh pr checks <N> --watch` until every check passes before running `gh pr merge <N> --squash`. Don't merge with red or pending checks; if a check fails, fix the underlying issue rather than bypassing with `--admin`. Exception: the `Run Tests` silent-stop bug noted in the CI section — run `pytest` locally and merge.
- **`[docs-only]` commit-subject opt-in for comment/docstring/import-reorder PRs.** When every change in the PR is non-behavioural (a comment text fix, a docstring update, an `is*` typo, ruff I001 import reordering) and you are 100% certain there is no metric or runtime impact, include `[docs-only]` in at least one commit **subject line** in the PR (the squash-merge subject becomes the PR title, and squash bodies preserve constituent subjects as `* `-prefixed bullets — both kinds count). **Prose mentions in commit bodies do not count and never have to** — the four consumers below all use a subject-line awk filter (`awk 'NR==1 || /^\* /'`). Previously they did a flat substring match against the full commit message, which false-positived on PRs whose commits *described* the `[docs-only]` mechanism: PR [#293](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/293) (BF16 perf change) skipped its own image rebuild because a constituent CI commit's body wrote "the [docs-only] opt-in now also clears…", and the original feature PR's title literally contained the tag and tripped its own gate. See [TODO.md](TODO.md)'s Fixed archive for the postmortem. The tag is respected by:
  - [tests.yml](.github/workflows/tests.yml)'s `detect` job — emits an empty shard matrix; `tests-pass` reports green via the `skipped` branch.
  - [batch-image.yml](.github/workflows/batch-image.yml)'s `check-docs-only` leading job — skips `build-and-push` entirely (image rebuild is wasteful when runtime behaviour is unchanged).
  - [_detect-positions.yml](.github/workflows/_detect-positions.yml) (shared by `train-batch.yml` and `train-ec2.yml`) — emits empty `positions`; cascades to training-job skip.
  - [.claude/hooks/pre-pr.sh](.claude/hooks/pre-pr.sh) — early-exits before B1 (ruff/pytest) and B2 (benchmark) gates, so a flaky test or worktree env issue doesn't block a clean docs PR locally.
  `lint` + `detect` still run as cheap sanity gates. [deploy.yml](.github/workflows/deploy.yml) is **not** gated by the tag — its `paths:` filter on `docs/**` + `README.md` + wiki sources is the correct gate (docs changes are rendered by `src.serving.app._render_wiki_doc` in the in-app wiki tab, so they need redeploy to land on alexfree.me). Trust contract — CI cannot verify the assertion, the author owns correctness. Don't reach for the tag to bypass a flaky test; if `[docs-only]` is wrong, the next substantive PR's CI will surface the regression with no signal about which commit introduced it.
- Respect the **TODO.md archive** — it encodes the project's accumulated "already tried" knowledge.
- **Update the ADR + decision log alongside non-trivial changes.**
  - **ADR ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):** if your change touches an existing decision (D1–D15) or introduces a new one of similar weight, update the affected D-entry's `Decision`/`Context`/`Chosen`/`Rejected`/`References`/`Consequence` fields, add a line to the `Update history` block at the top (date + one-line summary + commit/PR), and link the commit or PR under `References`. Superseding a decision? Mark the old D-entry as superseded and add a new D-entry — don't rewrite the original.
  - **Decision log ([TODO.md](TODO.md)):** for non-trivial bug fixes, move the corresponding `Open` entry into the `Fixed archive` using the existing `### [FIXED] Title` + **File(s)** (paths + commit SHA) / **What** / **Fix** / **Lesson** format. If the bug wasn't tracked, add a fresh archive entry anyway — that's the project's "already tried" knowledge.
  - **Skip both** for truly trivial changes (typos, formatting, lockfile bumps, comment-only tweaks). When in doubt, lean toward writing the entry — a thin archive is the failure mode, not over-documentation.
- Update tests and fixtures when you change feature lists or targets (archive has multiple entries where this was missed).
- Don't add error handling, fallbacks, or validation for cases that can't happen (see top-level CLAUDE-Code guidance on scope). One exception: network/data-source boundaries are real and should be defensive.
- **For NN/feature/loss/target changes, run the actual pipeline before merging.** `pytest -m unit` and CI tests don't catch metric regressions. Run `python -m src.{pos}.run_pipeline` on the affected position and diff `benchmark_history/` output against the prior run. The K refactor regression and the QB metrics-label bug both shipped because tests passed without the pipeline being run.
- **Sub-agent contract — two shapes, picked by batch size.**
  - **>10 items (default for code-review remediation and similar multi-bundle cleanups):** spawn **as many file-disjoint worker sub-agents as possible** (max the parallelism), but workers DRAFT commits in their isolated worktrees and do **NOT** push or open PRs. The orchestrator cherry-picks each risk-tier's worker commits onto a fresh staging branch from `origin/main` and opens **one PR per tier** (safest → highest-risk). The 113-finding remediation captured in [TODO.md](TODO.md)'s `Code-review remediation: 110 of 113 findings…` archive entry is the canonical example: 12 file-disjoint bundles → 3 PRs ([#312](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/312) / [#314](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/314) / [#315](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/315)). See auto-memory `feedback_tier_by_risk_pr_consolidation` for the per-tier orchestrator recipe (worktree from `origin/main` → cherry-pick → symlink data → pytest+ruff → `gh pr create` with HEREDOC body listing bundles + cumulative benchmark deltas).
  - **<10 truly independent tasks:** the older per-worker-opens-PR shape is still fine — each worker commits, pushes, and runs `gh pr create` itself.
  - **Either shape:** workers MUST report ship-status back to the orchestrator (commit SHA, branch name, files modified, findings skipped + why). Orchestrator verifies via `gh pr list` or `git log <worker-branch>` — incomplete worker output (uncommitted state, no PR, no commit) is a recurring failure mode the orchestrator catches, not the user.
- **File-disjointness is for parallelism, not correctness.** File-disjoint worker bundles cherry-pick clean and combined `pytest -m unit` + ruff stay green, but the primitive does **NOT** protect against API-signature changes in shared code that downstream consumers in per-position bundles still call. The 2026-05-21 Tier A inter-bundle conflict (W.SHARED-A dropped `_train_nn`'s `position` parameter + `val_preds` return value under L-S10/L-S11; W.QB independently collapsed `src/qb/diagnose_outliers.py`'s NN loop into a `_train_nn` call with the *old* signature under L-QB3; broken call site only surfaced at PR-review time because the CLI script is `__main__`-guarded so `pytest -m unit` never imported it) is the canonical instance. **When a worker bundle changes a shared-code signature**, the worker brief must include "grep for every caller of any function whose signature you change" as a discovery step — the worker reports the conflict back to the orchestrator (or rebundles), not leaves it for review to catch. Operator-only CLI scripts (`diagnose_outliers.py`, `analyze_errors.py`, `audit_features.py`) should have at least an import-smoke test in their per-position test suite so signature-change drift fails the unit-test shard rather than the PR-review pass.
- **After a non-routine session, invoke the `post-session-critique` skill.** Run when the user corrected your approach mid-flight, a non-obvious convention bit you, or something went unusually well because of a specific rule. Captures *prompt* lessons (proposed CLAUDE.md or auto-memory edits) the way [TODO.md](TODO.md)'s Fixed archive captures *code* lessons. Skip if the session was routine.
