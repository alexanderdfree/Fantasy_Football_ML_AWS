# CLAUDE.md

Orientation file for Claude Code. Human-facing docs live elsewhere — this file exists to surface the conventions, gotchas, and "before you touch X, read Y" rules that aren't obvious from a first pass through the tree.

## Orient yourself first
- **[README.md](README.md)** — overview, architecture diagram, eval results.
- **[SETUP.md](SETUP.md)** — install, first-time data pull, how to run everything locally. If you need a command, it's probably here.
- **[TODO.md](TODO.md)** — open issues and a **Fixed archive** with root-cause + lesson for every non-trivial bug ever squashed. **Read this before proposing changes near anything it mentions** — most "obvious" fixes have been tried and the archive explains why they were wrong. **Update it as you ship**: move Open → Fixed archive (or add a fresh archive entry) using the existing `### [FIXED] Title` + **File(s)/What/Fix/Lesson** format.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the project's ADR (decisions D1–D14 + a dated `Update history`), with rejected alternatives. **Living doc** — update it whenever a non-trivial change touches or adds an architectural decision (see "When making changes").

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

Shared plumbing is in [src/shared/](src/shared/): `pipeline.py` (train/eval loop), `models.py` (Ridge + MultiHeadNet), `neural_net.py` (attention), `aggregate_targets.py` (raw-stat → fantasy-point scoring), `training.py`, `evaluation.py`, `backtest.py`.

The rest of `src/` groups by purpose: `src/serving/` (Flask app + assets), `src/batch/` (training orchestration), `src/benchmarking/`, `src/tuning/` (Optuna + ablations), `src/analysis/` (post-hoc analyses), `src/scripts/` (operator CLIs).

All six positions train an attention NN (DST landed via `cc0c627`, K via `801b61a`). There is no "skill-positions-only" carve-out anymore — if you're adding an NN-related knob, wire it through every position.

**Adding a new position**: copy an existing folder under `src/`, rename files/constants, wire it into `src/batch/train.py` and the position list in both `.github/workflows/train-batch.yml` (active path) and `.github/workflows/train-ec2.yml` (rollback path), add tests under `tests/{pos}/`.

## Conventions that bite if ignored

### Raw-stat targets, never fantasy-point targets
Every position predicts raw NFL stats (yards, TDs, receptions, etc.). Fantasy points are computed *after* prediction via `src.shared.aggregate_targets.predictions_to_fantasy_points(pos, preds)`. If you find yourself training a model directly on `fantasy_points`, stop — you'll break scoring-format flexibility and regress the ~1.9 pt/game double-count fix documented in TODO.md's archive.

### Feature whitelist is explicit, not inferred
`INCLUDE_FEATURES` in each `src/{pos}/config.py` is an opt-in list. New columns must be added explicitly — the training code will *not* pick them up automatically. This prevents silent feature leakage. When you add a feature, update both the feature-engineering file *and* the include dict, then update the test fixture (`tests/conftest.py` or `tests/{pos}/conftest.py`).

### Attention static-feature whitelist is separate per position
The attention NN's static branch reads a *second*, smaller allowlist: `ATTN_STATIC_FEATURES` (commit `2500ecc`). It is defined per position (QB/RB/WR/TE derive it from an `ATTN_STATIC_CATEGORIES` subset of `INCLUDE_FEATURES`; DST/K enumerate it directly) and deliberately excludes rolling/ewma/trend columns so the attention branch doesn't double-count signal it already learns from `ATTN_HISTORY_STATS`. Adding a feature to `INCLUDE_FEATURES` does **not** feed it into attention — add it to `ATTN_STATIC_FEATURES` too if that's what you want.

### Loss weights are tuned inverse-to-Huber-delta
`LOSS_WEIGHTS` ≈ `2.0 / HUBER_DELTAS[target]`. The rationale is baked into QB's config comment ([src/qb/config.py](src/qb/config.py)): without this rebalance, yards targets (δ=15–25) dominated count heads (δ=0.5) ~2500× per sample and the count heads collapsed to the mean. If you retune a Huber delta, re-derive the matching loss weight — don't change one without the other.

### `non_negative_targets` is per-head, not global
The NN clamps outputs to ≥ 0 per head. Default (`non_negative_targets=None` in `MultiHeadNet`) clamps *every* head, which is what QB/RB/WR/TE rely on — they don't set the config key. K and DST set `NN_NON_NEGATIVE_TARGETS = set(TARGETS)` explicitly (same effect as the default, written out for clarity); if a position ever adds a signed head (e.g. a bonus that can go negative), pass a set that *excludes* that head rather than flipping the behaviour globally. If you construct `MultiHeadNet(...)` anywhere outside `src/shared/pipeline.py::_train_nn`, mirror the `non_negative_targets=cfg.get("nn_non_negative_targets")` kwarg — the CV path was missed once (see TODO.md archive).

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

## Running code

Commands live in [SETUP.md](SETUP.md). Shortcuts:
- `python -m src.benchmarking.benchmark [POS ...]` — benchmark & refresh artifacts (writes a `{run_id}.json` file under `benchmark_history/`).
- `python -m src.{pos}.run_pipeline` — single position, full local run.
- `pytest -m unit` — fast subset, runs in seconds. `pytest` for the full suite (requires `data/splits/*.parquet`).
- `ruff check . && ruff format --check .` — lint/format gate used by CI.

## CI & training

- `tests.yml` — ruff + pytest on push/PR. Installs via `uv` (migrated in `3c897d8`) and shards pytest across `QB/RB/WR/TE/K/DST/shared` matrix jobs (per-position paths under `tests/{pos}/`; the `shared` shard runs `tests/` excluding the per-position dirs). Each shard uploads coverage to Codecov under a matching flag; the project target is **80% per component/flag** (see [codecov.yml](codecov.yml)). Diagnostic CLIs (`src/qb/diagnose_outliers.py`, `src/rb/analyze_errors.py`) are excluded from the coverage denominator. If `Run Tests` silently stops firing on rapid force-push cadence (occasional GitHub Actions bug), run `pytest` locally and merge with `gh pr merge --squash`.
- `batch-image.yml` → `train-batch.yml` OR `train-ec2.yml` — image build triggers training; which workflow fires is controlled by the `BATCH_ACTIVE` repo variable, currently `true` (default since 2026-05-20). `true` → parallel Spot fan-out via `train-batch.yml` (six g4dn.xlarge Spot instances, one per position, ~25–30 min wall-clock; see [docs/batch_design.md](docs/batch_design.md), D13). `false` (rollback) → warm-EC2 path via `train-ec2.yml` (~120 min sequential; see [docs/ec2_design.md](docs/ec2_design.md), D7/D9). `workflow_dispatch` on either workflow bypasses the gate (break-glass). Both paths use the same `detect` job (diff the merge commit, retrain only changed positions); the path → positions mapping is centralized in [src/scripts/scope_positions.py](src/scripts/scope_positions.py) and contract-tested by [tests/scripts/test_scope_positions.py](tests/scripts/test_scope_positions.py) — touch both when changing the global-trigger list. AWS quotas: g4dn.xlarge OD = 4 vCPU (one instance, EC2 path); Spot G+VT = 24 vCPU (six instances, Batch path) — sized exactly for the fan-out.
- `deploy.yml` — ECS Flask deploy.

AWS-side operational notes live in auto-memory (GPU quota, training path, CI anomaly) — Claude loads those automatically, so this file doesn't duplicate them.

## Worktree workflow

This repo is regularly worked from `.claude/worktrees/<name>` clones where the parent worktree holds `main`. Two quirks:

- **`gh pr merge --delete-branch` fails** in a worktree (it tries to `git checkout main`, which is held by the parent). Use `gh pr merge <N> --squash` then `git push origin --delete <branch>` separately. Local feature branch can stay.
- **"Is X on `main`?" / dead-link checks** must read `origin/main:<path>` via `git fetch origin main --quiet && git show origin/main:<path>` — never `cat <path>` in the worktree, which lags `main`.

## When making changes
- **Before `gh pr create`, invoke the `pre-pr-judge` skill.** The deterministic hook at [.claude/hooks/pre-pr.sh](.claude/hooks/pre-pr.sh) gates ruff / pytest / benchmark freshness. The [pre-pr-judge skill](.claude/skills/pre-pr-judge/SKILL.md) spawns a worker subagent to diff the branch against `origin/main`, compare against the original task, and flag scope creep — the "agent did more than I asked" failure mode behind the reverted shared-venv ([#110](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/110) / [#111](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/111)) and gunicorn `--preload` ([#148](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/148) / [#149](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/149)) PRs. Skip only for trivial changes — see the skill for the skip list.
- **Open a PR, wait for green CI, then merge.** Push to a feature branch, open the PR with `gh pr create`, then `gh pr checks <N> --watch` until every check passes before running `gh pr merge <N> --squash`. Don't merge with red or pending checks; if a check fails, fix the underlying issue rather than bypassing with `--admin`. Exception: the `Run Tests` silent-stop bug noted in the CI section — run `pytest` locally and merge.
- Respect the **TODO.md archive** — it encodes the project's accumulated "already tried" knowledge.
- **Update the ADR + decision log alongside non-trivial changes.**
  - **ADR ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):** if your change touches an existing decision (D1–D13) or introduces a new one of similar weight, update the affected D-entry's `Decision`/`Context`/`Chosen`/`Rejected`/`References`/`Consequence` fields, add a line to the `Update history` block at the top (date + one-line summary + commit/PR), and link the commit or PR under `References`. Superseding a decision? Mark the old D-entry as superseded and add a new D-entry — don't rewrite the original.
  - **Decision log ([TODO.md](TODO.md)):** for non-trivial bug fixes, move the corresponding `Open` entry into the `Fixed archive` using the existing `### [FIXED] Title` + **File(s)** (paths + commit SHA) / **What** / **Fix** / **Lesson** format. If the bug wasn't tracked, add a fresh archive entry anyway — that's the project's "already tried" knowledge.
  - **Skip both** for truly trivial changes (typos, formatting, lockfile bumps, comment-only tweaks). When in doubt, lean toward writing the entry — a thin archive is the failure mode, not over-documentation.
- Update tests and fixtures when you change feature lists or targets (archive has multiple entries where this was missed).
- Don't add error handling, fallbacks, or validation for cases that can't happen (see top-level CLAUDE-Code guidance on scope). One exception: network/data-source boundaries are real and should be defensive.
- **For NN/feature/loss/target changes, run the actual pipeline before merging.** `pytest -m unit` and CI tests don't catch metric regressions. Run `python -m src.{pos}.run_pipeline` on the affected position and diff `benchmark_history/` output against the prior run. The K refactor regression and the QB metrics-label bug both shipped because tests passed without the pipeline being run.
- **Sub-agent contract.** When dispatching parallel worker agents, each worker must commit, push its branch, and open the PR as part of its task. Before reporting completion, verify with `gh pr list` that every worker shipped — incomplete worker output (uncommitted worktree state, no PR) is a recurring failure mode and the orchestrator must catch it, not the user.
- **After a non-routine session, invoke the `post-session-critique` skill.** Run when the user corrected your approach mid-flight, a non-obvious convention bit you, or something went unusually well because of a specific rule. Captures *prompt* lessons (proposed CLAUDE.md or auto-memory edits) the way [TODO.md](TODO.md)'s Fixed archive captures *code* lessons. Skip if the session was routine.
