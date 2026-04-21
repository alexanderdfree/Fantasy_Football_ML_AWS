# CLAUDE.md

Orientation file for Claude Code. Human-facing docs live elsewhere — this file exists to surface the conventions, gotchas, and "before you touch X, read Y" rules that aren't obvious from a first pass through the tree.

## Orient yourself first
- **[README.md](README.md)** — overview, architecture diagram, eval results.
- **[SETUP.md](SETUP.md)** — install, first-time data pull, how to run everything locally. If you need a command, it's probably here.
- **[TODO.md](TODO.md)** — open issues and a **Fixed archive** with root-cause + lesson for every non-trivial bug ever squashed. **Read this before proposing changes near anything it mentions** — most "obvious" fixes have been tried and the archive explains why they were wrong.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — design decisions with rejected alternatives.

## Project shape (six-position symmetry)
Each of `QB/ RB/ WR/ TE/ K/ DST/` follows the same template:

```
{POS}/
  {pos}_config.py     # hyperparams (Ridge alpha grids, NN dims, loss weights, Huber deltas, LightGBM params)
  {pos}_data.py       # loading + temporal split specifics
  {pos}_features.py   # position-specific feature engineering
  {pos}_targets.py    # raw-stat target definitions
  run_{pos}_pipeline.py
  tests/
```

Shared plumbing is in [shared/](shared/): `pipeline.py` (train/eval loop), `models.py` (Ridge + MultiHeadNet), `neural_net.py` (attention), `aggregate_targets.py` (raw-stat → fantasy-point scoring), `training.py`, `evaluation.py`, `backtest.py`.

All six positions train an attention NN (DST landed via `cc0c627`, K via `801b61a`). There is no "skill-positions-only" carve-out anymore — if you're adding an NN-related knob, wire it through every position.

**Adding a new position**: copy an existing folder, rename files/constants, wire it into `batch/train.py` and `.github/workflows/train-ec2.yml`'s position list, add tests under `{POS}/tests/`.

## Conventions that bite if ignored

### Raw-stat targets, never fantasy-point targets
Every position predicts raw NFL stats (yards, TDs, receptions, etc.). Fantasy points are computed *after* prediction via `shared.aggregate_targets.predictions_to_fantasy_points(pos, preds)`. If you find yourself training a model directly on `fantasy_points`, stop — you'll break scoring-format flexibility and regress the ~1.9 pt/game double-count fix documented in TODO.md's archive.

### Feature whitelist is explicit, not inferred
`{POS}_INCLUDE_FEATURES` in each `{pos}_config.py` is an opt-in list. New columns must be added explicitly — the training code will *not* pick them up automatically. This prevents silent feature leakage. When you add a feature, update both the feature-engineering file *and* the include dict, then update the test fixture (`tests/conftest.py` or `{POS}/tests/conftest.py`).

### Attention static-feature whitelist is separate per position
The attention NN's static branch reads a *second*, smaller allowlist: `{POS}_ATTN_STATIC_FEATURES` (commit `2500ecc`). It is defined per position (QB/RB/WR/TE derive it from a `{POS}_ATTN_STATIC_CATEGORIES` subset of `{POS}_INCLUDE_FEATURES`; DST/K enumerate it directly) and deliberately excludes rolling/ewma/trend columns so the attention branch doesn't double-count signal it already learns from `{POS}_ATTN_HISTORY_STATS`. Adding a feature to `{POS}_INCLUDE_FEATURES` does **not** feed it into attention — add it to `{POS}_ATTN_STATIC_FEATURES` too if that's what you want.

### Loss weights are tuned inverse-to-Huber-delta
`{POS}_LOSS_WEIGHTS` ≈ `2.0 / {POS}_HUBER_DELTAS[target]`. The rationale is baked into QB's config comment ([QB/qb_config.py](QB/qb_config.py)): without this rebalance, yards targets (δ=15–25) dominated count heads (δ=0.5) ~2500× per sample and the count heads collapsed to the mean. If you retune a Huber delta, re-derive the matching loss weight — don't change one without the other.

### `non_negative_targets` is per-head, not global
The NN clamps outputs to ≥ 0 per head. Positions opt in by wiring `{POS}_NN_NON_NEGATIVE_TARGETS` into their config's `nn_non_negative_targets` key (see `DST/dst_config.py` and `K/k_config.py`). If a position ever adds a signed head (e.g. a bonus that can go negative), exclude it from that set explicitly rather than flipping the behaviour globally. If you construct `MultiHeadNet(...)` anywhere outside `shared/pipeline.py::_train_nn`, mirror the kwarg — the CV path was missed once (see TODO.md archive).

### Always diff training vs inference paths
The training pipeline in `shared/pipeline.py` and the serving code in `app.py` both build features. They have drifted silently in the past (weather/Vegas merge in training but not serving; scaler clip in one path but not the other). If you touch feature building in either, check the other.

## Running code

Commands live in [SETUP.md](SETUP.md). Shortcuts:
- `python benchmark.py [POS ...]` — benchmark & refresh artifacts (append row to `benchmark_history.json`).
- `python {POS}/run_{pos}_pipeline.py` — single position, full local run.
- `pytest -m unit` — fast subset, runs in seconds. `pytest` for the full suite (requires `data/splits/*.parquet`).
- `ruff check . && ruff format --check .` — lint/format gate used by CI.

## CI & training

- `tests.yml` — ruff + pytest on push/PR. Installs via `uv` (migrated in `3c897d8`) and shards pytest across `QB/RB/WR/TE/K/DST/shared` matrix jobs. If `Run Tests` silently stops firing on rapid force-push cadence (occasional GitHub Actions bug), run `pytest` locally and merge with `gh pr merge --squash`.
- `batch-image.yml` → `train-ec2.yml` — image build triggers EC2 training. The `detect` job diffs the merge commit and only retrains positions whose code changed. AWS g4dn.xlarge OD quota is 4 vCPU (one instance); spot quota is higher. Check quota before dispatching if you touch infra.
- `deploy.yml` — ECS Flask deploy.

AWS-side operational notes live in auto-memory (GPU quota, training path, CI anomaly) — Claude loads those automatically, so this file doesn't duplicate them.

## When making changes
- Respect the **TODO.md archive** — it encodes the project's accumulated "already tried" knowledge.
- Update tests and fixtures when you change feature lists or targets (archive has multiple entries where this was missed).
- Don't add error handling, fallbacks, or validation for cases that can't happen (see top-level CLAUDE-Code guidance on scope). One exception: network/data-source boundaries are real and should be defensive.
