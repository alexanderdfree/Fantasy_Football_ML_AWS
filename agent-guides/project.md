# Project layout

Read only the sections relevant to the task. [AGENTS.md](../AGENTS.md) supplies the shared entrypoint; current code/config and linked decisions supply operational state. Dated measurements describe their recorded regime, not a promise about today.

## Project shape (six-position symmetry)
Each of `src/qb/ src/rb/ src/wr/ src/te/ src/k/ src/dst/` follows the same template:

```
src/{pos}/
  config.py        # hyperparams (Ridge alpha grids, NN dims, loss weights, Huber deltas, LightGBM params)
  data.py          # loading + temporal split specifics
  features.py      # position-specific feature engineering
  targets.py       # raw-stat target definitions
  run_pipeline.py  # exposes run() and run_cv()
```

Tests for each position live under `tests/{pos}/`.

Shared plumbing is in [src/shared/](../src/shared): `pipeline.py` (train/eval loop), `models.py` (single-target `RidgeModel`/`ElasticNetModel`/`SeasonAverageBaseline`, multi-target wrappers `RidgeMultiTarget`/`ElasticNetMultiTarget`/`LightGBMMultiTarget`/`TabPFNMultiTarget` (the last is an **opt-in, default-off** 5th comparison variant — TabPFN-3 pretrained tabular transformer, pinned to the `tabpfn` 8.x default; non-commercial license so benchmark-only / never served; not enabled for any position, `tabpfn` not in requirements, see [docs/adr/0003-three-way-model-comparison-no-ensemble.md](../docs/adr/0003-three-way-model-comparison-no-ensemble.md)), plus `TwoStageRidge` and gated-ordinal classifiers), `neural_net.py` (attention + gated NN heads), `aggregate_targets.py` (raw-stat → fantasy-point scoring), `training.py`, `evaluation.py`, `backtest.py`. The root-level `models/` dir is a separate placeholder for trained artifacts that load from S3 — different beast.

The rest of `src/` groups by purpose:
- `src/data/` — cross-position data loading + temporal split (per-position `data.py` files wrap these): `loader.py`, `nflcom_loader.py`, `preprocessing.py`, `redzone_pbp.py`, `split.py`.
- `src/features/engineer.py` — cross-position feature engineering coordinator.
- `src/shared/evaluation.py` — position-aware visualization/aggregation layer plus the `compute_metrics(y_true, y_pred)` helper used by backtest and pipeline.
- `src/serving/` — Flask app + assets. The dashboard UI is React: **sources in `src/serving/frontend/` (edit these), committed esbuild bundle at `src/serving/static/js/app.js` (never edit; `cd src/serving/frontend && npm run build` regenerates — CI's `frontend-bundle` job fails a PR whose bundle is stale)**. See SETUP.md § "Frontend build" + ADR-0023.
- `src/batch/` — training orchestration (AWS Batch path). New tuner/ablation files go in `src/tuning/`, **never** here — files under `src/batch/` trigger a full 6-position retrain via [src/scripts/scope_positions.py](../src/scripts/scope_positions.py), except names containing `tune`/`ablate` and the exact basenames `launch.py` / `benchmark.py` (job submission / read-only aggregation). PR #280 burned ~4 GPU-jobs from a tuner-only change placed here.
- `src/benchmarking/`, `src/tuning/` — Optuna + ablations.
- `src/analysis/` — post-hoc analyses.
- `src/scripts/` — operator CLIs.
- `src/config.py` — global constants (`SEASONS`, `POSITIONS`, scoring dicts, `TOP_K_RANKING`). Distinct from per-position `src/{pos}/config.py`, which holds model hyperparams.

All six positions train an attention NN (DST landed via `cc0c627`, K via `801b61a`). There is no "skill-positions-only" carve-out anymore — if you're adding an NN-related knob, wire it through every position.

**Adding a new position**: copy an existing `src/` folder, rename files/constants, add it to the `Position` StrEnum in [src/shared/position.py](../src/shared/position.py) — the canonical list [src/shared/registry.py](../src/shared/registry.py) exposes via `Position.values()` and `src/batch/train.py` dispatches off (no per-position dict in `train.py` anymore) — and the position list in `.github/workflows/_detect-positions.yml` (shared by `train-batch.yml` (active) and `train-ec2.yml` (rollback)). Also update [src/scripts/scope_positions.py](../src/scripts/scope_positions.py) — the canonical path → positions mapping (contract-tested by [tests/scripts/test_scope_positions.py](../tests/scripts/test_scope_positions.py)) used by both workflows' `detect` job. Add tests under `tests/{pos}/`.
