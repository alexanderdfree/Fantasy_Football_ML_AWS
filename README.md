# Fantasy Football Weekly Points Predictor

A per-position machine learning system that predicts weekly NFL fantasy points for QBs, RBs, WRs, TEs, Kickers, and D/STs — comparing a Ridge baseline, a custom PyTorch multi-head neural network (with an attention variant at every position), and LightGBM across the 2012–2025 seasons. Served as a Flask dashboard at [alexfree.me](https://alexfree.me).

Personal project. Solo, ongoing.

## What it Does

The system ingests weekly NFL data from [nflverse](https://github.com/nflverse) (player stats, rosters, schedules, snap counts), engineers rolling/EWMA/share/matchup features plus Vegas odds and weather joins, and trains one model per position family per architecture. Each position is evaluated against actual 2025 fantasy output in three scoring formats (Standard, Half-PPR, Full PPR). A Flask dashboard lets users look up any player and compare Ridge / neural net / LightGBM projections side-by-side with the real result.

## Research Question

> Can a multi-head neural network with engineered temporal features and target decomposition meaningfully outperform Ridge regression at predicting weekly fantasy output, and what features matter most?

The answer differs by position — see the Evaluation section below. Each position's model predicts a small set of raw NFL stats rather than pre-scored fantasy-point buckets, and a deterministic aggregator ([src/shared/aggregate_targets.py](src/shared/aggregate_targets.py)) converts those raw stats to fantasy points under any scoring format. The raw targets per position:

- **QB** (6): `passing_yards`, `rushing_yards`, `passing_tds`, `rushing_tds`, `interceptions`, `fumbles_lost`
- **RB** (6): `rushing_tds`, `receiving_tds`, `rushing_yards`, `receiving_yards`, `receptions`, `fumbles_lost`
- **WR** (4): `receiving_tds`, `receiving_yards`, `receptions`, `fumbles_lost`
- **TE** (4): `receiving_tds`, `receiving_yards`, `receptions`, `fumbles_lost`
- **K** (4): `fg_yard_points`, `pat_points`, `fg_misses`, `xp_misses` — signs `[+1, +1, -1, -1]` applied at aggregation
- **DST** (10): `def_sacks`, `def_ints`, `def_fumble_rec`, `def_fumbles_forced`, `def_safeties`, `def_tds`, `def_blocked_kicks`, `special_teams_tds`, `points_allowed`, `yards_allowed` — aggregated via NFL-standard linear coefficients plus PA/YA tier bonuses

K was out of scope for the raw-stat migration (its heads were already raw counts/values); DST was migrated to the 10-target raw-stat decomposition in commit `cc0c627`.

Sharing a backbone across per-target heads is what makes the neural net competitive at small sample sizes; reporting MAE in raw units (yards / TDs / receptions) keeps per-target accuracy interpretable and decouples model error from scoring-format choice. Design rationale is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Architecture at a glance

```
nflverse API ─┐
(2012–2025)   │
              ▼
  ┌──────────────────────┐    ┌──────────────────────────────────────────┐
  │ src/data + src/features │▶ │ Three model families per position        │
  │ (rolling, EWMA, share,   │  │   Ridge  │  MultiHeadNet (PyTorch)       │
  │  matchup, weather/odds)  │  │          │  + Attention (all positions)  │
  └──────────────────────┘    │                LightGBM                  │
                               └──────────────────────────────────────────┘
                                                │
                                                ▼
                             ┌────────────────────────────────┐
                             │ GPU training (BATCH_ACTIVE)    │◀── GitHub Actions
                             │  true  → 6× Spot g4dn (Batch)  │    push to main
                             │          (default; ~15 min)    │
                             │  false → warm OD g4dn (SSM)    │
                             │          (rollback; ~120 min)  │
                             │ Same Dockerfile.train; D13/D7  │
                             └────────────────────────────────┘
                                                │ model.tar.gz
                                                ▼
                             ┌──────────────────────────────┐
                             │ S3 ──▶ Flask app on ECS       │
                             │        (CPU-only, 6 models   │
                             │         loaded in-memory)    │
                             └──────────────────────────────┘
```

Full decision log with rejected alternatives: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Start

Install and first-time setup: see [SETUP.md](SETUP.md).

Common commands once set up:

```bash
# Run the comparison benchmark (all positions) — writes one file per run to benchmark_history/
python -m src.benchmarking.benchmark

# Benchmark one position
python -m src.benchmarking.benchmark RB

# Serve the dashboard locally → http://localhost:5050
python -m src.serving.app

# Run the test suite
pytest
pytest -m unit        # fast tests only
```

Coverage is tracked on [Codecov](https://app.codecov.io/gh/alexanderdfree/Fantasy_Football_ML_AWS) with an **80% target per position and shared component** (see [codecov.yml](codecov.yml)). One-off diagnostic CLIs (`src/qb/diagnose_outliers.py`, `src/rb/analyze_errors.py`) are excluded from the denominator — everything else gets pulled in.

Full training on GPU runs in CI: by default a push to `main` fans out six Spot g4dn.xlarge instances via AWS Batch ([docs/batch_design.md](docs/batch_design.md), [infra/batch/README.md](infra/batch/README.md)). Setting `BATCH_ACTIVE=false` falls back to the warm-EC2 trainer ([docs/ec2_design.md](docs/ec2_design.md), [infra/ec2/README.md](infra/ec2/README.md)).

## Video Links

- Technical walkthrough (YouTube): https://youtu.be/eyuTnk3qLk8

## Evaluation

Holdout: 2025 season. Metric definitions: MAE (mean absolute error in fantasy points), R² (coefficient of determination), top-12 hit rate (agreement with the actual weekly top 12 at the position, PPR scoring). Numbers from [benchmark_history/2026-04-27T19-01-49_86fe9c0.json](benchmark_history/2026-04-27T19-01-49_86fe9c0.json) — the first full-positions EC2 run after the K nested attention (`801b61a`), DST raw-stat migration (`cc0c627`), RB feature audit cycle (PRs #190–#192), and the K/DST aggregate-metric fix (`0c66171`).

| Position | Ridge MAE | NN MAE | Attn NN MAE | LGBM MAE | Best | R² (best) | Top-12 (best) |
|---|---|---|---|---|---|---|---|
| QB  | 6.546 | 6.435 | 6.549 | **6.256** | LightGBM     | 0.269  | 0.548 |
| RB  | 4.399 | 4.278 | 4.315 | **4.159** | LightGBM     | 0.417  | 0.512 |
| WR  | 4.351 | 4.182 | **4.106** | 4.231 | Attention NN | 0.347  | 0.385 |
| TE  | 3.546 | 3.494 | **3.444** | 3.517 | Attention NN | 0.299  | 0.496 |
| K   | 4.079 | 4.128 | 4.132 | **4.061** | LightGBM     | −0.013 | 0.431 |
| DST | 5.212 | **5.162** | 5.346 | 5.274 | MultiHeadNet | 0.067  | 0.500 |

**Takeaways:**
- **LightGBM wins QB and RB.** It beats both NN families on the positions with the densest count-heavy targets (passing/rushing yards, TDs). The RB feature audit cycle (drop 14 redundant features, restore prior-season signals, add 3 orthogonal upstream aggregates) closed the gap between the audit-driven model variants — LightGBM is now the front-runner at RB by a small margin.
- **Attention NN wins WR and TE.** Sequence + interaction structure pays off when the target is heavily reception-driven. The attention pool's positional encoding catches recent-game weighting that pure rolling features lose.
- **K and DST report negative aggregate R².** This is the real signal, not a measurement bug: commit `0c66171` (PR #178) fixed the K/DST evaluation aggregator to use the same signed/tiered logic the serving layer uses, and the result is that all three model families struggle to beat predicting the per-player mean. K is the harder of the two — kickers genuinely have weak week-over-week signal at the available sample size. The per-position winners are decided by tiny margins well inside run-to-run variance (LightGBM edges K at 4.061 MAE; the Attention NN edges DST at 5.162) — no family is a clear winner here. This is consistent with the [docs/expert_comparison.md](docs/expert_comparison.md) finding that published expert projections also have low R² on these positions.
- **Aggregate MAE rose for K and DST compared to the pre-`0c66171` table.** The earlier table reported K=3.6 and DST=3.8 against a partially-wrong unit space (unsigned sum for K, missing PA/YA tier lookup for DST). The corrected aggregator produces the real fantasy-point error: K MAE ≈ 6.7, DST MAE ≈ 5.2. The dashboard always showed correct values; only the eval-table aggregate was wrong.
- Error analysis and per-target breakdown: [docs/expert_comparison.md](docs/expert_comparison.md) and the per-position breakdowns in the linked benchmark JSON.

## Repo Layout

The repository is organized with top-level `src/` (all source code), `data/`, `models/`, `notebooks/`, and `docs/` directories alongside `README.md` / `SETUP.md` / `ATTRIBUTION.md`.

```
src/                                All Python source code
  qb/ rb/ wr/ te/ k/ dst/           Per-position configs, features, targets, runners
  shared/                           Cross-position infrastructure
    neural_net.py                   MultiHeadNet, AttentionPool, GatedHead
    models.py                       RidgeMultiTarget, LightGBMMultiTarget, baselines
    training.py                     MultiTargetLoss, trainer, schedulers
    pipeline.py                     Position-pipeline orchestrator
    aggregate_targets.py            Raw-stat → fantasy-point scoring
    evaluation.py                   compute_metrics + position-aware aggregation
    backtest.py                     Walk-forward backtest harness
  batch/                            Training orchestration (Batch active, EC2 rollback)
    launch.py                       Local submitter (uploads data, polls, pulls models)
    train.py                        In-container training entrypoint
    Dockerfile.train                Heavy CUDA/PyTorch training image
  serving/                          Flask serving stack
    app.py                          Flask dashboard + /api/predictions + /api/wiki
    static/, templates/             Frontend assets
  benchmarking/benchmark.py         Multi-position Ridge/NN/Attn/LGBM comparison
  tuning/                           Optuna LGBM tuner, NN tuner, RB-gate ablation
  analysis/                         Post-hoc data + model analysis scripts
  scripts/                          Operator CLIs (artifact promote, scope positions)
  config.py                         Shared seasons, scoring, rolling windows
  data/                             loader, nflcom_loader, preprocessing, redzone_pbp, split
  features/engineer.py              Feature engineering coordinator

data/README.md                      Pointer to nflverse loaders + gitignored caches
models/README.md                    Pointer to per-position artifact dirs + S3 layout
notebooks/README.md                 Project doesn't use notebooks; analysis lives in src/analysis/
docs/                               ARCHITECTURE (ADR-001), design docs, runbooks
infra/ec2/                          Rollback training host (warm g4dn.xlarge)
infra/batch/                        Active training stack (AWS Batch + Spot)
infra/aws/                          ECS/ALB serving stack
tests/                              Per-position + shared test trees
Dockerfile                          Slim image for ECS serving (root for build context)
conftest.py                         pytest project-root sys.path bootstrap
```

Tests live under the top-level `tests/` tree, mirroring the `src/` layout (`tests/qb/` for `src/qb/`, etc., plus `tests/shared/`, `tests/batch/`, `tests/scripts/`, and root-level `tests/test_*.py` for cross-cutting integration + e2e).

## Deeper Reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — ADR-001, consolidated decision log
- [docs/method_contracts.md](docs/method_contracts.md) — function signatures + data-layer contracts
- [docs/batch_design.md](docs/batch_design.md) — active training infrastructure (Batch + Spot fan-out, default since 2026-05-20)
- [docs/ec2_design.md](docs/ec2_design.md) — rollback training path (reactivated via `BATCH_ACTIVE=false`)
- [docs/expert_comparison.md](docs/expert_comparison.md) — error analysis
- [docs/archive/](docs/archive/) — historical design docs folded into ADR-001 (LSTM proposal, XGBoost ensemble, weather/Vegas features)
- [infra/ec2/README.md](infra/ec2/README.md), [infra/aws/README.md](infra/aws/README.md) — operator runbooks
- [TODO.md](TODO.md) — issue log, open items, and a "Fixed" archive that doubles as a lessons-learned catalog
- [ATTRIBUTION.md](ATTRIBUTION.md) — data, libraries, and AI tool usage

## Individual Contributions

This project was built individually (no partner).

## Full-Stack Engineering

Beyond the ML core, the project ships a production deploy at [alexfree.me](https://alexfree.me), GPU training on AWS, and CI/CD that gates every push. Operator runbooks are linked above in **Deeper Reading** — this section summarizes the architecture and the meaningful enhancements that landed along the way.

### AWS Infrastructure

```
GitHub Actions
   │ push to main
   ├──▶ batch-image.yml ──▶ ECR (training image, ECR pull-through + SOCI v2)
   │                              │
   │                              ▼ workflow_run
   │        ┌──────────────────────────────────────────────┐
   │        │ BATCH_ACTIVE=true (default):                 │
   │        │   train-batch.yml → 6× g4dn.xlarge Spot      │
   │        │     (one position per host, parallel)        │
   │        │     src.batch.launch submits, polls, downloads│
   │        │   ~15 min wall-clock                         │
   │        │ ─────────────────────────────────────────────│
   │        │ BATCH_ACTIVE=false (rollback):               │
   │        │   train-ec2.yml → warm g4dn.xlarge OD via SSM│
   │        │     (six positions sequential on one T4)     │
   │        │   ~120 min wall-clock                        │
   │        │ ─────────────────────────────────────────────│
   │        │ Either path: per-position change detection   │
   │        └──────────────────────────────────────────────┘
   │                              │ manifest + tar.gz
   │                              ▼
   │        ┌────────────────────────────────────────┐
   │        │ S3: ff-predictor-training              │
   │        │   manifest v2 (stable / current /      │
   │        │   previous + 5-version history)        │
   │        │   smoke-test gate before promotion     │
   │        └────────────────────────────────────────┘
   │                              │ s3:GetObject (task role)
   ▼                              ▼
deploy.yml ──▶ ECR ──▶ ECS Fargate (arm64) ──▶ ALB + ACM HTTPS ──▶ alexfree.me
```

- **Training** — two interchangeable GPU paths, selected by the `BATCH_ACTIVE` repo variable. Default since 2026-05-20: `BATCH_ACTIVE=true` fires [.github/workflows/train-batch.yml](.github/workflows/train-batch.yml), which fans out across six g4dn.xlarge Spot instances via AWS Batch (one position per host, parallel; ~15 min wall-clock, ~$0.40/run). Rollback path: `BATCH_ACTIVE=false` fires [.github/workflows/train-ec2.yml](.github/workflows/train-ec2.yml) and drives a warm OD g4dn.xlarge sequentially via SSM Run Command (~120 min wall-clock, auto-shuts down on idle). Both paths use the same `detect` job to retrain only positions whose code changed, and both reuse [src/batch/Dockerfile.train](src/batch/Dockerfile.train) as the training container. See D7 + D13 in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the trade-off.
- **Artifact safety** — S3 manifest schema v2 tracks `stable` / `current` / `previous` plus a 5-version `history`. New artifacts must clear a smoke-test gate before being promoted to `stable`. [src/scripts/promote.py](src/scripts/promote.py) supports manual rollback to any history entry; bucket versioning is defense-in-depth.
- **Serving** — ECS Fargate (arm64, 1 vCPU / 2 GB) sits behind an ALB with ACM-terminated HTTPS. The slim Flask image fetches models from S3 at boot rather than baking them in — keeps the image roughly 3× smaller and lets prod track new artifacts without a full redeploy.
- **IAM** — the serving task role is scoped to `s3:GetObject` on `ff-predictor-training/models/*` only.
- **Rollback symmetry** — flipping `BATCH_ACTIVE` is a one-command operation either direction. Both training paths stay provisioned indefinitely; `workflow_dispatch` on either workflow bypasses the gate as a break-glass.

**Notable enhancements**
- Always-stable serving + smoke-test gate + S3 bucket versioning (PR #130, `c7fa2d7`)
- Versioned history + manifest-driven rollback (PR #104, `1b20e9e`)
- Operational rollback CLI [src/scripts/promote.py](src/scripts/promote.py) (PR #122, `e8bf2a7`)
- ECS force-rollover after training so promoted artifacts get picked up by a fresh task (PR #179, `8c42e88`)
- PPR / Half-PPR / Standard scoring switch wired end-to-end through the dashboard (PR #153, `a533990`)
- Wiki tab renders repo markdown docs in-app (PR #138, `ce4543e`)
- Benchmark History tab — per-PR rows fetched from S3 at boot, auto-updates after every training run without a redeploy (PR #201, `056423b`)
- Slim arm64 serving image with runtime S3 model fetch (PR #83, `3243d72`)
- Parallel Spot fan-out across six g4dn.xlarge Spot instances collapses full-retrain wall-clock from sum(per-position) to max(per-position), ~15 min vs ~120 min (D13); the warm-EC2 rollback path (D7) eliminates 3–5 min Batch scale-up on the days a single-position iteration matters more than parallelism

### GitHub CI/CD

```
push to main ──▶ tests.yml   (7-shard pytest matrix · per-flag Codecov · 80% target)
             │
             │                                            BATCH_ACTIVE=true
             ├──▶ batch-image.yml ──┬─▶ train-batch.yml ──▶ 6× Spot g4dn (parallel)
             │                      │                       (~15 min full retrain)
             │                      │
             │                      │                            BATCH_ACTIVE=false
             │                      └─▶ train-ec2.yml ────────▶ warm OD g4dn (sequential)
             │                                                  (~120 min full retrain)
             │
             └──▶ deploy.yml ─────────▶ ECR ──▶ ECS Fargate
```

- Five production workflows ([tests.yml](.github/workflows/tests.yml), [batch-image.yml](.github/workflows/batch-image.yml), [train-batch.yml](.github/workflows/train-batch.yml), [train-ec2.yml](.github/workflows/train-ec2.yml), [deploy.yml](.github/workflows/deploy.yml)) plus three diagnostic ones ([ablate-rb-gate.yml](.github/workflows/ablate-rb-gate.yml), [retune-lgbm.yml](.github/workflows/retune-lgbm.yml), [retune-nn-batch.yml](.github/workflows/retune-nn-batch.yml)) for one-click experiments on the GPU host.
- **`tests.yml`** — 7-shard pytest matrix (QB / RB / WR / TE / K / DST / shared) with per-shard Codecov flags and an 80% per-component target enforced via [codecov.yml](codecov.yml). Within-shard parallelism via `pytest-xdist`.
- **`batch-image.yml` → `train-batch.yml` *or* `train-ec2.yml`** — the image build is gated by path filters and pushes to ECR with both an ECR pull-through-cache-routed base layer and a SOCI v2 index for fast Spot cold-start. The `detect` job (shared by both training workflows) diffs `HEAD^..HEAD` and retrains *only* the positions whose code changed; cross-cutting changes (`src/shared/`, `src/batch/`, shared `src/` modules, `requirements.txt`) retrain all six. `BATCH_ACTIVE` selects which training workflow fires; `workflow_dispatch` on either bypasses the gate.
- **`deploy.yml`** — native `arm64` runner (no QEMU emulation), BuildKit cache persisted across runs, path-filtered to the serving surface so docs-only or test-only changes don't trigger a deploy.
- All Python installs use `uv` for ~10× faster cold starts than pip.

**Notable enhancements**
- `uv` migration across CI (PR #51, `3c897d8`)
- Per-component Codecov flags with 80% per-flag target (PR #78, `84b45b9`)
- Position-level change detection — skip retrains for tests-only or docs-only PRs (PR #84, `b087189`)
- Per-shard test scoping via `detect` job — only run shards whose code changed (PR #185, `8150689`)
- Docs-only filter folded into `detect`; standalone `tests-skip.yml` retired (PR #186, `5737c67`)
- Training-step perf composition: Feather parquet cache + async DataLoader + cuDNN benchmark + phase timings (PR #183, `48ef419`)
- Static-pad attention sequences (−30% on attention training) + disk-backed feature-engineering cache (−86× on `prepare_data` hits) (PR #200, `349aa4a`)
- `torch.compile` measured and rejected on T4 (+32% wall time on sm_75 + variable-length sequences) — wrapper kept on EC2 image but short-circuited (PR #189, `3167b56`)
- Native `arm64` deploy runner + BuildKit cache (PR #83, `3243d72`)
- Pytest sharding (PR #48, `40f49b2`) + xdist within shards (PR #57, `2f42867`)
- Diagnostic workflows: RB TD-gate ablation (PR #97, `3e49419`) and LightGBM Optuna retune (PR #98, `b7fde11`)
