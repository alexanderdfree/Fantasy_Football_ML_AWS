# Fantasy Football Weekly Points Predictor

A per-position machine learning system that predicts weekly NFL fantasy points for QBs, RBs, WRs, TEs, Kickers, and D/STs — comparing a Ridge baseline, a custom PyTorch multi-head neural network (with an attention variant at every position), and LightGBM across the 2012–2025 seasons. Served as a Flask dashboard at [alexfree.me](https://alexfree.me).

CS 372 (Duke, Spring 2026) final project. Solo.

## What it Does

The system ingests weekly NFL data from [nflverse](https://github.com/nflverse) (player stats, rosters, schedules, snap counts), engineers rolling/EWMA/share/matchup features plus Vegas odds and weather joins, and trains one model per position family per architecture. Each position is evaluated against actual 2025 fantasy output in three scoring formats (Standard, Half-PPR, Full PPR). A Flask dashboard lets users look up any player and compare Ridge / neural net / LightGBM projections side-by-side with the real result.

## Research Question

> Can a multi-head neural network with engineered temporal features and target decomposition meaningfully outperform Ridge regression at predicting weekly fantasy output, and what features matter most?

The answer differs by position — see the Evaluation section below. Each position's model predicts a small set of raw NFL stats rather than pre-scored fantasy-point buckets, and a deterministic aggregator ([shared/aggregate_targets.py](shared/aggregate_targets.py)) converts those raw stats to fantasy points under any scoring format. The raw targets per position:

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
                             ┌──────────────────────────────┐
                             │ EC2 g4dn warm instance (train)│◀── GitHub Actions
                             │   batch/train.py, 6 parallel │     push to main
                             │   positions. AWS Batch       │
                             │   remains the standby path.  │
                             └──────────────────────────────┘
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
python benchmark.py

# Benchmark one position
python benchmark.py RB

# Serve the dashboard locally → http://localhost:5000
python app.py

# Run the test suite
pytest
pytest -m unit        # fast tests only
```

Coverage is tracked on [Codecov](https://app.codecov.io/gh/alexanderdfree/Fantasy_Football_ML_AWS) with an **80% target per position and shared component** (see [codecov.yml](codecov.yml)). One-off diagnostic CLIs (`QB/diagnose_qb_outliers.py`, `RB/analyze_rb_errors.py`) are excluded from the denominator — everything else gets pulled in.

Full training on GPU runs on EC2 via CI; see [docs/ec2_design.md](docs/ec2_design.md) for the pipeline and [infra/ec2/README.md](infra/ec2/README.md) for operator notes.

## Video Links

- Demo video — non-technical (3–5 min): [DemoRecording.mov](DemoRecording.mov)
- Technical walkthrough (hosted on YouTube — file too large for the repo): https://youtu.be/eyuTnk3qLk8

## Evaluation

Holdout: 2025 season. Metric definitions: MAE (mean absolute error in fantasy points), R² (coefficient of determination), top-12 hit rate (agreement with the actual weekly top 12 at the position, PPR scoring). Numbers from [benchmark_results.json](benchmark_results.json).

> The numbers below come from a benchmark snapshot taken before the K/DST attention-NN additions (commits `801b61a`, `cc0c627`) and the DST raw-stat migration. Re-run `python benchmark.py` to refresh against the current code; the K/DST attention cells will populate at that point.

| Position | Ridge MAE | NN MAE | Attn NN MAE | LGBM MAE | Best | R² (best) | Top-12 (best) |
|---|---|---|---|---|---|---|---|
| QB  | 6.471 | 6.303 | 6.360 | **6.209** | LightGBM     | 0.293  | 0.526 |
| RB  | 4.358 | **4.169** | 4.245 | 4.191 | MultiHeadNet | 0.418  | 0.528 |
| WR  | 4.439 | 4.218 | **4.185** | 4.233 | Attention NN | 0.359  | 0.377 |
| TE  | 3.592 | 3.513 | **3.486** | 3.524 | Attention NN | 0.292  | 0.483 |
| K   | **3.605** | 3.707 | —ᵃ | — | Ridge        | 0.067  | 0.495 |
| DST | **3.826** | 3.875 | —ᵃ | — | Ridge        | 0.055  | 0.472 |

ᵃ K/DST attention-NN cells are pending the next benchmark refresh (the snapshot in [benchmark_results.json](benchmark_results.json) pre-dates commits `801b61a` and `cc0c627`). The Ridge / NN columns above are still meaningful baselines for those positions.

**Takeaways:**
- The neural nets win on skill positions (RB, WR, TE) where interaction and sequence structure matters. Attention pulls slightly ahead on WR/TE.
- Ridge wins K and DST — the signal at those positions is too weak for a higher-capacity model to pay off.
- LightGBM is the best model at QB, narrowly. It's within a rounding error of NN everywhere else.
- Error analysis and per-target breakdown: [docs/expert_comparison.md](docs/expert_comparison.md) and the per-position NN/Ridge/LightGBM breakdowns in [benchmark_results.json](benchmark_results.json).

## Repo Layout

```
app.py                              Flask dashboard + /api/predictions
benchmark.py                        Multi-position Ridge/NN/Attn/LGBM comparison
Dockerfile                          Slim image for ECS serving
batch/                              Training orchestration (EC2 + Batch standby)
  launch.py                         Local submitter (uploads data, polls, pulls models)
  train.py                          In-container training entrypoint
  Dockerfile.train                  Heavy CUDA/PyTorch training image
QB/ RB/ WR/ TE/ K/ DST/             Per-position configs, features, targets, runners, tests
shared/                             Cross-position infrastructure
  neural_net.py                     MultiHeadNet, AttentionPool, GatedTDHead
  models.py                         RidgeMultiTarget, LightGBMMultiTarget
  training.py                       MultiTargetLoss, trainer, schedulers
  pipeline.py                       Position-pipeline orchestrator
  registry.py                       Position runner dispatch
  weather_features.py               Vegas odds + weather joins
src/                                Data + general features
  config.py                         Seasons, scoring, rolling windows
  data/                             loader, split, preprocessing
  features/engineer.py              Feature engineering
  models/                           SeasonAverageBaseline, LastWeekBaseline, RidgeModel (imported by shared/pipeline)
  training/trainer.py               Single-head Trainer (pre-MultiHeadNet; unused by current pipelines)
  evaluation/                       metrics, backtest
infra/ec2/                          Active training host (warm g4dn.xlarge)
infra/aws/                          ECS/ALB serving stack
docs/                               ARCHITECTURE (ADR-001), design docs, runbooks
instructions/                       CS 372 handout + rubric-mapping design doc
tests/ + QB/tests/ + ...            Unit, integration, e2e tests
```

## Deeper Reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — ADR-001, consolidated decision log
- [instructions/DESIGN_DOC.md](instructions/DESIGN_DOC.md) — rubric mapping and module specifications
- [docs/ec2_design.md](docs/ec2_design.md) — active training infrastructure
- [docs/batch_design.md](docs/batch_design.md) — standby training path (reactivated via `BATCH_ACTIVE=true`)
- [docs/expert_comparison.md](docs/expert_comparison.md) — error analysis
- [docs/design_lstm_multihead.md](docs/design_lstm_multihead.md), [docs/design_xgboost_ensemble.md](docs/design_xgboost_ensemble.md), [docs/design_weather_and_odds.md](docs/design_weather_and_odds.md) — rejected-alternative design docs
- [infra/ec2/README.md](infra/ec2/README.md), [infra/aws/README.md](infra/aws/README.md) — operator runbooks
- [TODO.md](TODO.md) — issue log, open items, and a "Fixed" archive that doubles as a lessons-learned catalog
- [ATTRIBUTION.md](ATTRIBUTION.md) — data, libraries, and AI tool usage

## Individual Contributions

This project was completed individually (no partner) — claiming the 10-pt solo project rubric item.

## Full-Stack Engineering

Beyond the ML rubric core, the project ships a production deploy at [alexfree.me](https://alexfree.me), GPU training on AWS, and CI/CD that gates every push. Operator runbooks are linked above in **Deeper Reading** — this section summarizes the architecture and the meaningful enhancements that landed along the way.

### AWS Infrastructure

```
GitHub Actions
   │ push to main
   ├──▶ batch-image.yml ──▶ ECR (training image)
   │                              │
   │                              ▼ workflow_run
   │        ┌────────────────────────────────────────┐
   │        │ EC2 g4dn.xlarge (warm T4 GPU host)     │
   │        │   batch/train.py via SSM Run Command   │
   │        │   per-position change detection        │
   │        └────────────────────────────────────────┘
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

- **Training** — a warm `g4dn.xlarge` T4 GPU host is driven via SSM Run Command from [.github/workflows/train-ec2.yml](.github/workflows/train-ec2.yml). A `detect` job diffs the merge commit and only retrains positions whose code actually changed; the instance auto-shuts down on idle to keep cost flat.
- **Artifact safety** — S3 manifest schema v2 tracks `stable` / `current` / `previous` plus a 5-version `history`. New artifacts must clear a smoke-test gate before being promoted to `stable`. [scripts/promote.py](scripts/promote.py) supports manual rollback to any history entry; bucket versioning is defense-in-depth.
- **Serving** — ECS Fargate (arm64, 1 vCPU / 2 GB) sits behind an ALB with ACM-terminated HTTPS. The slim Flask image fetches models from S3 at boot rather than baking them in — keeps the image roughly 3× smaller and lets prod track new artifacts without a full redeploy.
- **IAM** — the serving task role is scoped to `s3:GetObject` on `ff-predictor-training/models/*` only.
- **Standby path** — the AWS Batch image is still built behind a `BATCH_ACTIVE` repo variable so the GPU layer can fail over without a code change.

**Notable enhancements**
- Always-stable serving + smoke-test gate + S3 bucket versioning (PR #130, `c7fa2d7`)
- Versioned history + manifest-driven rollback (PR #104, `1b20e9e`)
- Operational rollback CLI [scripts/promote.py](scripts/promote.py) (PR #122, `e8bf2a7`)
- Slim arm64 serving image with runtime S3 model fetch (PR #83, `3243d72`)
- 24/7 warm EC2 host eliminates 3–5 min Batch scale-up → sub-15-min push-to-serve

### GitHub CI/CD

```
push to main ──▶ tests.yml   (7-shard pytest matrix · per-flag Codecov · 80% target)
             │
             ├──▶ batch-image.yml ──▶ train-ec2.yml ──▶ EC2 GPU host
             │                                          (per-position retrain)
             │
             └──▶ deploy.yml ────────▶ ECR ──▶ ECS Fargate
```

- Four production workflows ([tests.yml](.github/workflows/tests.yml), [batch-image.yml](.github/workflows/batch-image.yml), [train-ec2.yml](.github/workflows/train-ec2.yml), [deploy.yml](.github/workflows/deploy.yml)) plus two diagnostic ones ([ablate-rb-gate.yml](.github/workflows/ablate-rb-gate.yml), [retune-lgbm.yml](.github/workflows/retune-lgbm.yml)) for one-click experiments on the GPU host.
- **`tests.yml`** — 7-shard pytest matrix (QB / RB / WR / TE / K / DST / shared) with per-shard Codecov flags and an 80% per-component target enforced via [codecov.yml](codecov.yml). Within-shard parallelism via `pytest-xdist`.
- **`batch-image.yml` → `train-ec2.yml`** — the image build is gated by path filters; the `detect` job in `train-ec2.yml` diffs `HEAD^..HEAD` and retrains *only* the positions whose code changed. Cross-cutting changes (`shared/`, `src/`, `requirements.txt`) retrain all six.
- **`deploy.yml`** — native `arm64` runner (no QEMU emulation), BuildKit cache persisted across runs, path-filtered to the serving surface so docs-only or test-only changes don't trigger a deploy.
- All Python installs use `uv` for ~10× faster cold starts than pip.

**Notable enhancements**
- `uv` migration across CI (PR #51, `3c897d8`)
- Per-component Codecov flags with 80% per-flag target (PR #78, `84b45b9`)
- Position-level change detection — skip retrains for tests-only or docs-only PRs (PR #84, `b087189`)
- Native `arm64` deploy runner + BuildKit cache (PR #83, `3243d72`)
- Pytest sharding (PR #48, `40f49b2`) + xdist within shards (PR #57, `2f42867`)
- Diagnostic workflows: RB TD-gate ablation (PR #97, `3e49419`) and LightGBM Optuna retune (PR #98, `b7fde11`)
