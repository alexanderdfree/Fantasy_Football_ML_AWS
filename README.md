# Fantasy Football Weekly Points Predictor

A per-position machine learning system that predicts weekly NFL fantasy points for QBs, RBs, WRs, TEs, Kickers, and D/STs — comparing a Ridge baseline, a custom PyTorch multi-head neural network (with attention for skill positions), and LightGBM across the 2012–2025 seasons. Served as a Flask dashboard at [alexfree.me](https://alexfree.me).

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

Sharing a backbone across per-target heads is what makes the neural net competitive at small sample sizes; reporting MAE in raw units (yards / TDs / receptions) keeps per-target accuracy interpretable and decouples model error from scoring-format choice. Design rationale is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Architecture at a glance

```
nflverse API ─┐
(2012–2025)   │
              ▼
  ┌──────────────────────┐    ┌──────────────────────────────────────────┐
  │ src/data + src/features │▶ │ Three model families per position        │
  │ (rolling, EWMA, share,   │  │   Ridge  │  MultiHeadNet (PyTorch)       │
  │  matchup, weather/odds)  │  │          │  + Attention (skill positions)│
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
# Run the comparison benchmark (all positions) and append to benchmark_history.json
python benchmark.py

# Benchmark one position
python benchmark.py RB

# Serve the dashboard locally → http://localhost:5000
python app.py

# Run the test suite
pytest
pytest -m unit        # fast tests only
```

Full training on GPU runs on EC2 via CI; see [docs/ec2_design.md](docs/ec2_design.md) for the pipeline and [infra/ec2/README.md](infra/ec2/README.md) for operator notes.

## Evaluation

Holdout: 2025 season. Metric definitions: MAE (mean absolute error in fantasy points), R² (coefficient of determination), top-12 hit rate (agreement with the actual weekly top 12 at the position, PPR scoring). Numbers from [benchmark_results.json](benchmark_results.json).

> Benchmarks pending retrain on new raw-stat targets — the values below were produced against the prior `*_floor` / `td_points` decomposition and will be refreshed once the migration training run completes.

| Position | Ridge MAE | NN MAE | Attn NN MAE | LGBM MAE | Best | R² (best) | Top-12 (best) |
|---|---|---|---|---|---|---|---|
| QB  | 6.471 | 6.303 | 6.360 | **6.209** | LightGBM     | 0.293  | 0.526 |
| RB  | 4.358 | **4.169** | 4.245 | 4.191 | MultiHeadNet | 0.418  | 0.528 |
| WR  | 4.439 | 4.218 | **4.185** | 4.233 | Attention NN | 0.359  | 0.377 |
| TE  | 3.592 | 3.513 | **3.486** | 3.524 | Attention NN | 0.292  | 0.483 |
| K   | **3.605** | 3.707 | — | — | Ridge        | 0.067  | 0.495 |
| DST | **3.826** | 3.875 | — | — | Ridge        | 0.055  | 0.472 |

**Takeaways:**
- The neural nets win on skill positions (RB, WR, TE) where interaction and sequence structure matters. Attention pulls slightly ahead on WR/TE.
- Ridge wins K and DST — the signal at those positions is too weak for a higher-capacity model to pay off.
- LightGBM is the best model at QB, narrowly. It's within a rounding error of NN everywhere else.
- Error analysis and per-target breakdown: [docs/expert_comparison.md](docs/expert_comparison.md) and the per-position NN/Ridge/LightGBM breakdowns in [benchmark_results.json](benchmark_results.json).

## Video Links

- Demo video: _TODO — add link before submission_
- Technical walkthrough: _TODO — add link before submission_

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

## Solo Project

This project was completed individually (no partner) — claiming the 10-pt solo project rubric item.
