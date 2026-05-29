# Setup

First-time installation and local-run instructions. Once the environment is up, see the Quick Start section of [README.md](README.md) for everyday commands.

## Prerequisites

- Python 3.12 (the training Dockerfile targets 3.12; local venvs should match)
- `git`
- ~2 GB free disk space for the cached NFL dataset
- (Optional) AWS credentials, only needed for EC2 training or deploying to ECS

## Install

```bash
git clone https://github.com/alexanderdfree/Fantasy_Football_ML_AWS.git
cd Fantasy_Football_ML_AWS

python3.12 -m venv .venv
source .venv/bin/activate

# Core deps (numpy, pandas, sklearn, flask, lightgbm, boto3, …)
pip install -r requirements.txt

# PyTorch (CPU wheel — swap in the CUDA wheel if training locally)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# nfl_data_py is installed without its transitive deps on purpose
# (it pulls in heavy/old packages that conflict with pandas 3.x).
pip install --no-deps nfl_data_py==0.3.3

# Dev/test tooling (pytest, ruff) — only needed for running tests or lint
pip install -r requirements-dev.txt
```

## First-time data pull and split

`src.data.loader.load_raw_data()` caches the nflverse pulls to `data/raw/`. `src.features.engineer.build_features()` materialises the ~150 engineered columns (rolling_*, ewma_*, trend_*, prior_season_*, opp_*, contextual, position one-hots) every position's `include_features` whitelist references — without this step every engineered column ends up constant-zero via the silent backfill that used to live in `src/shared/feature_build.py` (now raises `KeyError`). `src.data.split.temporal_split()` writes `train.parquet`, `val.parquet`, `test.parquet` under `data/splits/`. The app and benchmark both read from `data/splits/`, so these must exist before anything else runs.

```bash
python - <<'PY'
from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess
from src.features.engineer import build_features
from src.data.split import temporal_split

df = build_features(preprocess(load_raw_data()))
temporal_split(df)           # writes data/splits/{train,val,test}.parquet
PY
```

First run takes several minutes (downloads ~14 seasons of weekly stats, rosters, schedules, snap counts, injuries, depth charts). Subsequent runs use the parquet cache in `data/raw/` and are near-instant.

## Run the Flask app locally

```bash
python -m src.serving.app
# → http://localhost:5050
```

The dashboard loads pre-trained model artifacts from each position's `src/{pos}/outputs/models/` directory. If a position's models are missing, run the benchmark for that position first (see below) to populate them.

## Run benchmarks

```bash
python -m src.benchmarking.benchmark              # all positions, full comparison
python -m src.benchmarking.benchmark RB           # one position
python -m src.benchmarking.benchmark QB RB WR     # several positions
python -m src.benchmarking.benchmark RB --no-sync # skip the S3 mirror (throwaway run)
```

Each run writes a `{run_id}.json` file under [benchmark_history/](benchmark_history/) with the git SHA, timestamp, and per-position config snapshot (used by CI to track benchmark drift) and refreshes the model artifacts under `src/{pos}/outputs/models/`. Headline results are summarized in the Evaluation section of [README.md](README.md).

**Cloud sync (so local runs appear on the website's History tab).** When `FF_MODEL_S3_BUCKET` is set (with AWS credentials available — e.g. `export FF_MODEL_S3_BUCKET=ff-predictor-training`), each run also mirrors its JSON to `s3://{bucket}/{FF_MODEL_S3_PREFIX:-models}/benchmark_history/`, the same key the serving container syncs at boot — so the run shows up on the History tab at the next container restart. Without the env var the upload is skipped (you'll see a one-line notice and the local JSON is still written); pass `--no-sync` to skip the upload even when a bucket is configured.

## Run tests

```bash
pytest                       # full suite — unit, integration, e2e
pytest -m unit               # fast subset (<1 s per test)
pytest tests/rb/             # just one position's tests
```

The e2e tests read `data/splits/*.parquet`, so the first-time data pull must have been done. Individual markers are defined in [pyproject.toml](pyproject.toml).

## Lint

```bash
ruff check .
ruff format --check .
```

Config is in `[tool.ruff]` in [pyproject.toml](pyproject.toml).

## Train in the cloud (owner only)

Pushes to `main` trigger one of two training workflows, selected by the `BATCH_ACTIVE` repo variable. The default since 2026-05-20 is `true` — Spot fan-out via AWS Batch.

- **`BATCH_ACTIVE=true`** (default) → [.github/workflows/train-batch.yml](.github/workflows/train-batch.yml): fans out six g4dn.xlarge Spot instances via AWS Batch (one position per host, parallel; ~15 min wall-clock, ~$0.40/run). One-time provisioning via [infra/batch/setup.sh](infra/batch/setup.sh); operator runbook in [infra/batch/README.md](infra/batch/README.md); full design in [docs/batch_design.md](docs/batch_design.md).
- **`BATCH_ACTIVE=false`** (rollback) → [.github/workflows/train-ec2.yml](.github/workflows/train-ec2.yml): starts the warm g4dn.xlarge OD instance, runs the six position pipelines sequentially via SSM (~120 min wall-clock; one T4 can't fit concurrent NN runs). See [infra/ec2/README.md](infra/ec2/README.md) and [docs/ec2_design.md](docs/ec2_design.md).

Flip with `gh variable set BATCH_ACTIVE --body "true"` or `--body "false"`; the new value takes effect on the next push. `workflow_dispatch` on either workflow bypasses the gate for smoke-testing the inactive path. See D7 + D13 in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the trade-off.
