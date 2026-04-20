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

`src.data.loader.load_raw_data()` caches the nflverse pulls to `data/raw/`. `src.data.split.temporal_split()` writes `train.parquet`, `val.parquet`, `test.parquet` under `data/splits/`. The app and benchmark both read from `data/splits/`, so these must exist before anything else runs.

```bash
python - <<'PY'
from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess
from src.data.split import temporal_split

df = preprocess(load_raw_data())
temporal_split(df)           # writes data/splits/{train,val,test}.parquet
PY
```

First run takes several minutes (downloads ~14 seasons of weekly stats, rosters, schedules, snap counts, injuries, depth charts). Subsequent runs use the parquet cache in `data/raw/` and are near-instant.

## Run the Flask app locally

```bash
python app.py
# → http://localhost:5000
```

The dashboard loads pre-trained model artifacts from each position's `outputs/models/` directory. If a position's models are missing, run the benchmark for that position first (see below) to populate them.

## Run benchmarks

```bash
python benchmark.py              # all positions, full comparison
python benchmark.py RB           # one position
python benchmark.py QB RB WR     # several positions
```

Each run writes a comparison row to [benchmark_history.json](benchmark_history.json) and refreshes the model artifacts under `{POS}/outputs/models/`. Headline results are summarized in the Evaluation section of [README.md](README.md).

## Run tests

```bash
pytest                 # full suite — unit, integration, e2e
pytest -m unit         # fast subset (<1 s per test)
pytest RB/tests/       # just one position's tests
```

The e2e tests read `data/splits/*.parquet`, so the first-time data pull must have been done. Individual markers are defined in [pyproject.toml](pyproject.toml).

## Lint

```bash
ruff check .
ruff format --check .
```

Config is in `[tool.ruff]` in [pyproject.toml](pyproject.toml).

## Train on EC2 (owner only)

Pushes to `main` trigger [.github/workflows/train-ec2.yml](.github/workflows/train-ec2.yml), which starts the warm g4dn instance, runs all six position pipelines in parallel, and uploads model tarballs to S3. See [infra/ec2/README.md](infra/ec2/README.md) for manual start/stop/teardown and [docs/ec2_design.md](docs/ec2_design.md) for the full design. AWS Batch remains available as a standby path — see [docs/batch_design.md](docs/batch_design.md) for how to reactivate it.
