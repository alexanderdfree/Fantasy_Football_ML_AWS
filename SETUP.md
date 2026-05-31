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

# Dev/test tooling (pytest, ruff) — only needed for running tests or lint
pip install -r requirements-dev.txt
```

## Windows 11 + NVIDIA GPU install (e.g. RTX 5080)

The block above installs the **CPU** PyTorch wheel. To train/tune locally on an NVIDIA GPU —
Windows 11 with an RTX 5080 (or any Blackwell `sm_120` card), or Linux + NVIDIA — use the
**CUDA 12.8** build instead. It's the same `torch==2.11.0`, just the `cu128` wheel (the `cu126`
wheel the AWS Tesla-T4 path uses tops out at `sm_90`, so Blackwell needs `cu128` specifically).

**Prerequisites**

- Python **3.12** — matches the project; the `cu128` cp312 wheel exists. Install from python.org and tick *"Add python.exe to PATH"*.
- A recent NVIDIA driver (R570+, the GeForce 50-series launch driver or newer). You do **not** need a separate CUDA Toolkit — the pip wheel bundles its own CUDA 12.8 runtime via `nvidia-*` packages.
- `git`.

**Create and activate a venv** (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked by execution policy, run once in this shell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

(`cmd` users: `.venv\Scripts\activate.bat`.)

**Install** the GPU dependency set ([requirements-gpu.txt](requirements-gpu.txt) — the GPU analog of `requirements-dev.txt`, with the `cu128` torch build):

```powershell
pip install -r requirements-gpu.txt
```

To match CI's `uv` path instead, set `UV_INDEX_STRATEGY` first — `$env:UV_INDEX_STRATEGY="unsafe-best-match"` in PowerShell (or `set UV_INDEX_STRATEGY=unsafe-best-match` in `cmd`) — then `uv pip install -r requirements-gpu.txt`. To swap an existing CPU env in place without a full reinstall: `pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.

**Verify the GPU is visible:**

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → 2.11.0+cu128 12.8 True NVIDIA GeForce RTX 5080
```

**First-time data pull** — the heredoc in the next section is bash-only; on Windows use this cross-shell one-liner instead:

```powershell
python -c "from src.data.loader import load_raw_data; from src.data.preprocessing import preprocess; from src.features.engineer import build_features; from src.data.split import temporal_split; temporal_split(build_features(preprocess(load_raw_data())))"
```

**Run training / tuning** — identical commands to macOS/Linux, run from the repo root.
**Set `$env:OPENBLAS_NUM_THREADS='1'` first — REQUIRED on Windows, not a perf knob:** without it,
local pipeline / benchmark / tuning runs crash partway through a position with a `0xC0000005` access
violation (process exit `-1073741819`, no Python traceback), because concurrent threaded LAPACK `eigh`
calls (the Ridge-PCA alpha CV in [src/shared/pipeline.py](src/shared/pipeline.py)) segfault Windows'
bundled OpenBLAS. Linux OpenBLAS tolerates it, which is why AWS Batch / CI never hit it (the Batch
job-def and `conftest.py` pin the same thread caps — this is the local-dev equivalent). Capping it to
one thread fixes it and only throttles numpy's BLAS — LightGBM uses a separate OpenMP runtime
(`LGBM_N_JOBS`, below), so tree-building still uses every core. Set it once per PowerShell session; it
covers every command below, including the tuners.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"   # REQUIRED on Windows (cmd: set OPENBLAS_NUM_THREADS=1)

python -m src.qb.run_pipeline                       # one position, full pipeline
python -m src.benchmarking.benchmark RB --no-sync   # benchmark one position
python -m src.tuning.tune_nn RB --n-trials 30       # Optuna NN tuning
python -m src.tuning.tune_lgbm RB                   # Optuna LightGBM tuning
```

Run from the repo root — the tuners look for a relative `data/raw/` (same as on macOS, not a
Windows-specific quirk). The RTX 5080 is picked up automatically by the `torch.cuda.is_available()`
device check ([src/shared/pipeline.py](src/shared/pipeline.py)); the attention NN trains in FP16,
which is native (and full-throughput) on Blackwell. The remaining sections below (*Run benchmarks*,
*Run tests*, *Lint*) work as written with the same `python -m …` / `pytest` / `ruff` commands. To
get full use of the 9950X3D's cores, see the next subsection.

### Use all 16 cores (performance)

Most parallelism is already wired: Ridge/ElasticNet CV fans its alpha grid across cores via `joblib`
(one thread per alpha — each doing a single-threaded BLAS solve under the REQUIRED
`OPENBLAS_NUM_THREADS=1` above, so the cores stay busy via the alpha fan-out, not multi-threaded
BLAS), and a single position overlaps its CPU classical-ML models with the GPU attention NN
([src/shared/pipeline.py](src/shared/pipeline.py)). Two things to know to fully use the 9950X3D:

**NN training is GPU-bound.** During the attention-NN phase the CPU will look mostly idle — that's
expected; the 5080 is doing the work. The CPU-heavy parts are the classical-ML models and tuning.

**LightGBM is single-threaded by default** — the one lever that matters on a 16-core box.
`_lgbm_n_jobs()` ([src/shared/models.py](src/shared/models.py)) reads the `LGBM_N_JOBS` env var,
defaulting to `1` (a guard for macOS's nested-OpenMP segfault and small Linux-CI boxes — neither
applies to Windows). Set it to your **physical** core count for multithreaded tree-building:

```powershell
$env:LGBM_N_JOBS = "16"     # cmd: set LGBM_N_JOBS=16
```

Use `16` (physical cores), not `32` (logical) — LightGBM tree-building usually regresses under
hyperthreading, and `-1` means "all 32 logical". Quick A/B on your box: time
`python -m src.rb.run_pipeline` with `LGBM_N_JOBS=1` vs `16`.

**Parallel tuning — match the knob to the bottleneck:**

```powershell
# LightGBM tuning is CPU-bound: parallelize TRIALS, keep each trial single-threaded.
$env:LGBM_N_JOBS = "1"
python -m src.tuning.tune_lgbm RB --n-jobs 16

# NN tuning is GPU-bound: trials share the one GPU. --n-jobs default is 2 (fits 16 GB);
# try 3 if VRAM allows — diminishing past that.
python -m src.tuning.tune_nn RB --n-jobs 2
```

Don't stack the two — many parallel LightGBM-tuning trials *and* `LGBM_N_JOBS=16` is 16×16 thread
oversubscription (the `--n-jobs` help flags this). And unset `LGBM_N_JOBS` (or use a separate shell)
when running the full `pytest` suite — with `-n auto` xdist workers a high per-process thread count
oversubscribes the runner.

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

- **`BATCH_ACTIVE=true`** (default) → [.github/workflows/train-batch.yml](.github/workflows/train-batch.yml): fans out six g6.xlarge Spot instances via AWS Batch (one position per host, parallel; ~15 min wall-clock, ~$0.65/run estimate on g6 — migrated from g4dn 2026-05-31). One-time provisioning via [infra/batch/setup.sh](infra/batch/setup.sh); operator runbook in [infra/batch/README.md](infra/batch/README.md); full design in [docs/batch_design.md](docs/batch_design.md).
- **`BATCH_ACTIVE=false`** (rollback) → [.github/workflows/train-ec2.yml](.github/workflows/train-ec2.yml): starts the warm g4dn.xlarge OD instance, runs the six position pipelines sequentially via SSM (~120 min wall-clock; one T4 can't fit concurrent NN runs). See [infra/ec2/README.md](infra/ec2/README.md) and [docs/ec2_design.md](docs/ec2_design.md).

Flip with `gh variable set BATCH_ACTIVE --body "true"` or `--body "false"`; the new value takes effect on the next push. `workflow_dispatch` on either workflow bypasses the gate for smoke-testing the inactive path. See D7 + D13 in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the trade-off.
