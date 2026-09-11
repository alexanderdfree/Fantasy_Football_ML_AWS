# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# uv: pinned single-binary installer (~10x faster than pip, parallel wheels).
# Pin the same minor as batch/Dockerfile.train so wheel resolution stays
# consistent across training and serving images.
COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /uvx /usr/local/bin/

WORKDIR /app

# uv defaults to hardlinking wheels from its cache into site-packages. The
# BuildKit cache mount below lives on a different overlay than the image
# root, so hardlinking always fails and falls back to copy — explicit copy
# mode skips the failed probe and silences the warning in CI.
ENV UV_LINK_MODE=copy
ENV FF_ALLOW_RUNTIME_INFERENCE=0

# Serving reads verified projections and metadata. Model execution and provider
# ingestion dependencies belong only to the offline prediction/training image.
COPY requirements-serving.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-serving.txt

# All Python source, Flask templates/static, and per-position assets live
# under src/. data/ and src/**/outputs/models/ are deliberately NOT copied —
# src.shared.model_sync fetches them from S3 at container startup, which
# shrinks the image and decouples deploys from data/model changes.
COPY src/ src/

# Benchmark history JSONs power the History tab. We bundle the git-tracked
# floor in the image so the tab is never empty, even if S3 sync is no-op
# (bucket env var unset, network blip, etc.). sync_benchmark_history_from_s3
# at container boot layers on any newer runs uploaded since this build.
COPY benchmark_history/ benchmark_history/

# Markdown source files served by the in-app Wiki tab (/api/wiki/<slug>).
# Only the .md files in src/serving/app.py:WIKI_DOCS — non-MD files in
# infra/*/ (shell scripts, JSON policies) are intentionally excluded to keep
# the slim image lean.
COPY README.md SETUP.md TODO.md ./
COPY docs/ docs/

# Gunicorn config — post_fork pre-warm hook (see file for the why).
COPY gunicorn.conf.py ./
COPY infra/ec2/README.md infra/ec2/
COPY infra/aws/README.md infra/aws/

# Verify the installed image, including successful artifact-backed API requests.
COPY scripts/check-serving-runtime.py /tmp/check-serving-runtime.py
RUN python /tmp/check-serving-runtime.py --require-absent && rm /tmp/check-serving-runtime.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "--bind", "0.0.0.0:8000", "--workers", "2", "--preload", "--timeout", "120", "--access-logfile", "-", "src.serving.app:app"]
