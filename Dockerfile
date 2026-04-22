# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# uv: pinned single-binary installer (~10x faster than pip, parallel wheels).
# Pin the same minor as batch/Dockerfile.train so wheel resolution stays
# consistent across training and serving images.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Single uv-install layer. Any change to requirements.txt invalidates exactly
# one cached layer. The /root/.cache/uv mount persists wheel downloads across
# builds (CI mirrors this via actions/cache in deploy.yml). Torch uses the
# CPU-only index; --extra-index-url keeps pypi.org as a fallback because uv's
# --index-url fully overrides the default index (pip's does not). nfl_data_py
# needs --no-deps (mord/old-numpy conflict) and runs as a separate uv call in
# the same RUN so the layer boundary matches the previous Dockerfile.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.11.0 && \
    uv pip install --system -r requirements.txt && \
    uv pip install --system --no-deps nfl_data_py==0.3.3

# Only the paths app.py actually imports. data/ and **/outputs/models/ are
# deliberately NOT copied — shared.model_sync fetches them from S3 at
# container startup, which shrinks the image and decouples deploys from
# data/model changes.
COPY app.py .
COPY shared/ shared/
COPY src/ src/
COPY QB/ QB/
COPY RB/ RB/
COPY WR/ WR/
COPY TE/ TE/
COPY K/ K/
COPY DST/ DST/
COPY templates/ templates/
COPY static/ static/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--preload", "--timeout", "120", "--access-logfile", "-", "app:app"]
