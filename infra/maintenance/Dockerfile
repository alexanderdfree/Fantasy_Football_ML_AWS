# syntax=docker/dockerfile:1.7
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /uvx /usr/local/bin/
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy UV_INDEX_STRATEGY=unsafe-best-match
COPY requirements.txt requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -r requirements-dev.txt
COPY src/ src/
# This file belongs to the producer fingerprint and must exist in both runtimes.
COPY .github/workflows/refresh-splits.yml .github/workflows/refresh-splits.yml
ARG SOURCE_SHA
RUN python -m src.maintenance.image_metadata --source-sha "$SOURCE_SHA"
ENV FF_DEVICE=cpu NFLREADPY_CACHE=off
ENTRYPOINT ["python", "-m", "src.maintenance.worker"]
