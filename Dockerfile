# syntax=docker/dockerfile:1
# Pinned to 3.12, matching Day 1/2/3 -- consistency across the projects,
# not a hard requirement of anything used here.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
# --mount=type=cache persists pip's download cache across build attempts
# in its own BuildKit cache, not the image layer -- so a killed/retried
# build on a slow connection doesn't re-download everything from zero.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 30 -r requirements.txt

COPY rag/ ./rag/
COPY ui/ ./ui/
COPY tests/ ./tests/
COPY pytest.ini .

# Downloads the embedding model's ONNX weights into FASTEMBED_CACHE_DIR at
# build time (as root, before the chown below) rather than on first real
# use -- so `docker compose run --rm ingest` doesn't stall on a Hugging
# Face download in the middle of what's supposed to be a local/offline step.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "from rag.embed import _get_model; _get_model()"

RUN mkdir -p data \
    && useradd --create-home --uid 1000 raguser \
    && chown -R raguser:raguser /app
USER raguser

# Overridden per-service in docker-compose.yml.
CMD ["python", "-m", "rag.cli"]
