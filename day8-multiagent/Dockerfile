# syntax=docker/dockerfile:1
# Same base and shape as day4-rag/Dockerfile -- consistency across this
# curriculum's projects, not a hard requirement of anything used here.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 30 -r requirements.txt

COPY agents/ ./agents/
COPY baseline/ ./baseline/
COPY tests/ ./tests/
COPY pytest.ini .
COPY eval_multiagent.py .
COPY data/ ./data/

# Downloads fastembed's ONNX weights at build time, same reasoning as
# day4-rag's Dockerfile: a query-time download would stall the first real
# `agents.cli` call instead of failing loudly and early at build time.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "from agents.embed import _get_model; _get_model()"

RUN mkdir -p data \
    && useradd --create-home --uid 1000 raguser \
    && chown -R raguser:raguser /app
USER raguser

# Overridden per-service in docker-compose.yml.
CMD ["python", "-m", "agents.cli"]
