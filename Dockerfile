# syntax=docker/dockerfile:1.7
# Multi-stage Dockerfile for the Statistical Test Agent (FastAPI + React + LangGraph).
# Build:   docker build -t statistical-test-agent .
# Run:     docker run -p 8000:8000 -e OPENAI_API_KEY=... statistical-test-agent

############################
# Stage 1: frontend build
############################
FROM node:22-slim AS frontend

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

############################
# Stage 2: Python dependencies
############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Runtime dependencies only: no dev extra (pytest/ruff/mypy stay out of the image).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src ./src
COPY app.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

############################
# Stage 3: runtime
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv

RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home --home-dir /home/appuser appuser

WORKDIR /app

# Only what the server needs: venv, source, entrypoint, sample data, built UI.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/app.py /app/app.py
COPY --chown=appuser:appuser data/sample_ab_data.csv data/sample_ab_data_alt.csv /app/data/
COPY --from=frontend --chown=appuser:appuser /frontend/dist /app/frontend/dist

# Mountable/writable directories: data, per-session stores, uploads.
RUN mkdir -p /app/data /app/output /app/.uploads \
    && chown -R appuser:appuser /app/data /app/output /app/.uploads

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
