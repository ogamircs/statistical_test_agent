# syntax=docker/dockerfile:1.7
# Multi-stage Dockerfile for the Statistical Test Agent (Chainlit + LangGraph).
# Build:   docker build -t statistical-test-agent .
# Run:     docker run -p 8000:8000 -e OPENAI_API_KEY=... statistical-test-agent
#
# Spark extras are intentionally excluded from the default image: pulling in a
# Java runtime for PySpark would more than double the image size and is only
# required for the optional large-file backend. See docs/deployment.md.

############################
# Stage 1: builder
############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

# Install uv (single static binary) without pulling pip's full dependency tree.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# First, sync only dependency metadata so this layer caches across source edits.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev --frozen --no-install-project

# Copy the project source and finish the install (registers the local package).
COPY src ./src
COPY app.py chainlit.md ./
COPY .chainlit ./.chainlit
COPY public ./public
COPY scripts ./scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev --frozen

############################
# Stage 2: runtime
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv

# Create a non-root user for the runtime.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home --home-dir /home/appuser appuser

WORKDIR /app

# Bring over the resolved virtualenv and project source from the builder.
COPY --from=builder --chown=appuser:appuser /app /app

# Ensure mountable directories exist with the right ownership for bind mounts.
RUN mkdir -p /app/data /app/output && chown -R appuser:appuser /app/data /app/output

USER appuser

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
