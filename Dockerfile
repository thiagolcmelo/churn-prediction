# Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer — only rebuilds when pyproject.toml changes)
COPY pyproject.toml .
# This is costly because of Torch
# RUN pip install --no-cache-dir .
RUN --mount=type=cache,target=/root/.cache/pip pip install .

# Copy application code
COPY src/ src/
COPY models/ models/

EXPOSE 8000

# Health check — Docker can auto-restart unhealthy containers
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]