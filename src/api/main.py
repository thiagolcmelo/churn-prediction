"""FastAPI application for churn prediction inference."""

import time

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from src.utils import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.middleware("http")
async def log_latency(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Log request latency for every endpoint."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} — {duration_ms:.1f}ms")
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint — verify the API is running."""
    return {"status": "healthy", "model": "mlp_model"}
