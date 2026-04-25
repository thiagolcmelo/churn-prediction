"""FastAPI application for churn prediction inference."""

from fastapi import FastAPI

from src.utils import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint — verify the API is running."""
    return {"status": "healthy", "model": "mlp_model"}
