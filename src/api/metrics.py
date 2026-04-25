# src/api/metrics.py
"""Prometheus metrics instrumentation for the Churn Prediction API."""

from fastapi import Response
from prometheus_client import Counter, Histogram, Info, generate_latest

# --- Metric definitions ---
# Each metric has a name, help text, and optional labels.
# Labels let you slice metrics by dimensions (e.g., by endpoint, status code).

REQUEST_COUNT = Counter(
    "churn_api_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "churn_api_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    # Custom buckets tuned for ML inference latency (10ms to 1s)
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)

PREDICTION_VALUE = Histogram(
    "churn_api_prediction_value",
    "Distribution of churn probability predictions",
    # Buckets from 0.0 to 1.0 in 0.1 increments — matches probability range
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PREDICTION_CLASS = Counter(
    "churn_api_prediction_class_total",
    "Count of predictions by class",
    ["prediction"],  # "churn" or "no_churn"
)

MODEL_INFO = Info(
    "churn_api_model",
    "Model metadata",
)

# Set model info once at import time
MODEL_INFO.info(
    {
        "version": "1.0.0",
        "architecture": "MLP_64_32",
        "framework": "pytorch",
    }
)


def metrics_endpoint() -> Response:
    """Expose Prometheus metrics in the expected text format."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
