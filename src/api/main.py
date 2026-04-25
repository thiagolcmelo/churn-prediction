"""FastAPI application for churn prediction inference."""

import json
import pickle
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pandas as pd
import torch
from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from src.api.metrics import (
    PREDICTION_CLASS,
    PREDICTION_VALUE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    metrics_endpoint,
)
from src.api.schemas import CustomerInput, PredictionOutput
from src.models.mlp import ChurnMLP
from src.utils import get_logger

logger = get_logger(__name__)

THRESHOLD = 0.5

# It reflects the example in CustomerInput.model_config
_DUMMY_ROW: dict[str, object] = {
    "tenure": 1,
    "MonthlyCharges": 50.0,
    "TotalCharges": 50.0,
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    with open("models/preprocessor.pkl", "rb") as f:
        app.state.preprocessor = pickle.load(f)

    # The preprocessor can go through retraining and it can alter the final number
    # of columns. The raw data is expected and enforced through schema to stay stable.
    input_dim: int = app.state.preprocessor.transform(pd.DataFrame([_DUMMY_ROW])).shape[
        1
    ]

    # It should be fetched from somewhere else . For the sake of convenience it
    # will be stored directly in this file system and an exception will be made
    # ro add it to versioning, to facilite download of usage.
    model = ChurnMLP(input_dim=input_dim)
    model.load_state_dict(
        torch.load("models/mlp_churn.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    app.state.model = model

    with open("models/mlp_churn_meta.json") as f:
        app.state.model_meta = json.load(f)

    logger.info(f"Model loaded: input_dim={input_dim}, threshold={THRESHOLD}")
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def track_metrics(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Middleware that records request count and latency for every request."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    route = request.scope.get("route")
    endpoint_label = route.path if route else "unmatched_route"

    # Skip metrics endpoint itself to avoid recursion
    if endpoint_label != "/metrics":
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint_label,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint_label,
        ).observe(duration)
        logger.info(
            f"{request.method} {endpoint_label} — {duration * 1000:.1f}ms — {response.status_code}"
        )

    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint — verify the API is running."""
    meta = app.state.model_meta
    return {"status": "healthy", "model": meta["name"], "run_id": meta["run_id"]}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint — scraped by Prometheus every 5 seconds."""
    return metrics_endpoint()


@app.post("/predict", response_model=PredictionOutput)
def predict(request: Request, customer: CustomerInput) -> PredictionOutput:
    """Predict churn probability for a single customer."""
    # 1. Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([customer.model_dump()])

    # 2. Run through preprocessing pipeline
    X_processed = request.app.state.preprocessor.transform(input_df)

    # 3. Run through MLP model
    X_tensor = torch.FloatTensor(X_processed)
    with torch.no_grad():
        probability = torch.sigmoid(request.app.state.model(X_tensor)).item()
    churn = probability >= THRESHOLD

    # 4. Record metrics for monitoring
    PREDICTION_VALUE.observe(probability)
    PREDICTION_CLASS.labels(prediction="churn" if churn else "no_churn").inc()

    # 5. Return probability and binary prediction
    return PredictionOutput(
        churn_probability=round(probability, 4),
        churn_prediction=churn,
        threshold=THRESHOLD,
    )
