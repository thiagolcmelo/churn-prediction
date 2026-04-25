"""FastAPI application for churn prediction inference."""

import pickle
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pandas as pd
import torch
from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

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

    logger.info(f"Model loaded: input_dim={input_dim}, threshold={THRESHOLD}")
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


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


@app.post("/predict", response_model=PredictionOutput)
def predict(request: Request, customer: CustomerInput) -> PredictionOutput:
    """Predict churn probability for a single customer."""
    input_df = pd.DataFrame([customer.model_dump()])
    X_processed = request.app.state.preprocessor.transform(input_df)
    X_tensor = torch.FloatTensor(X_processed)
    with torch.no_grad():
        probability = torch.sigmoid(request.app.state.model(X_tensor)).item()
    return PredictionOutput(
        churn_probability=round(probability, 4),
        churn_prediction=probability >= THRESHOLD,
        threshold=THRESHOLD,
    )
