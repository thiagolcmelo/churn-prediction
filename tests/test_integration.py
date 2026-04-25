"""Integration tests for the ML pipeline (preprocessor + model, no HTTP layer)."""

import pickle
from typing import Any

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.api.schemas import CustomerInput
from src.models.mlp import ChurnMLP

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


@pytest.fixture(scope="module")
def preprocessor() -> Any:
    """Load the preprocessor artefact once for the module."""
    with open("models/preprocessor.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def model(preprocessor: Any) -> ChurnMLP:
    """Load the MLP model artefact once for the module."""
    input_dim: int = preprocessor.transform(pd.DataFrame([_DUMMY_ROW])).shape[1]
    m = ChurnMLP(input_dim=input_dim)
    m.load_state_dict(
        torch.load("models/mlp_churn.pt", map_location="cpu", weights_only=True)
    )
    m.eval()
    return m


@pytest.fixture
def high_risk_customer() -> dict[str, Any]:
    """Month-to-month, fiber optic, no add-ons, electronic check, short tenure."""
    return {
        "tenure": 1,
        "MonthlyCharges": 95.0,
        "TotalCharges": 95.0,
        "SeniorCitizen": 1,
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }


@pytest.fixture
def low_risk_customer() -> dict[str, Any]:
    """Two-year contract, DSL, all protections, bank transfer, long tenure."""
    return {
        "tenure": 60,
        "MonthlyCharges": 55.0,
        "TotalCharges": 3300.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
    }


def _run_pipeline(
    preprocessor: Any, model: ChurnMLP, customer: dict[str, Any]
) -> float:
    """Run the full prediction pipeline and return churn probability."""
    input_df = pd.DataFrame([CustomerInput(**customer).model_dump()])
    X = preprocessor.transform(input_df)
    with torch.no_grad():
        return float(torch.sigmoid(model(torch.FloatTensor(X))).item())


# ---------------------------------------------------------------------------
# Pipeline contract
# ---------------------------------------------------------------------------


def test_preprocessor_output_shape_matches_model_input(
    preprocessor: Any, model: ChurnMLP
) -> None:
    """Preprocessor output columns must equal the model's first-layer input dimension."""
    X = preprocessor.transform(pd.DataFrame([_DUMMY_ROW]))
    first_layer = model.network[0]
    assert isinstance(first_layer, nn.Linear)
    assert first_layer.in_features == X.shape[1]


def test_schema_dump_is_accepted_by_preprocessor(
    preprocessor: Any, high_risk_customer: dict[str, Any]
) -> None:
    """CustomerInput.model_dump() must be directly consumable by the preprocessor."""
    input_df = pd.DataFrame([CustomerInput(**high_risk_customer).model_dump()])
    X = preprocessor.transform(input_df)
    assert X.shape[0] == 1


# ---------------------------------------------------------------------------
# Pipeline correctness
# ---------------------------------------------------------------------------


def test_pipeline_output_is_valid_probability(
    preprocessor: Any, model: ChurnMLP, high_risk_customer: dict[str, Any]
) -> None:
    """Full pipeline (no HTTP) must return a probability in [0, 1]."""
    prob = _run_pipeline(preprocessor, model, high_risk_customer)
    assert 0.0 <= prob <= 1.0


def test_pipeline_is_deterministic(
    preprocessor: Any, model: ChurnMLP, high_risk_customer: dict[str, Any]
) -> None:
    """Same input must always produce the same probability."""
    prob1 = _run_pipeline(preprocessor, model, high_risk_customer)
    prob2 = _run_pipeline(preprocessor, model, high_risk_customer)
    assert prob1 == pytest.approx(prob2)


def test_high_risk_scores_above_low_risk(
    preprocessor: Any,
    model: ChurnMLP,
    high_risk_customer: dict[str, Any],
    low_risk_customer: dict[str, Any],
) -> None:
    """High-risk profile must score a higher churn probability than low-risk."""
    prob_high = _run_pipeline(preprocessor, model, high_risk_customer)
    prob_low = _run_pipeline(preprocessor, model, low_risk_customer)
    assert prob_high > prob_low, (
        f"high-risk={prob_high:.4f} not > low-risk={prob_low:.4f}"
    )


def test_pipeline_handles_missing_total_charges(
    preprocessor: Any, model: ChurnMLP, high_risk_customer: dict[str, Any]
) -> None:
    """Pipeline must produce a valid probability when TotalCharges is None (imputed)."""
    prob = _run_pipeline(
        preprocessor, model, {**high_risk_customer, "TotalCharges": None}
    )
    assert 0.0 <= prob <= 1.0
