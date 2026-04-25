"""Tests for the FastAPI app app and Pydantic schemas."""

from collections.abc import Generator
from typing import Any

from api.schemas import CustomerInput, PredictionOutput
from pydantic import ValidationError
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the API."""
    from src.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_customer() -> dict[str, Any]:
    """A valid customer payload for testing."""
    return {
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
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


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    """GET /health should return HTTP 200."""
    assert client.get("/health").status_code == 200


def test_health_response_body(client: TestClient) -> None:
    """GET /health should return the expected status and model name."""
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model"] is not None


# ---------------------------------------------------------------------------
# CustomerInput schema
# ---------------------------------------------------------------------------


def test_customer_input_valid_payload_parses(sample_customer: dict[str, Any]) -> None:
    """All required fields with valid values should parse without error."""
    customer = CustomerInput(**sample_customer)
    assert customer.tenure == 12
    assert customer.gender == "Male"


def test_customer_input_negative_tenure_fails(sample_customer: dict[str, Any]) -> None:
    """tenure < 0 should be rejected by the ge=0 constraint."""
    with pytest.raises(ValidationError):
        CustomerInput(**{**sample_customer, "tenure": -1})


def test_customer_input_invalid_gender_fails(sample_customer: dict[str, Any]) -> None:
    """gender must be 'Male' or 'Female' — any other value should be rejected."""
    with pytest.raises(ValidationError):
        CustomerInput(**{**sample_customer, "gender": "Other"})


def test_customer_input_invalid_contract_fails(sample_customer: dict[str, Any]) -> None:
    """Contract must be one of the three allowed values."""
    with pytest.raises(ValidationError):
        CustomerInput(**{**sample_customer, "Contract": "Weekly"})


def test_customer_input_senior_citizen_out_of_range_fails(
    sample_customer: dict[str, Any],
) -> None:
    """SeniorCitizen must be 0 or 1 — values outside that range should be rejected."""
    with pytest.raises(ValidationError):
        CustomerInput(**{**sample_customer, "SeniorCitizen": 2})


def test_customer_input_missing_required_field_fails(
    sample_customer: dict[str, Any],
) -> None:
    """Omitting a required field should raise a ValidationError."""
    payload = {k: v for k, v in sample_customer.items() if k != "tenure"}
    with pytest.raises(ValidationError):
        CustomerInput(**payload)


def test_customer_input_total_charges_none_becomes_none(
    sample_customer: dict[str, Any],
) -> None:
    """Explicit None for TotalCharges is accepted — the pipeline imputes it."""
    customer = CustomerInput(**{**sample_customer, "TotalCharges": None})
    assert customer.TotalCharges is None


def test_customer_input_total_charges_blank_string_becomes_none(
    sample_customer: dict[str, Any],
) -> None:
    """Blank string for TotalCharges is coerced to None before validation."""
    customer = CustomerInput(**{**sample_customer, "TotalCharges": "  "})
    assert customer.TotalCharges is None


def test_customer_input_total_charges_string_float_is_coerced(
    sample_customer: dict[str, Any],
) -> None:
    """A numeric string for TotalCharges is coerced to float."""
    customer = CustomerInput(**{**sample_customer, "TotalCharges": "840.0"})
    assert customer.TotalCharges == pytest.approx(840.0)


# ---------------------------------------------------------------------------
# PredictionOutput schema
# ---------------------------------------------------------------------------


def test_prediction_output_valid_parses() -> None:
    """Valid probability, prediction and threshold values should parse cleanly."""
    output = PredictionOutput(
        churn_probability=0.73, churn_prediction=True, threshold=0.5
    )
    assert output.churn_probability == pytest.approx(0.73)
    assert output.churn_prediction is True


def test_prediction_output_probability_above_one_fails() -> None:
    """churn_probability > 1 violates the le=1 constraint."""
    with pytest.raises(ValidationError):
        PredictionOutput(churn_probability=1.1, churn_prediction=True, threshold=0.5)


def test_prediction_output_negative_probability_fails() -> None:
    """churn_probability < 0 violates the ge=0 constraint."""
    with pytest.raises(ValidationError):
        PredictionOutput(churn_probability=-0.1, churn_prediction=False, threshold=0.5)
