"""Data validation tests using pandera."""

import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.preprocessing import RAW_SCHEMA


def test_valid_data_passes_schema():
    """Known-good data should pass validation."""
    df = pd.DataFrame(
        {
            "tenure": [12, 36],
            "MonthlyCharges": [50.0, 70.0],
            "TotalCharges": [600.0, 2520.0],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "PhoneService": ["Yes", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "Contract": ["Month-to-month", "Two year"],
        }
    )
    RAW_SCHEMA.validate(df)  # Should not raise


def test_negative_tenure_fails():
    """Negative tenure should fail validation."""
    df = pd.DataFrame(
        {
            "tenure": [-5],
            "MonthlyCharges": [50.0],
            "TotalCharges": [600.0],
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "PhoneService": ["Yes"],
            "InternetService": ["DSL"],
            "Contract": ["Month-to-month"],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)


def test_invalid_contract_fails():
    """Unknown contract type should fail validation."""
    df = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [50.0],
            "TotalCharges": [600.0],
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "PhoneService": ["Yes"],
            "InternetService": ["DSL"],
            "Contract": ["Weekly"],  # Invalid!
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)
