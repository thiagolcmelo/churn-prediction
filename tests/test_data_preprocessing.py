"""Data validation tests using pandera."""

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.preprocessing import CAT_COLS, NUM_COLS, RAW_SCHEMA, prepare_dataset


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


def test_prepare_dataset(sample_data):
    """Verify prepare_dataset returns correct shapes and types."""
    X, y, num_feats, cat_feats = prepare_dataset(sample_data)
    assert X.shape[0] == 5, "Should have 5 rows"
    assert "customerID" not in X.columns, "customerID should be dropped"
    assert set(num_feats) == set(NUM_COLS)
    assert set(cat_feats) == set(CAT_COLS)
    assert y.dtype in [np.int64, np.int32, int], "Target should be integer"
    assert set(y.unique()).issubset({0, 1}), "Target should be 0 or 1"
