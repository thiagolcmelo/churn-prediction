"""Data validation tests using pandera."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.preprocessing import (
    CAT_COLS,
    NUM_COLS,
    PREPARED_SCHEMA,
    build_preprocessor,
    load_and_split,
    prepare_dataset,
)


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
    PREPARED_SCHEMA.validate(df)  # Should not raise


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
        PREPARED_SCHEMA.validate(df)


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
        PREPARED_SCHEMA.validate(df)


def test_prepare_dataset(sample_data):
    """Verify prepare_dataset returns correct shapes and types."""
    X, y, num_feats, cat_feats = prepare_dataset(sample_data)
    assert X.shape[0] == 5, "Should have 5 rows"
    assert "customerID" not in X.columns, "customerID should be dropped"
    assert set(num_feats) == set(NUM_COLS)
    assert set(cat_feats) == set(CAT_COLS)
    assert y.dtype in [np.int64, np.int32, int], "Target should be integer"
    assert set(y.unique()).issubset({0, 1}), "Target should be 0 or 1"


def test_prepare_dataset_encodes_churn(sample_data):
    """Prepare should conver "Yes" to 1 and "No" to 0."""
    _, y, _, _ = prepare_dataset(sample_data)
    assert sorted(set(y)) == [0, 1], "Churn should be encoded as 0/1"


def test_prepare_dataset_drops_customer_id(sample_data):
    """Column customerID was deemed not useful and must be dropped."""
    X, _, _, _ = prepare_dataset(sample_data)
    assert "customerID" not in X.columns


def test_prepare_dataset_coerces_total_charges_blanks(sample_data):
    """TotalCharges should become NaN for imputation later."""
    sample_data.loc[0, "TotalCharges"] = " "
    X, _, _, _ = prepare_dataset(sample_data)
    assert pd.isna(X["TotalCharges"].iloc[0]), "Blank TotalCharges should become NaN"


def test_preprocessor_output_shape(sample_data):
    """Verify preprocessor produces the right number of columns."""
    X, _, _, _ = prepare_dataset(sample_data)
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    assert X_processed.shape[0] == 5, "Row count should be preserved"
    assert X_processed.shape[1] > len(NUM_COLS), "One-hot encoding should add columns"


def test_build_preprocessor_produces_numeric_output(sample_data):
    """Preprocessor output must be completely numeric."""
    preprocessor = build_preprocessor()
    X, _, _, _ = prepare_dataset(sample_data)
    result = preprocessor.fit_transform(X)
    assert result.dtype == float
    assert not pd.isna(result).any()


@patch("src.data.preprocessing.pd.read_csv")
def test_load_and_split_stratification(mock_read_csv, sample_data_n):
    """Churn rate should be similar in train and test."""
    mock_read_csv.return_value = sample_data_n
    _, _, y_train, y_test = load_and_split()
    assert abs(y_train.mean() - y_test.mean()) < 0.1, (
        "Churn rate should be similar in train and test"
    )
