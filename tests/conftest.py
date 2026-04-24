import pandas as pd
import pytest

from utils import set_seeds


@pytest.fixture(autouse=True)
def seed_everything():
    """Fix random seeds for reproducible tests."""
    set_seeds(42)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    return pd.DataFrame(
        {
            "customerID": ["001", "002", "003", "004", "005"],
            "tenure": [1, 12, 36, 48, 72],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
            "TotalCharges": ["29.85", "683.4", "1938.6", "2030.4", "5088.4"],
            "gender": ["Female", "Male", "Male", "Male", "Female"],
            "SeniorCitizen": [0, 0, 0, 0, 1],
            "Partner": ["Yes", "No", "No", "Yes", "Yes"],
            "Dependents": ["No", "No", "No", "No", "No"],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "No", "Yes", "Yes"],
            "InternetService": ["DSL", "DSL", "DSL", "Fiber optic", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes", "Yes", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "No", "No", "No"],
            "DeviceProtection": ["No", "Yes", "Yes", "Yes", "Yes"],
            "TechSupport": ["No", "No", "No", "No", "No"],
            "StreamingTV": ["No", "No", "No", "No", "Yes"],
            "StreamingMovies": ["No", "No", "Yes", "Yes", "Yes"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Electronic check",
            ],
            "Churn": ["No", "No", "Yes", "No", "Yes"],
        }
    )
