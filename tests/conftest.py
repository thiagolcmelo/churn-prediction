import pandas as pd
import pytest

from src.utils import set_seeds


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    """Fix random seeds for reproducible tests."""
    set_seeds(42)


@pytest.fixture
def sample_data() -> pd.DataFrame:
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


@pytest.fixture
def sample_data_n(n: int = 20) -> pd.DataFrame:
    """Create a sample dataset as big as needed for testing."""
    return pd.DataFrame(
        {
            "customerID": [str(i) for i in range(n)],
            "tenure": [i % 72 for i in range(n)],
            "MonthlyCharges": [50.0 + i for i in range(n)],
            "TotalCharges": [f"{50.0 * (i % 72 + 1)}" for i in range(n)],
            "gender": ["Male" if i % 2 == 0 else "Female" for i in range(n)],
            "SeniorCitizen": [i % 2 for i in range(n)],
            "Partner": ["Yes" if i % 2 == 0 else "No" for i in range(n)],
            "Dependents": ["No"] * n,
            "PhoneService": ["Yes"] * n,
            "MultipleLines": ["No"] * n,
            "InternetService": [
                "DSL" if i % 2 == 0 else "Fiber optic" for i in range(n)
            ],
            "OnlineSecurity": ["No"] * n,
            "OnlineBackup": ["No"] * n,
            "DeviceProtection": ["No"] * n,
            "TechSupport": ["No"] * n,
            "StreamingTV": ["No"] * n,
            "StreamingMovies": ["No"] * n,
            "Contract": ["Month-to-month"] * n,
            "PaperlessBilling": ["Yes"] * n,
            "PaymentMethod": ["Electronic check"] * n,
            "Churn": ["Yes" if i % 5 == 0 else "No" for i in range(n)],
        }
    )
