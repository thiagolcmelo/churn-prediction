"""Pydantic schemas for API request/response validation."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CustomerInput(BaseModel):
    """Input schema for churn prediction.

    Each field has type annotations and validation constraints.
    FastAPI uses this to validate incoming JSON automatically.
    """

    # Numerical features
    tenure: int = Field(ge=0, description="Months as customer")
    MonthlyCharges: float = Field(ge=0, description="Monthly charge amount")
    TotalCharges: float | str | None = Field(
        default=None,
        description="Total charges to date. Accepts float, blank string, or null — "
        "the ML pipeline's TotalChargesFixer handles missing values via median imputation.",
    )

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def coerce_total_charges(cls, v: float | str | None) -> float | None:
        """Accept blank strings and None — let the preprocessor handle imputation."""
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None  # will become NaN in the DataFrame → TotalChargesFixer imputes
        return float(v)  # normal case: convert to float

    SeniorCitizen: int = Field(ge=0, le=1, description="1 if senior citizen")

    # Categorical features
    gender: Literal["Male", "Female"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 12,
                    "MonthlyCharges": 70.0,
                    "TotalCharges": 840.0,
                    "SeniorCitizen": 0,
                    "gender": "Male",
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
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Output schema for churn prediction."""

    churn_probability: float = Field(
        ge=0, le=1, description="Probability of churn (0 to 1)"
    )
    churn_prediction: bool = Field(description="True if churn_probability >= threshold")
    threshold: float = Field(ge=0, le=1, description="Classification threshold used")
