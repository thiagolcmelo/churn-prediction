"""Data preprocessing helper for churn prediction."""

from __future__ import annotations

import pandas as pd
from pandera import Check, Column, DataFrameSchema
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from src.features.preprocessing import TotalChargesFixer
from src.utils import get_logger, set_seeds

logger = get_logger(__name__)

# Define column groups
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
# NOTE: SeniorCitizen is in NUM_COLS (not CAT_COLS) because it's already
# binary 0/1 integer in the dataset. Putting it in CAT_COLS would cause
# dtype mismatches when the API receives it as a JSON integer.
CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET = "Churn"


# Define expected schema for raw input data
RAW_SCHEMA = DataFrameSchema(  # type: ignore[no-untyped-call]
    {
        "customerID": Column(str),
        "tenure": Column(int, Check.ge(0), nullable=False),
        "MonthlyCharges": Column(float, Check.ge(0), nullable=False),
        "TotalCharges": Column(object),
        "gender": Column(str, Check.isin(["Male", "Female"]), nullable=False),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), nullable=False),
        "Partner": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "Dependents": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "PhoneService": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "InternetService": Column(
            str, Check.isin(["DSL", "Fiber optic", "No"]), nullable=False
        ),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"]), nullable=False
        ),
        "Churn": Column(str, Check.isin(["Yes", "No"])),
    }
)


# Useful for catching inconsistencies in data preparation
PREPARED_SCHEMA = DataFrameSchema(  # type: ignore[no-untyped-call]
    {
        "tenure": Column(int, Check.ge(0), nullable=False),
        "MonthlyCharges": Column(float, Check.ge(0), nullable=False),
        # nullable=True: blank TotalCharges rows become NaN here intentionally;
        # TotalChargesFixer imputes them inside the pipeline to avoid data leakage.
        "TotalCharges": Column(float, Check.ge(0), nullable=True),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), nullable=False),
        "gender": Column(str, Check.isin(["Male", "Female"]), nullable=False),
        "Partner": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "Dependents": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "PhoneService": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "MultipleLines": Column(
            str, Check.isin(["Yes", "No", "No phone service"]), nullable=False
        ),
        "InternetService": Column(
            str, Check.isin(["DSL", "Fiber optic", "No"]), nullable=False
        ),
        "OnlineSecurity": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=False
        ),
        "OnlineBackup": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=False
        ),
        "DeviceProtection": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=False
        ),
        "TechSupport": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=False
        ),
        "StreamingTV": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=False
        ),
        "StreamingMovies": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=False
        ),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"]), nullable=False
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
            nullable=False,
        ),
    }
)


def build_preprocessor(polynomial: bool = False) -> Pipeline:
    """Build the full preprocessing pipeline: imputation + scaling + encoding.

    The Pipeline ensures TotalCharges imputation uses only training-set
    statistics, preventing the data leakage bug described in Section 1.5.
    """
    num_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            *(
                [
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=2, interaction_only=True, include_bias=False
                        ),
                    )
                ]
                if polynomial
                else []
            ),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_COLS),
            (
                "cat",
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
                CAT_COLS,
            ),
        ]
    )
    return Pipeline(
        [
            ("fix_total_charges", TotalChargesFixer()),
            ("column_transform", column_transformer),
        ]
    )


def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Clean raw dataframe and return X, y, feature lists.

    NOTE: TotalCharges NaN imputation is NOT done here — it happens
    inside the pipeline (TotalChargesFixer) to prevent data leakage.

    Returns:
        (X, y, num_features, cat_features) tuple
    """
    df = df.copy()

    # Convert TotalCharges blanks to NaN (but do NOT impute yet — that
    # happens in the pipeline to avoid leaking test-set statistics)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop ID column
    df.drop(columns=["customerID"], inplace=True)

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    return X, y, NUM_COLS, CAT_COLS


def load_and_split(
    path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load CSV, clean, and split into train/test.

    NOTE: We do NOT impute TotalCharges here — that happens inside the
    preprocessing pipeline (TotalChargesFixer) to prevent data leakage.
    We only convert blanks to NaN so the pipeline can handle them.

    Returns:
        (X_train, X_test, y_train, y_test) tuple
    """
    set_seeds(seed)

    df = pd.read_csv(path)
    RAW_SCHEMA.validate(df)

    X, y, _, _ = prepare_dataset(df)
    PREPARED_SCHEMA.validate(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    logger.info(
        f"Split: train={len(X_train)}, test={len(X_test)}, "
        f"churn_rate_train={y_train.mean():.3f}, "
        f"churn_rate_test={y_test.mean():.3f}"
    )

    return X_train, X_test, y_train, y_test
