"""Data preprocessing pipeline for churn prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

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


class TotalChargesFixer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Fix TotalCharges: impute NaN with median learned from training data.

    This prevents data leakage — the median is computed only from the
    training set during fit(), then applied identically to train and test.
    """

    def __init__(self) -> None:
        self.median_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TotalChargesFixer:
        self.median_ = X["TotalCharges"].median()
        logger.info(f"TotalChargesFixer: learned median={self.median_:.2f}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TotalCharges"] = X["TotalCharges"].fillna(self.median_)
        return X


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

    # Convert TotalCharges blanks to NaN (but do NOT impute yet — that
    # happens in the pipeline to avoid leaking test-set statistics)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop ID column
    df.drop(columns=["customerID"], inplace=True)

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    logger.info(
        f"Split: train={len(X_train)}, test={len(X_test)}, "
        f"churn_rate_train={y_train.mean():.3f}, "
        f"churn_rate_test={y_test.mean():.3f}"
    )

    return X_train, X_test, y_train, y_test
