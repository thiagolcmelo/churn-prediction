"""Data preprocessing pipeline for churn prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import get_logger

logger = get_logger(__name__)


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
