"""Tests for TotalChargesFixer transformer."""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import TotalChargesFixer


@pytest.fixture
def charges_df() -> pd.DataFrame:
    """DataFrame with TotalCharges including one NaN."""
    return pd.DataFrame({"TotalCharges": [100.0, 200.0, np.nan, 400.0]})


def test_fit_learns_correct_median(charges_df: pd.DataFrame) -> None:
    """fit() stores the median of TotalCharges from the training data."""
    fixer = TotalChargesFixer().fit(charges_df)
    assert fixer.median_ == pytest.approx(200.0)


def test_fit_returns_self(charges_df: pd.DataFrame) -> None:
    """fit() returns the transformer instance for pipeline chaining."""
    fixer = TotalChargesFixer()
    assert fixer.fit(charges_df) is fixer


def test_transform_fills_nan_with_median(charges_df: pd.DataFrame) -> None:
    """transform() replaces NaN with the median learned during fit."""
    fixer = TotalChargesFixer().fit(charges_df)
    result = fixer.transform(charges_df)
    assert result["TotalCharges"].iloc[2] == pytest.approx(200.0)


def test_transform_preserves_non_nan_values(charges_df: pd.DataFrame) -> None:
    """transform() leaves existing non-NaN values unchanged."""
    fixer = TotalChargesFixer().fit(charges_df)
    result = fixer.transform(charges_df)
    assert list(result["TotalCharges"].dropna()) == [100.0, 200.0, 200.0, 400.0]


def test_transform_does_not_mutate_input(charges_df: pd.DataFrame) -> None:
    """transform() returns a copy and leaves the original DataFrame intact."""
    fixer = TotalChargesFixer().fit(charges_df)
    fixer.transform(charges_df)
    assert pd.isna(charges_df["TotalCharges"].iloc[2])


def test_transform_uses_fit_time_median() -> None:
    """transform() uses the median from fit, not recomputed from current data."""
    train = pd.DataFrame({"TotalCharges": [10.0, 20.0, 30.0]})
    test = pd.DataFrame({"TotalCharges": [np.nan, 1000.0]})

    fixer = TotalChargesFixer().fit(train)
    result = fixer.transform(test)

    assert result["TotalCharges"].iloc[0] == pytest.approx(20.0)


def test_transform_no_nans_changes_nothing(charges_df: pd.DataFrame) -> None:
    """transform() on a complete column produces identical values."""
    df = pd.DataFrame({"TotalCharges": [10.0, 20.0, 30.0]})
    fixer = TotalChargesFixer().fit(df)
    result = fixer.transform(df)
    pd.testing.assert_series_equal(result["TotalCharges"], df["TotalCharges"])
