"""Tests for ChurnMLP and SklearnChurnMLP."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.mlp import ChurnMLP, SklearnChurnMLP

INPUT_DIM = 10
BATCH_SIZE = 4


@pytest.fixture
def model() -> ChurnMLP:
    """Small ChurnMLP with default hidden dims for fast testing."""
    return ChurnMLP(input_dim=INPUT_DIM)


@pytest.fixture
def x_tensor() -> torch.Tensor:
    """Random float32 input tensor of shape (BATCH_SIZE, INPUT_DIM)."""
    return torch.randn(BATCH_SIZE, INPUT_DIM)


@pytest.fixture
def X_array() -> np.ndarray:
    """Random numpy input array of shape (BATCH_SIZE, INPUT_DIM)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype(np.float32)


# --- ChurnMLP ---

def test_mlp_forward_pass() -> None:
    """Verify MLP produces output with correct shape for any input dimension.

    NOTE: We don't hardcode 46 (the OneHotEncoder output size) because that
    number depends on the dataset's unique categories and could change.
    Instead, we test with an arbitrary dimension to verify the architecture works.
    """
    input_dim = 20  # arbitrary — tests architecture, not data shape
    model = ChurnMLP(input_dim=input_dim)
    model.eval()
    x = torch.randn(8, input_dim)  # batch of 8 samples

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (8,), f"Expected shape (8,), got {logits.shape}"
    # Model outputs raw logits now (not probabilities), so no [0,1] check here


def test_mlp_output_changes_after_train_step() -> None:
    """Verify that one training step actually updates the model weights."""
    model = ChurnMLP(input_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()  # matches production training

    x = torch.randn(4, 10)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])

    # Get output before training (raw logits)
    model.eval()
    with torch.no_grad():
        before = model(x).clone()

    # One training step
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(x), y)  # logits -> BCEWithLogitsLoss
    loss.backward()
    optimizer.step()

    # Get output after training
    model.eval()
    with torch.no_grad():
        after = model(x)

    assert not torch.allclose(before, after), \
        "Model output should change after a training step"


def test_mlp_end_to_end_with_preprocessor() -> None:
    """End-to-end test: raw DataFrame → preprocessor → MLP → prediction.

    This is the ROBUST test — it uses the actual preprocessor to determine
    input_dim dynamically, so it never breaks if the dataset changes.
    """
    import pickle, os
    preprocessor_path = "models/preprocessor.pkl"
    if not os.path.exists(preprocessor_path):
        pytest.skip("Preprocessor not fitted yet — run training first")

    preprocessor = pickle.load(open(preprocessor_path, "rb"))
    sample = pd.DataFrame([{
        "tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0,
        "Contract": "Month-to-month", "InternetService": "Fiber optic",
        "TechSupport": "No", "OnlineSecurity": "No", "gender": "Male",
        "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    }])
    X_processed = preprocessor.transform(sample)
    input_dim = X_processed.shape[1]

    model = ChurnMLP(input_dim=input_dim)
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_processed))
        proba = torch.sigmoid(logits)

    assert proba.shape == (1,)
    assert 0 <= proba.item() <= 1, "Probability must be in [0, 1]"


def test_forward_output_shape(model: ChurnMLP, x_tensor: torch.Tensor) -> None:
    """forward() returns shape (batch,), not (batch, 1) — squeeze is applied."""
    model.eval()
    out = model(x_tensor)
    assert out.shape == (BATCH_SIZE,)


def test_forward_outputs_logits_not_probabilities(
    model: ChurnMLP, x_tensor: torch.Tensor
) -> None:
    """forward() returns raw logits; sigmoid(logits) must equal predict_proba()."""
    model.eval()
    logits = model(x_tensor)
    assert torch.allclose(torch.sigmoid(logits), model.predict_proba(x_tensor))


def test_predict_proba_range(model: ChurnMLP, x_tensor: torch.Tensor) -> None:
    """predict_proba() values are bounded in [0, 1]."""
    probs = model.predict_proba(x_tensor)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_proba_sets_eval_mode(model: ChurnMLP, x_tensor: torch.Tensor) -> None:
    """predict_proba() switches the model to eval mode."""
    model.train()
    model.predict_proba(x_tensor)
    assert not model.training


def test_predict_proba_no_grad(model: ChurnMLP, x_tensor: torch.Tensor) -> None:
    """predict_proba() runs under no_grad — output tensor has no gradient."""
    probs = model.predict_proba(x_tensor)
    assert not probs.requires_grad


def test_default_hidden_dims_architecture(model: ChurnMLP) -> None:
    """Omitting hidden_dims builds [64, 32] layers by default."""
    linear_layers = [m for m in model.network if isinstance(m, torch.nn.Linear)]
    assert linear_layers[0].out_features == 64
    assert linear_layers[1].out_features == 32
    assert linear_layers[2].out_features == 1


def test_custom_hidden_dims_architecture() -> None:
    """Passing hidden_dims builds exactly those layer sizes."""
    m = ChurnMLP(input_dim=INPUT_DIM, hidden_dims=[128, 64, 32])
    linear_layers = [layer for layer in m.network if isinstance(layer, torch.nn.Linear)]
    assert [ll.out_features for ll in linear_layers] == [128, 64, 32, 1]


def test_single_hidden_dim() -> None:
    """A single hidden layer produces a valid model."""
    m = ChurnMLP(input_dim=INPUT_DIM, hidden_dims=[16])
    m.eval()
    out = m(torch.randn(2, INPUT_DIM))
    assert out.shape == (2,)


def test_forward_single_sample_in_eval_mode() -> None:
    """forward() works with batch size 1 when model is in eval mode."""
    m = ChurnMLP(input_dim=INPUT_DIM)
    m.eval()
    out = m(torch.randn(1, INPUT_DIM))
    assert out.shape == (1,)


# --- SklearnChurnMLP ---


@pytest.fixture
def wrapped(model: ChurnMLP) -> SklearnChurnMLP:
    """SklearnChurnMLP wrapping the default ChurnMLP fixture."""
    return SklearnChurnMLP(model)


def test_sklearn_predict_returns_binary(
    wrapped: SklearnChurnMLP, X_array: np.ndarray
) -> None:
    """predict() returns only 0s and 1s."""
    preds = wrapped.predict(X_array)
    assert set(preds).issubset({0, 1})


def test_sklearn_predict_output_shape(
    wrapped: SklearnChurnMLP, X_array: np.ndarray
) -> None:
    """predict() returns a 1-D array with one label per sample."""
    preds = wrapped.predict(X_array)
    assert preds.shape == (BATCH_SIZE,)


def test_sklearn_predict_proba_shape(
    wrapped: SklearnChurnMLP, X_array: np.ndarray
) -> None:
    """predict_proba() returns shape (n_samples, 2)."""
    probs = wrapped.predict_proba(X_array)
    assert probs.shape == (BATCH_SIZE, 2)


def test_sklearn_predict_proba_rows_sum_to_one(
    wrapped: SklearnChurnMLP, X_array: np.ndarray
) -> None:
    """predict_proba() columns are complementary and sum to 1 per row."""
    probs = wrapped.predict_proba(X_array)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(BATCH_SIZE), atol=1e-6)


def test_sklearn_threshold_zero_predicts_all_positive(
    wrapped: SklearnChurnMLP, X_array: np.ndarray
) -> None:
    """threshold=0 makes every sample positive since prob >= 0 is always true."""
    wrapped.threshold = 0.0
    assert wrapped.predict(X_array).all()


def test_sklearn_threshold_one_predicts_all_negative(
    wrapped: SklearnChurnMLP, X_array: np.ndarray
) -> None:
    """threshold=1 makes every sample negative since sigmoid output is always < 1."""
    wrapped.threshold = 1.0
    assert not wrapped.predict(X_array).any()


def test_sklearn_classes_attribute(wrapped: SklearnChurnMLP) -> None:
    """classes_ exposes [0, 1] as required by the sklearn classifier interface."""
    np.testing.assert_array_equal(wrapped.classes_, np.array([0, 1]))


def test_sklearn_estimator_type(wrapped: SklearnChurnMLP) -> None:
    """_estimator_type is 'classifier' for sklearn compatibility."""
    assert wrapped._estimator_type == "classifier"
