"""Tests for the train_mlp training loop."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.models.mlp import ChurnMLP
from src.models.train import train_mlp

INPUT_DIM = 4
N_TRAIN = 20
N_VAL = 10


@pytest.fixture
def model() -> ChurnMLP:
    """Tiny ChurnMLP for fast training in tests."""
    torch.manual_seed(0)
    return ChurnMLP(input_dim=INPUT_DIM, hidden_dims=[8])


@pytest.fixture
def train_data() -> tuple[np.ndarray, np.ndarray]:
    """Small training set with a learnable binary pattern."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_TRAIN, INPUT_DIM)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y


@pytest.fixture
def val_data() -> tuple[np.ndarray, np.ndarray]:
    """Small validation set matching the training pattern."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N_VAL, INPUT_DIM)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y


def test_train_mlp_returns_same_model(
    model: ChurnMLP,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_mlp() returns the exact same object it received."""
    X_tr, y_tr = train_data
    X_val, y_val = val_data
    result = train_mlp(model, X_tr, y_tr, X_val, y_val, epochs=2, patience=10)
    assert result is model


def test_train_mlp_preserves_concrete_type(
    model: ChurnMLP,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_mlp() return type is ChurnMLP, not the base nn.Module."""
    X_tr, y_tr = train_data
    X_val, y_val = val_data
    result = train_mlp(model, X_tr, y_tr, X_val, y_val, epochs=2, patience=10)
    assert isinstance(result, ChurnMLP)


def test_train_mlp_model_in_eval_mode_after_training(
    model: ChurnMLP,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_mlp() leaves the model in eval mode, ready for inference."""
    X_tr, y_tr = train_data
    X_val, y_val = val_data
    train_mlp(model, X_tr, y_tr, X_val, y_val, epochs=2, patience=10)
    assert not model.training


def test_train_mlp_weights_update_from_initial(
    model: ChurnMLP,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_mlp() updates model weights — they differ from random initialisation."""
    X_tr, y_tr = train_data
    X_val, y_val = val_data
    initial_weights = [p.clone() for p in model.parameters()]
    train_mlp(model, X_tr, y_tr, X_val, y_val, epochs=5, patience=10)
    any_changed = any(
        not torch.equal(p, w) for p, w in zip(model.parameters(), initial_weights)
    )
    assert any_changed


def test_train_mlp_output_shape_preserved(
    model: ChurnMLP,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Model produces correct output shape after training."""
    X_tr, y_tr = train_data
    X_val, y_val = val_data
    trained = train_mlp(model, X_tr, y_tr, X_val, y_val, epochs=2, patience=10)
    assert trained(torch.randn(N_VAL, INPUT_DIM)).shape == (N_VAL,)


def test_train_mlp_early_stopping_triggers(
    model: ChurnMLP,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Early stopping halts training before epochs are exhausted when val metrics plateau."""
    X_tr, y_tr = train_data
    X_val, y_val = val_data

    # AUC returns 0.8 on epoch 1 then 0.1 forever — no further improvement possible.
    auc_seq = [0.8] + [0.1] * 200
    with patch(
        "src.models.train.average_precision_score", side_effect=auc_seq
    ) as mock_auc:
        train_mlp(model, X_tr, y_tr, X_val, y_val, epochs=100, patience=2)

    # patience=2 means at most 3 AUC calls (epoch 1 improves, epochs 2-3 don't).
    assert mock_auc.call_count <= 4


def test_train_mlp_restores_best_weights(
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """train_mlp() restores the best-epoch checkpoint, not the final epoch weights."""
    X_tr, y_tr = train_data
    X_val, y_val = val_data

    # Train m1 for exactly 1 epoch to capture true epoch-1 weights.
    torch.manual_seed(0)
    m1 = ChurnMLP(input_dim=INPUT_DIM, hidden_dims=[8])
    with patch("src.models.train.average_precision_score", return_value=0.9):
        train_mlp(m1, X_tr, y_tr, X_val, y_val, epochs=1, patience=10)
    epoch1_state = {k: v.clone() for k, v in m1.state_dict().items()}

    # Train m2 with AUC peaking at epoch 1 then dropping — early stopping at epoch 2.
    # With the same seed, epoch-1 weights are identical. Best state must be restored.
    torch.manual_seed(0)
    m2 = ChurnMLP(input_dim=INPUT_DIM, hidden_dims=[8])
    auc_seq = [0.9] + [0.1] * 100
    with patch("src.models.train.average_precision_score", side_effect=auc_seq):
        train_mlp(m2, X_tr, y_tr, X_val, y_val, epochs=50, patience=1)

    for key, expected in epoch1_state.items():
        assert torch.allclose(m2.state_dict()[key], expected, atol=1e-6), (
            f"Parameter '{key}' was not restored to its best-epoch value"
        )
