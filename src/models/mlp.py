"""MLP model definition for churn prediction."""

from typing import cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class ChurnMLP(nn.Module):
    """Multi-Layer Perceptron for binary churn classification.

    Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear -> Sigmoid

    Args:
        input_dim: Number of input features (after preprocessing).
        hidden_dims: List of hidden layer sizes. Default [64, 32].
        dropout: Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        # NOTE: No nn.Sigmoid() here! We output raw logits and use
        # BCEWithLogitsLoss during training (numerically stable).
        # Sigmoid is applied only during inference (see predict methods).

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: input tensor -> raw logits (NOT probabilities).

        During training: pass logits directly to BCEWithLogitsLoss.
        During inference: apply torch.sigmoid() to get probabilities.
        """
        return cast(torch.Tensor, self.network(x).squeeze(-1))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper: returns calibrated probabilities in [0, 1]."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


class SklearnChurnMLP(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Sklearn-compatible wrapper around ChurnMLP for use with sklearn utilities
    (ConfusionMatrixDisplay, classification_report, cross_val_score, etc.).

    Accepts numpy arrays, converts internally to tensors.
    """

    _estimator_type = "classifier"
    classes_ = np.array([0, 1])

    def __init__(self, model: ChurnMLP, threshold: float = 0.5, device: str = "cpu"):
        self.model = model.to(device)
        self.threshold = threshold
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = torch.tensor(X, dtype=torch.float32).to(self.device)
        probs = self.model.predict_proba(x).cpu().numpy()
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x = torch.tensor(X, dtype=torch.float32).to(self.device)
        probs = self.model.predict_proba(x).cpu().numpy()
        return np.column_stack([1 - probs, probs])
