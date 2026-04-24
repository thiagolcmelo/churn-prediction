"""Training loop with early stopping for the churn MLP."""

from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from src.utils import get_logger

_ModelT = TypeVar("_ModelT", bound=nn.Module)

logger = get_logger(__name__)


def train_mlp(
    model: _ModelT,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 50,
) -> _ModelT:
    """Train the MLP with early stopping on validation loss.

    Args:
        model: ChurnMLP instance.
        X_train: Preprocessed training features (numpy array).
        y_train: Training labels (numpy array).
        X_val: Preprocessed validation features.
        y_val: Validation labels.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizer.
        patience: Early stopping patience (epochs without improvement).

    Returns:
        Trained model with best weights restored.
    """
    # Convert numpy arrays to PyTorch tensors
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_X = torch.FloatTensor(X_val)
    val_y = torch.FloatTensor(y_val)

    # Class weighting: penalize missing churners more heavily.
    # pos_weight = n_negative / n_positive (same idea as class_weight="balanced")
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])
    logger.info(
        f"Class weights: pos_weight={pos_weight.item():.2f} (neg={n_neg}, pos={n_pos})"
    )

    # BCEWithLogitsLoss: combines sigmoid + BCE in one numerically stable op.
    # The model outputs raw logits (no sigmoid layer), and this loss applies
    # sigmoid internally using the log-sum-exp trick for stability.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_val_pr_auc = 0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)  # raw logits, NOT probabilities
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation phase ---
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y).item()
            val_probs = torch.sigmoid(model(torch.FloatTensor(X_val))).numpy().flatten()

        val_pr_auc = average_precision_score(val_y.cpu().numpy(), val_probs)
        train_loss = np.mean(train_losses)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} — "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f} — "
                f"val_pr_auc: {val_pr_auc:.4f}"
            )

        # --- Early stopping check ---
        if (val_pr_auc > best_val_pr_auc) and (val_loss < best_val_loss):
            best_val_pr_auc = val_pr_auc
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_loss: {best_val_loss:.4f}. "
                    f"Best best_val_loss: {best_val_loss:.4f}."
                )
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model
