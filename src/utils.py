"""Shared utilities: reproducibility, logging, and data versioning."""

import hashlib
import logging
import random
from typing import Any

import numpy as np
import pandas as pd
import torch


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger with consistent formatting.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_data_fingerprint(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute a reproducible fingerprint of the full dataset for MLflow tracking.

    Hashes features AND labels together so the fingerprint catches relabelling,
    target-encoding changes, or different splits — not just feature drift.

    Returns:
        Dict with data_version, data_hash, data_rows, data_train_rows,
        data_test_rows, and data_source.
    """
    full_X = pd.concat([X_train, X_test], ignore_index=True)
    full_y = pd.concat([y_train, y_test], ignore_index=True)
    hash_bytes = (
        pd.util.hash_pandas_object(full_X).values.tobytes()
        + pd.util.hash_pandas_object(full_y).values.tobytes()
    )
    data_hash = hashlib.md5(hash_bytes).hexdigest()[:8]
    return {
        "data_version": "v1.0",
        "data_hash": data_hash,
        "data_rows": len(full_X),
        "data_train_rows": len(X_train),
        "data_test_rows": len(X_test),
        "data_source": "telco-customer-churn (IBM/Kaggle)",
    }


def set_seeds(seed: int = 42) -> None:
    """Pin every source of randomness for reproducible results.

    Args:
        seed: Integer seed value. Default 42 (convention).
    """
    random.seed(seed)  # Python stdlib random
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch CPU random
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random (all GPUs)

    # Force deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
