"""Shared utilities: reproducibility, logging, and data versioning."""

import hashlib
import logging
import random

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


def get_data_fingerprint(df: pd.DataFrame) -> dict:
    """Compute a reproducible fingerprint of the dataset for MLflow tracking.

    The challenge requires logging 'dataset version' alongside parameters and metrics.
    This function creates a hash of the data content so you can detect if the dataset
    changed between experiments — essential for debugging model regressions.

    Returns:
        Dict with data_version, data_hash, data_rows, and data_source.
    """
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:8]
    return {
        "data_version": "v1.0",
        "data_hash": data_hash,
        "data_rows": len(df),
        "data_source": "telco-customer-churn (IBM/Kaggle)",
    }


def set_seeds(seed: int = 42) -> None:
    """Pin every source of randomness for reproducible results.

    Args:
        seed: Integer seed value. Default 42 (convention).
    """
    random.seed(seed)           # Python stdlib random
    np.random.seed(seed)        # NumPy random generator
    torch.manual_seed(seed)     # PyTorch CPU random
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random (all GPUs)

    # Force deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
