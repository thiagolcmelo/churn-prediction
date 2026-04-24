"""Baseline models: DummyClassifier and Logistic Regression with MLflow tracking."""

import matplotlib.pyplot as plt
import mlflow
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessor, load_and_split
from src.utils import get_data_fingerprint, get_logger, set_seeds

logger = get_logger(__name__)
set_seeds(42)

EXPERIMENT_NAME = "churn-baselines"


def evaluate_model(
    model: BaseEstimator, X_test: ArrayLike, y_test: ArrayLike
) -> dict[str, float]:
    """Compute all metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        metrics["pr_auc"] = average_precision_score(y_test, y_prob)

    return metrics


def log_confusion_matrix(
    model: BaseEstimator, X_test: ArrayLike, y_test: ArrayLike, name: str
) -> None:
    """Save confusion matrix plot as MLflow artifact."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"docs/figures/cm_{name}.png", dpi=150)
    mlflow.log_artifact(f"docs/figures/cm_{name}.png")
    plt.close()


def log_auc_curve(
    model: BaseEstimator, X_test: ArrayLike, y_test: ArrayLike, name: str
) -> None:
    """Save PR-ROC and ROC-AUC curves plot as MLflow artifact."""
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
    axes[0].set_title("PR Curve — logistic_regression")
    axes[0].plot([0, 1], [1, 0], "k--", label="Random (AUC=0.5)")
    axes[0].legend()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].set_title("ROC Curve — logistic_regression")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"docs/figures/auc_{name}.png", dpi=150)
    mlflow.log_artifact(f"docs/figures/auc_{name}.png")
    plt.close()


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Log dataset version in every run (challenge requirement)
    data_fp = get_data_fingerprint(X_train)

    # -- Baseline 1: DummyClassifier ------------------------------------
    with mlflow.start_run(run_name="dummy-most-frequent"):
        mlflow.log_params(data_fp)  # dataset version tracking
        dummy_pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        )
        dummy_pipe.fit(X_train, y_train)
        metrics = evaluate_model(dummy_pipe, X_test, y_test)

        mlflow.log_param("model_type", "DummyClassifier")
        mlflow.log_param("strategy", "most_frequent")
        mlflow.log_metrics(metrics)

        logger.info(
            f"Dummy — Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}"
        )
        log_confusion_matrix(dummy_pipe, X_test, y_test, "dummy")

    # -- Baseline 2: Logistic Regression --------------------------------
    with mlflow.start_run(run_name="logistic-regression"):
        mlflow.log_params(data_fp)  # dataset version tracking
        lr_pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        # Scoring metrics
        scoring_metrics = [
            "accuracy",
            "average_precision",  # pr_auc
            "f1",
            "roc_auc",
        ]

        # Cross-validation for reliable estimate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            lr_pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring_metrics,
            return_train_score=False,
        )

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("cv_folds", 5)

        for metric_name in scoring_metrics:
            scores = cv_results[f"test_{metric_name}"]
            mlflow.log_metric(f"cv_{metric_name}_mean", scores.mean())
            mlflow.log_metric(f"cv_{metric_name}_std", scores.std())
            logger.info(
                f"LR CV {metric_name}: {scores.mean():.3f} +/- {scores.std():.3f}"
            )

        # Fit on full training set for test evaluation
        lr_pipe.fit(X_train, y_train)
        test_metrics = evaluate_model(lr_pipe, X_test, y_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        logger.info(
            f"LR Test — PR-AUC: {test_metrics.get('pr_auc', 'N/A'):.3f}, "
            f"ROC-AUC: {test_metrics.get('roc_auc', 'N/A'):.3f}, "
            f"F1: {test_metrics['f1']:.3f}"
        )

        log_confusion_matrix(lr_pipe, X_test, y_test, "logistic_regression")
        log_auc_curve(lr_pipe, X_test, y_test, "logistic_regression")

        # Save the pipeline
        mlflow.sklearn.log_model(lr_pipe, "model")

    logger.info("Baselines complete. View results: mlflow ui --port 5001")


if __name__ == "__main__":
    main()
