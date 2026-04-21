import pickle
import tempfile

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features.preprocessing import build_preprocessor, load_and_split
from src.models.mlp import ChurnMLP, SklearnChurnMLP
from src.models.train import train_mlp
from src.utils import get_data_fingerprint, get_logger, set_seeds

logger = get_logger(__name__)
set_seeds(42)

EXPERIMENT_NAME = "mlp-vs-baselines"


def evaluate_and_log(model_name, y_true, y_pred, y_proba):
    """Compute metrics, print report, and log to the active MLflow run."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

    mlflow.log_metrics(metrics)

    print(f"\n{model_name}:")
    print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]))
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    return metrics


def log_confusion_matrix(model, X_test, y_test, name: str):
    """Save confusion matrix plot as MLflow artifact."""
    fig, ax = plt.subplots(figsize=(6, 5))
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"docs/figures/cm_{name}.png", dpi=150)
    mlflow.log_artifact(f"docs/figures/cm_{name}.png")
    plt.close()


def log_auc_curve(y_test, y_proba, name: str):
    """Save PR and ROC curves as MLflow artifact."""
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[0])
    axes[0].set_title(f"PR Curve — {name}")
    axes[0].plot([0, 1], [1, 0], "k--", label="Random (AUC=0.5)")
    axes[0].legend()

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title(f"ROC Curve — {name}")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"docs/figures/auc_{name}.png", dpi=150)
    mlflow.log_artifact(f"docs/figures/auc_{name}.png")
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    mlflow.set_experiment(EXPERIMENT_NAME)
    data_fp = get_data_fingerprint(X_train)

    # Save fitted preprocessor for the API (Stage 3)
    pickle.dump(preprocessor, open("models/preprocessor.pkl", "wb"))

    results = {}

    with mlflow.start_run(run_name="comparison") as parent_run:
        mlflow.log_params(data_fp)

        # --- 1. Logistic Regression ---
        with mlflow.start_run(run_name="logistic_regression", nested=True):
            mlflow.log_params(data_fp)
            mlflow.log_params(
                {
                    "model": "LogisticRegression",
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "random_state": 42,
                }
            )
            lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(X_train_processed, y_train)
            y_pred_lr = lr.predict(X_test_processed)
            y_proba_lr = lr.predict_proba(X_test_processed)[:, 1]
            results["Logistic Regression"] = evaluate_and_log(
                "Logistic Regression", y_test, y_pred_lr, y_proba_lr
            )
            log_confusion_matrix(lr, X_test_processed, y_test, "logistic_regression")
            log_auc_curve(y_test, y_proba_lr, "logistic_regression")
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="churn-logistic-regression"
            )

        # --- 2. Random Forest ---
        with mlflow.start_run(run_name="random_forest", nested=True):
            mlflow.log_params(data_fp)
            mlflow.log_params(
                {"model": "RandomForest", "n_estimators": 100, "max_depth": 10}
            )
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train_processed, y_train)
            y_pred_rf = rf.predict(X_test_processed)
            y_proba_rf = rf.predict_proba(X_test_processed)[:, 1]
            results["Random Forest"] = evaluate_and_log(
                "Random Forest", y_test, y_pred_rf, y_proba_rf
            )
            log_confusion_matrix(rf, X_test_processed, y_test, "random_forest")
            log_auc_curve(y_test, y_proba_rf, "random_forest")
            mlflow.sklearn.log_model(
                rf, "model", registered_model_name="churn-random-forest"
            )

        # --- 3. Gradient Boosting ---
        with mlflow.start_run(run_name="gradient_boosting", nested=True):
            mlflow.log_params(data_fp)
            mlflow.log_params(
                {
                    "model": "GradientBoosting",
                    "n_estimators": 200,
                    "lr": 0.1,
                    "max_depth": 4,
                }
            )
            gb = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            )
            gb.fit(X_train_processed, y_train)
            y_pred_gb = gb.predict(X_test_processed)
            y_proba_gb = gb.predict_proba(X_test_processed)[:, 1]
            results["Gradient Boosting"] = evaluate_and_log(
                "Gradient Boosting", y_test, y_pred_gb, y_proba_gb
            )
            log_confusion_matrix(gb, X_test_processed, y_test, "gradient_boosting")
            log_auc_curve(y_test, y_proba_gb, "gradient_boosting")
            mlflow.sklearn.log_model(
                gb, "model", registered_model_name="churn-gradient-boosting"
            )

        # --- 4. MLP (PyTorch) ---
        with mlflow.start_run(run_name="mlp_v1", nested=True):
            mlflow.log_params(data_fp)
            mlflow.log_params(
                {
                    "model": "MLP",
                    "hidden_dims": "[64, 32]",
                    "lr": 1e-3,
                    "batch_size": 64,
                    "dropout": 0.2,
                    "epochs": 200,
                    "patience": 50,
                }
            )

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_processed,
                y_train.values,
                test_size=0.15,
                random_state=42,
                stratify=y_train,
            )

            input_dim = X_tr.shape[1]
            mlp = train_mlp(
                ChurnMLP(input_dim=input_dim),
                X_tr,
                y_tr,
                X_val,
                y_val,
                epochs=200,
                batch_size=64,
                lr=1e-3,
                patience=50,
            )

            mlp.eval()
            with torch.no_grad():
                y_proba_mlp = mlp(torch.FloatTensor(X_test_processed)).numpy()

            y_pred_mlp = (y_proba_mlp >= 0.5).astype(int)
            results["MLP"] = evaluate_and_log("MLP", y_test, y_pred_mlp, y_proba_mlp)

            wrapped = SklearnChurnMLP(mlp, threshold=0.5, device="cpu")
            log_confusion_matrix(wrapped, X_test_processed, y_test, "mlp")
            log_auc_curve(y_test, y_proba_mlp, "mlp")

            torch.save(mlp.state_dict(), "models/mlp_churn.pt")
            mlflow.pytorch.log_model(
                mlp, "model", registered_model_name="churn-mlp"
            )

        # --- Comparison table ---
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values("pr_auc", ascending=False)
        print("\n" + "=" * 70)
        print("MODEL COMPARISON (sorted by PR-AUC)")
        print("=" * 70)
        print(comparison_df.to_string())
        print("=" * 70)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="model_comparison_", delete=False
        ) as f:
            comparison_df.to_csv(f.name)
            mlflow.log_artifact(f.name, artifact_path="comparison")

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pkl", prefix="preprocessor_", delete=False
        ) as f:
            pickle.dump(preprocessor, f)
            mlflow.log_artifact(f.name, artifact_path="preprocessor")


if __name__ == "__main__":
    main()
