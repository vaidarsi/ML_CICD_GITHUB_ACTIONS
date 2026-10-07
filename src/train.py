"""Train the model, evaluate it, save it, and log everything to MLflow."""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # no GUI needed (works on servers and in CI)
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from src import config  # noqa: E402


def load_data(path=config.PROCESSED_DATA_PATH):
    """Return features X and target y from the processed CSV."""
    df = pd.read_csv(path)
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return X, y


def split_data(X, y):
    """Split into train and test sets using params.yaml."""
    p = config.PARAMS["training"]

    return train_test_split(
        X,
        y,
        test_size=p["test_size"],
        random_state=p["random_state"],
        stratify=y,
    )


def build_model() -> RandomForestClassifier:
    """Create the model from params.yaml."""
    m = config.PARAMS["model"]

    return RandomForestClassifier(
        n_estimators=m["n_estimators"],
        max_depth=m["max_depth"],
        random_state=m["random_state"],
    )


def evaluate(model, X_test, y_test) -> dict:
    """Return evaluation metrics for the 3-class Wine dataset."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="macro")
        ),
        "recall": float(
            recall_score(y_test, y_pred, average="macro")
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, average="macro")
        ),
    }


def save_reports(model, X_test, y_test, metrics: dict) -> None:
    """Write metrics.json, confusion_matrix.png and classification_report.txt."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    y_pred = model.predict(X_test)

    class_names = [
        "class_0",
        "class_1",
        "class_2",
    ]

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
    )

    (config.REPORTS_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
    )

    plt.title("Wine Dataset - Confusion Matrix")
    plt.savefig(
        config.REPORTS_DIR / "confusion_matrix.png",
        bbox_inches="tight",
    )
    plt.close()


def train() -> dict:
    """Run the full training pipeline and return the metrics."""
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    # Save the model file
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    # Save report files
    save_reports(model, X_test, y_test, metrics)

    # Log to MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(
        config.PARAMS["mlflow"]["experiment_name"]
    )

    with mlflow.start_run() as run:
        mlflow.log_param(
            "model_type",
            config.PARAMS["model"]["type"],
        )
        mlflow.log_param(
            "n_estimators",
            config.PARAMS["model"]["n_estimators"],
        )
        mlflow.log_param(
            "max_depth",
            config.PARAMS["model"]["max_depth"],
        )
        mlflow.log_param(
            "random_state",
            config.PARAMS["model"]["random_state"],
        )
        mlflow.log_param(
            "test_size",
            config.PARAMS["training"]["test_size"],
        )
        mlflow.log_param(
            "model_version_tag",
            config.MODEL_VERSION,
        )
        mlflow.log_param(
            "dataset",
            "Wine",
        )
        mlflow.log_param(
            "num_classes",
            3,
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        mlflow.log_artifacts(
            str(config.REPORTS_DIR),
            artifact_path="reports",
        )

        print(f"MLflow run ID: {run.info.run_id}")

    print("Metrics:")

    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print(f"Model saved to: {config.MODEL_PATH}")

    return metrics


if __name__ == "__main__":
    train()