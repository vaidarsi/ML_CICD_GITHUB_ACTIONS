"""Model validation gate: fail (exit code 1) if metrics are below thresholds."""

import json
import sys

from src import config

# metric name in metrics.json -> threshold name in params.yaml
THRESHOLDS = {
    "accuracy": "min_accuracy",
    "precision": "min_precision",
    "recall": "min_recall",
    "f1_score": "min_f1",
}


def load_metrics(path=config.METRICS_PATH) -> dict:
    """Read the metrics written by train.py."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_metrics(metrics: dict, thresholds: dict) -> list[str]:
    """Return a list of failures. An empty list means the model passed."""
    failures: list[str] = []
    for metric_name, threshold_name in THRESHOLDS.items():
        if metric_name not in metrics:
            failures.append(f"Metric '{metric_name}' is missing from metrics.json")
            continue
        minimum = thresholds[threshold_name]
        value = metrics[metric_name]
        if value < minimum:
            failures.append(f"{metric_name} = {value:.4f} is below minimum {minimum}")
    return failures


def main() -> None:
    if not config.METRICS_PATH.exists():
        print(f"MODEL VALIDATION FAILED: {config.METRICS_PATH} not found")
        sys.exit(1)

    metrics = load_metrics()
    failures = check_metrics(metrics, config.PARAMS["validation"])

    print("Metrics checked:")
    for name in THRESHOLDS:
        if name in metrics:
            print(f"  {name}: {metrics[name]:.4f}")

    if failures:
        print("MODEL VALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("MODEL VALIDATION PASSED")


if __name__ == "__main__":
    main()
