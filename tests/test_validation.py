"""Tests for the model validation gate."""

from src.model_validation import check_metrics

THRESHOLDS = {
    "min_accuracy": 0.90,
    "min_precision": 0.90,
    "min_recall": 0.90,
    "min_f1": 0.90,
}

GOOD_METRICS = {
    "accuracy": 0.95,
    "precision": 0.95,
    "recall": 0.95,
    "f1_score": 0.95,
}


def test_good_metrics_pass():
    assert check_metrics(GOOD_METRICS, THRESHOLDS) == []


def test_low_accuracy_fails():
    metrics = {**GOOD_METRICS, "accuracy": 0.80}
    failures = check_metrics(metrics, THRESHOLDS)
    assert len(failures) == 1
    assert "accuracy" in failures[0]


def test_multiple_low_metrics_all_reported():
    metrics = {**GOOD_METRICS, "recall": 0.5, "f1_score": 0.5}
    assert len(check_metrics(metrics, THRESHOLDS)) == 2


def test_missing_metric_fails():
    metrics = {k: v for k, v in GOOD_METRICS.items() if k != "f1_score"}
    failures = check_metrics(metrics, THRESHOLDS)
    assert any("missing" in f for f in failures)


def test_metric_exactly_at_threshold_passes():
    metrics = {**GOOD_METRICS, "accuracy": 0.90}
    assert check_metrics(metrics, THRESHOLDS) == []