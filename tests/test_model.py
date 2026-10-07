"""Tests for training, evaluation and prediction."""

import pytest

from src import config
from src.data_preprocessing import load_raw_data
from src.predict import FEATURE_NAMES, LABELS, load_model, predict_one
from src.train import build_model, evaluate, split_data


@pytest.fixture(scope="module")
def data():
    df = load_raw_data()
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return X, y


@pytest.fixture(scope="module")
def trained(data):
    """Train a model in memory. Writes nothing to models/ or reports/."""
    X, y = data
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_split_sizes(data):
    X, y = data

    X_train, X_test, _, _ = split_data(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == pytest.approx(len(X) * 0.2, abs=1)


def test_model_trains_and_predicts(trained):
    model, X_test, _ = trained

    preds = model.predict(X_test)

    assert len(preds) == len(X_test)

    # Wine dataset has 3 classes: 0, 1, 2
    assert set(preds) <= {0, 1, 2}


def test_metrics_are_generated(trained):
    model, X_test, y_test = trained

    metrics = evaluate(model, X_test, y_test)

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_trained_model_beats_random_guessing(trained):
    model, X_test, y_test = trained

    assert evaluate(model, X_test, y_test)["accuracy"] > 0.8


def test_feature_names_match_dataset(data):
    X, _ = data

    assert list(X.columns) == FEATURE_NAMES


def test_predict_one_returns_valid_label(trained, data):
    model, _, _ = trained
    X, _ = data

    label = predict_one(
        model,
        X.iloc[0].tolist(),
    )

    assert label in LABELS.values()


def test_predict_one_rejects_wrong_feature_count(trained):
    model, _, _ = trained

    with pytest.raises(ValueError):
        predict_one(
            model,
            [1.0, 2.0, 3.0],
        )


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")