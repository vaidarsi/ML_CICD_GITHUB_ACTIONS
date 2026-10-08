"""Load the trained Wine model and make predictions."""

import joblib
import pandas as pd

from src import config


LABELS = {
    0: "class_0",
    1: "class_1",
    2: "class_2",
}


FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]


def load_model(path=config.MODEL_PATH):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def predict_one(model, features: list[float]) -> str:
    """Predict the Wine class for one sample of 13 feature values."""

    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(features)}")

    row = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES,
    )

    label = int(model.predict(row)[0])

    return LABELS[label]


if __name__ == "__main__":
    df = pd.read_csv(config.RAW_DATA_PATH)

    sample = df.drop(columns=[config.TARGET_COLUMN]).iloc[0].tolist()

    prediction = predict_one(
        load_model(),
        sample,
    )

    actual = LABELS[int(df[config.TARGET_COLUMN].iloc[0])]

    print("Prediction:", prediction)
    print("Actual:", actual)
