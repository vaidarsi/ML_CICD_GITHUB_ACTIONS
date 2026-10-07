"""Load the trained model and make predictions."""

import joblib

from src import config

LABELS = {0: "malignant", 1: "benign"}

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


def load_model(path=config.MODEL_PATH):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_one(model, features: list[float]) -> str:
    """Predict the class label for one sample of 30 feature values."""
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"
        )
    import pandas as pd

    row = pd.DataFrame([features], columns=FEATURE_NAMES)
    label = int(model.predict(row)[0])
    return LABELS[label]


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(config.RAW_DATA_PATH)
    sample = df.drop(columns=[config.TARGET_COLUMN]).iloc[0].tolist()
    print("Prediction:", predict_one(load_model(), sample))
    print("Actual:", LABELS[int(df[config.TARGET_COLUMN].iloc[0])])