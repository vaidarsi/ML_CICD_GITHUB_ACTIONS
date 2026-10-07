"""Export the Breast Cancer Wisconsin dataset to data/raw/dataset.csv."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer

OUTPUT_PATH = Path("data/raw/dataset.csv")


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    df = data.frame  # 30 feature columns + a "target" column
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Target counts:\n{df['target'].value_counts()}")
    print("Target meaning: 0 = malignant, 1 = benign")


if __name__ == "__main__":
    main()