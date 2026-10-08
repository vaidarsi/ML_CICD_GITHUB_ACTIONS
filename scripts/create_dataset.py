"""Export the Wine dataset to data/raw/dataset.csv."""

from pathlib import Path

from sklearn.datasets import load_wine

OUTPUT_PATH = Path("data/raw/dataset.csv")


def main() -> None:
    data = load_wine(as_frame=True)
    df = data.frame  # 13 feature columns + a "target" column

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Target counts:\n{df['target'].value_counts().sort_index()}")
    print(
        "Target meaning: 0 = class_0, "
        "1 = class_1, "
        "2 = class_2 (three wine cultivars)"
    )


if __name__ == "__main__":
    main()
