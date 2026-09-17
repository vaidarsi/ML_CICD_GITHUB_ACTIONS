import pandas as pd

INPUT_FILE = "data/student_marks.csv"
OUTPUT_FILE = "data/processed.csv"


def preprocess_data():
    # Read the dataset
    df = pd.read_csv(INPUT_FILE)

    print("Original Dataset:")
    print(df)

    # Remove missing values
    df = df.dropna()

    # Save the processed dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing completed successfully.")
    print("Processed dataset shape:", df.shape)
    print("Processed file saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    preprocess_data()