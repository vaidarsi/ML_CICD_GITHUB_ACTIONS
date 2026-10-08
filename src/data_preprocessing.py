"""Load the raw dataset, clean it, and save the processed version."""

import pandas as pd

from src import config


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and rows with missing values."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, path=config.PROCESSED_DATA_PATH) -> None:
    """Write the processed DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run() -> pd.DataFrame:
    """Full preprocessing step: load -> clean -> save."""
    df = load_raw_data()
    clean_df = preprocess(df)
    save_processed_data(clean_df)
    print(f"Raw rows: {len(df)} | Processed rows: {len(clean_df)}")
    print(f"Saved: {config.PROCESSED_DATA_PATH}")
    return clean_df


if __name__ == "__main__":
    run()
