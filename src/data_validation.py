"""Validate the raw dataset before training. Exits with code 1 on failure."""

import sys

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src import config

VALID_TARGET_VALUES = {0, 1}


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Return a list of problems found. An empty list means the data is valid."""
    errors: list[str] = []
    target = config.TARGET_COLUMN
    min_rows = config.PARAMS["data"]["min_rows"]

    # Required column
    if target not in df.columns:
        errors.append(f"Missing target column: '{target}'")
        return errors

    # Minimum size
    if len(df) < min_rows:
        errors.append(f"Too few rows: {len(df)} (minimum {min_rows})")

    # Missing values
    missing = int(df.isnull().sum().sum())
    if missing > 0:
        errors.append(f"Dataset has {missing} missing values")

    # Data types: every column must be numeric
    non_numeric = [c for c in df.columns if not is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")

    # Target values
    invalid = set(df[target].dropna().unique()) - VALID_TARGET_VALUES
    if invalid:
        errors.append(f"Invalid target values: {sorted(invalid)}")

    return errors


def validate_file(path=config.RAW_DATA_PATH) -> list[str]:
    """Validate the dataset file at `path`."""
    if not path.exists():
        return [f"Dataset not found: {path}"]
    return validate_dataframe(pd.read_csv(path))


def main() -> None:
    errors = validate_file()
    if errors:
        print("DATA VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("DATA VALIDATION PASSED")


if __name__ == "__main__":
    main()