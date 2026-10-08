"""Tests for dataset loading, preprocessing and data validation."""

import pandas as pd
import pytest

from src import config
from src.data_preprocessing import load_raw_data, preprocess
from src.data_validation import validate_dataframe, validate_file


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_raw_data()


def test_dataset_file_exists():
    assert config.RAW_DATA_PATH.exists()


def test_dataset_loads_with_enough_rows(raw_df):
    assert len(raw_df) >= config.PARAMS["data"]["min_rows"]


def test_target_column_exists(raw_df):
    assert config.TARGET_COLUMN in raw_df.columns


def test_expected_number_of_columns(raw_df):
    assert raw_df.shape[1] == 14  # 13 features + target


def test_target_values_are_valid(raw_df):
    assert set(raw_df[config.TARGET_COLUMN].unique()) <= {0, 1, 2}


def test_real_dataset_passes_validation():
    assert validate_file() == []


def test_validation_detects_missing_values(raw_df):
    df = raw_df.copy()
    df.loc[0, "alcohol"] = None

    errors = validate_dataframe(df)

    assert any("missing" in e for e in errors)


def test_validation_detects_invalid_target(raw_df):
    df = raw_df.copy()
    df.loc[0, config.TARGET_COLUMN] = 7

    errors = validate_dataframe(df)

    assert any("Invalid target" in e for e in errors)


def test_validation_detects_too_few_rows(raw_df):
    errors = validate_dataframe(raw_df.head(10))

    assert any("Too few rows" in e for e in errors)


def test_validation_detects_missing_target_column(raw_df):
    df = raw_df.drop(columns=[config.TARGET_COLUMN])

    errors = validate_dataframe(df)

    assert any("Missing target column" in e for e in errors)


def test_preprocess_removes_missing_and_duplicates(raw_df):
    df = raw_df.head(10).copy()

    df.loc[0, "alcohol"] = None
    df = pd.concat([df, df.iloc[[1]]])

    result = preprocess(df)

    assert len(result) == 9
    assert result.isnull().sum().sum() == 0
