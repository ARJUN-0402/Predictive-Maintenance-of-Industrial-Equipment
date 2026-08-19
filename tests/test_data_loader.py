import pandas as pd
from src.data_loader import load_dataset
from src.validation import (
    validate_duplicate_rows,
    validate_dataset,
    validate_missing_values,
    validate_schema,
    validate_target_leakage,
)


class TestDataLoader:
    """Tests for the data loader module."""

    def test_load_dataset_returns_dataframe(self):
        df = load_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_dataset_has_expected_columns(self):
        df = load_dataset()
        assert "Machine failure" in df.columns
        assert "Air temperature [K]" in df.columns

    def test_load_dataset_shape(self):
        df = load_dataset()
        assert df.shape[0] == 10000
        assert df.shape[1] == 14


class TestValidation:
    """Tests for the validation module."""

    def test_validate_schema_positive(self):
        df = load_dataset()
        validate_schema(df)

    def test_validate_target_leakage_negative(self):
        df = load_dataset()
        validate_target_leakage(df, feature_columns=[
            "Air temperature [K]",
            "Process temperature [K]",
        ])

    def test_validate_dataset_full(self):
        df = load_dataset()
        validate_dataset(df, check_feature_columns=False)

    def test_validate_missing_values(self):
        df = load_dataset()
        missing = validate_missing_values(df)
        assert isinstance(missing, pd.DataFrame)

    def test_validate_duplicate_rows(self):
        df = load_dataset()
        had_dupes = validate_duplicate_rows(df)
        assert isinstance(had_dupes, bool)
