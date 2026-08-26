import logging

import pandas as pd

from src.config import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    DROP_COLUMNS,
    FEATURE_COLUMNS,
)

logger = logging.getLogger("validation")


class ValidationError(ValueError):
    """Raised when dataset validation fails."""


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has the expected schema.

    Checks:
    - All expected columns are present (no missing columns)
    - No unexpected columns (extra columns beyond what's expected)
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValidationError(f"Missing columns: {sorted(missing)}")
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        raise ValidationError(f"Unexpected columns: {sorted(extra)}")
    logger.info("Schema validation passed: all expected columns present, no unexpected columns")


def validate_data_types(df: pd.DataFrame) -> None:
    """Validate that columns have the expected data types.

    Expected types:
    - UDI: int
    - Product ID: str
    - Type: str
    - Air temperature [K]: float
    - Process temperature [K]: float
    - Rotational speed [rpm]: float
    - Torque [Nm]: float
    - Tool wear [min]: float
    - Machine failure: int (0 or 1)
    - TWF: int (0 or 1)
    - HDF: int (0 or 1)
    - PWF: int (0 or 1)
    - OSF: int (0 or 1)
    - RNF: int (0 or 1)
    """
    type_mapping = {
        "UDI": int,
        "Product ID": str,
        "Type": str,
        "Air temperature [K]": float,
        "Process temperature [K]": float,
        "Rotational speed [rpm]": float,
        "Torque [Nm]": float,
        "Tool wear [min]": float,
        "Machine failure": int,
        "TWF": int,
        "HDF": int,
        "PWF": int,
        "OSF": int,
        "RNF": int,
    }

    for col, expected_type in type_mapping.items():
        if col not in df.columns:
            continue
        actual_type = df[col].dtype
        if expected_type is int:
            if not pd.api.types.is_integer_dtype(actual_type):
                try:
                    pd.to_numeric(df[col], errors="raise")
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Column '{col}' expected to be integer, got {actual_type}"
                    )
        elif expected_type is float:
            if not pd.api.types.is_float_dtype(actual_type):
                try:
                    pd.to_numeric(df[col], errors="raise")
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Column '{col}' expected to be float, got {actual_type}"
                    )
    logger.info("Data type validation passed")


def validate_missing_values(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Detect and report missing values.

    Args:
        df: DataFrame to validate
        threshold: Fraction of rows above which to flag a column (default 10%)

    Returns:
        DataFrame with missing value counts and fractions
    """
    missing_count = df.isnull().sum()
    missing_frac = missing_count / len(df)
    missing_summary = pd.DataFrame({
        "column": missing_count.index[missing_count > 0],
        "missing_count": missing_count[missing_count > 0],
        "missing_fraction": missing_frac[missing_frac > 0],
    })

    if len(missing_summary) > 0:
        flagged = missing_summary[missing_summary["missing_fraction"] > threshold]
        if len(flagged) > 0:
            raise ValidationError(
                f"Missing values found in {len(flagged)} columns exceeding "
                f"{threshold*100}% threshold: {', '.join(flagged['column'].astype(str))}"
            )
        logger.warning(
            "Missing values detected (below threshold) in columns: %s",
            ", ".join(missing_summary["column"].astype(str)),
        )

    return missing_summary


def validate_duplicate_rows(df: pd.DataFrame) -> bool:
    """Detect duplicate rows in the dataset.

    Returns:
        True if duplicates were found and removed, False otherwise
    """
    original_len = len(df)
    df_deduped = df.drop_duplicates()
    deduped_len = len(df_deduped)
    removed = original_len - deduped_len

    if removed > 0:
        logger.warning(
            "Found %d duplicate rows, removed %d rows from dataset",
            original_len - deduped_len, removed
        )
        return True
    else:
        logger.info("No duplicate rows found")
        return False


def validate_categorical_values(df: pd.DataFrame, column: str, valid_values: set) -> None:
    """Validate that a categorical column only contains expected values.

    Args:
        df: DataFrame to validate
        column: Column name to check
        valid_values: Set of valid values for the column
    """
    if column not in df.columns:
        raise ValidationError(f"Column '{column}' not found in DataFrame")

    invalid = set(df[column].unique()) - valid_values
    if invalid:
        raise ValidationError(
            f"Column '{column}' has invalid values: {sorted(invalid)}. "
            f"Expected: {sorted(valid_values)}"
        )
    logger.info("Categorical validation passed for column '%s'", column)


def validate_numeric_ranges(df: pd.DataFrame, ranges: dict) -> None:
    """Validate that numeric columns have sensible values.

    Args:
        df: DataFrame to validate
        ranges: Dict mapping column names to (min, max) tuples
                e.g., {"Air temperature [K]": (270.0, 320.0)}
    """
    for column, (min_val, max_val) in ranges.items():
        if column not in df.columns:
            raise ValidationError(f"Column '{column}' not found in DataFrame")

        col_data = df[column]
        if not pd.api.types.is_numeric_dtype(col_data):
            raise ValidationError(f"Column '{column}' is not numeric")

        out_of_range = col_data[(col_data < min_val) | (col_data > max_val)]
        if len(out_of_range) > 0:
            raise ValidationError(
                f"Column '{column}' has {len(out_of_range)} values outside "
                f"valid range ({min_val}, {max_val})"
            )
    logger.info("Numeric range validation passed for %d columns", len(ranges))


def validate_target_leakage(df: pd.DataFrame, feature_columns: list[str] | None = None) -> None:
    """Validate that post-failure label columns are not used as features.

    The following columns MUST NOT be used as predictive features as they
    are post-failure indicators (would cause data leakage):
    - TWF
    - HDF
    - PWF
    - OSF
    - RNF

    Args:
        df: DataFrame to validate
        feature_columns: List of feature column names. If None, uses
            FEATURE_COLUMNS from config.
    """
    leakage_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    feat_cols = set(feature_columns) if feature_columns else set()

    leakage_in_features = set(leakage_cols) & feat_cols
    if leakage_in_features:
        raise ValidationError(
            f"Target leakage detected: columns {sorted(leakage_in_features)} "
            f"are post-failure labels and must not be used as predictive features. "
            f"They will be dropped during preprocessing."
        )

    for col in leakage_cols:
        if col in df.columns and col not in feat_cols:
            logger.info(
                "Column '%s' present in data but not in features (will be dropped)",
                col,
            )

    logger.info("Target leakage validation passed: no post-failure labels in features")


def validate_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Validate that the specified feature columns exist in the DataFrame.

    Also checks that target column and dropped columns are not in the feature list.
    """
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValidationError(f"Missing feature columns: {sorted(missing)}")

    forbidden = [c for c in DROP_COLUMNS + [TARGET_COLUMN] if c in feature_columns]
    if forbidden:
        raise ValidationError(
            f"Feature column(s) {sorted(forbidden)} should not be included as "
            f"features (target or dropped columns)"
        )

    logger.info("Feature column validation passed: %d features", len(feature_columns))


def validate_dataset(df: pd.DataFrame, check_feature_columns: bool = False, skip_schema: bool = False) -> None:  # noqa: E501
    """Run all validation checks on a dataset.

    Args:
        df: DataFrame to validate
        check_feature_columns: If True, validates that FEATURE_COLUMNS
            exist in the DataFrame (use after feature engineering).
            If False, only validates schema, types, missing values,
            duplicates, and target leakage for the raw dataset.
        skip_schema: If True, skips schema validation (use for datasets
            that don't have the full raw AI4I 2020 column set).
    """
    if not skip_schema:
        validate_schema(df)
    validate_data_types(df)
    validate_duplicate_rows(df)
    validate_target_leakage(df, feature_columns=FEATURE_COLUMNS if check_feature_columns else None)

    if check_feature_columns:
        validate_feature_columns(df, FEATURE_COLUMNS)

    logger.info("Full dataset validation completed successfully")


__all__ = [
    "ValidationError",
    "validate_schema",
    "validate_data_types",
    "validate_missing_values",
    "validate_duplicate_rows",
    "validate_categorical_values",
    "validate_numeric_ranges",
    "validate_target_leakage",
    "validate_feature_columns",
    "validate_dataset",
]
