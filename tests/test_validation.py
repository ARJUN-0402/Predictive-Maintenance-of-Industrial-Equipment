import pytest
import pandas as pd
from src.validation import (
    validate_schema,
    validate_data_types,
    validate_missing_values,
    validate_duplicate_rows,
    validate_categorical_values,
    validate_numeric_ranges,
    validate_target_leakage,
    validate_feature_columns,
    validate_dataset,
    ValidationError,
)


class TestValidationModule:
    """Tests for the validation module."""

    def test_validate_schema_with_valid_data(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Product ID": ["M_001"],
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "Machine failure": [0],
            "TWF": [0],
            "HDF": [0],
            "PWF": [0],
            "OSF": [0],
            "RNF": [0],
        })
        validate_schema(df)

    def test_validate_schema_missing_columns(self):
        df = pd.DataFrame({
            "UDI": [1],
        })
        with pytest.raises(ValidationError):
            validate_schema(df)

    def test_validate_schema_extra_columns(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Product ID": ["M_001"],
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "Machine failure": [0],
            "Extra column": [1],
        })
        with pytest.raises(ValidationError):
            validate_schema(df)

    def test_validate_data_types(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Product ID": ["M_001"],
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "Machine failure": [0],
            "TWF": [0],
            "HDF": [0],
            "PWF": [0],
            "OSF": [0],
            "RNF": [0],
        })
        validate_data_types(df)

    def test_validate_missing_values_no_missing(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Product ID": ["M_001"],
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "Machine failure": [0],
            "TWF": [0],
            "HDF": [0],
            "PWF": [0],
            "OSF": [0],
            "RNF": [0],
        })
        missing = validate_missing_values(df)
        assert len(missing) == 0

    def test_validate_duplicate_rows_no_dupes(self):
        df = pd.DataFrame({
            "UDI": [1, 2, 3],
            "Product ID": ["M_001", "L_002", "H_003"],
            "Type": ["M", "L", "H"],
            "Air temperature [K]": [300.0, 310.0, 305.0],
            "Process temperature [K]": [310.0, 312.0, 318.0],
            "Rotational speed [rpm]": [1500.0, 1200.0, 1800.0],
            "Torque [Nm]": [50.0, 30.0, 50.0],
            "Tool wear [min]": [100.0, 30.0, 80.0],
            "Machine failure": [0, 0, 1],
            "TWF": [0, 0, 1],
            "HDF": [0, 0, 1],
            "PWF": [0, 0, 0],
            "OSF": [0, 0, 0],
            "RNF": [0, 0, 0],
        })
        had_dupes = validate_duplicate_rows(df)
        assert had_dupes is False

    def test_validate_categorical_values(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Product ID": ["M_001"],
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "Machine failure": [0],
            "TWF": [0],
            "HDF": [0],
            "PWF": [0],
            "OSF": [0],
            "RNF": [0],
        })
        validate_categorical_values(df, "Type", {"M", "L", "H"})

    def test_validate_categorical_values_invalid(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Product ID": ["M_001"],
            "Type": ["X"],  # Invalid
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "Machine failure": [0],
            "TWF": [0],
            "HDF": [0],
            "PWF": [0],
            "OSF": [0],
            "RNF": [0],
        })
        with pytest.raises(ValidationError):
            validate_categorical_values(df, "Type", {"M", "L", "H"})

    def test_validate_numeric_ranges(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
        })
        validate_numeric_ranges(df, {
            "Air temperature [K]": (270.0, 330.0),
            "Process temperature [K]": (300.0, 340.0),
            "Rotational speed [rpm]": (0, 3000),
            "Torque [Nm]": (0, 150),
            "Tool wear [min]": (0, 500),
        })

    def test_validate_numeric_ranges_out_of_bounds(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Air temperature [K]": [500.0],  # Out of range
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
        })
        with pytest.raises(ValidationError):
            validate_numeric_ranges(df, {
                "Air temperature [K]": (270.0, 330.0),
            })

    def test_validate_target_leakage_ok(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
        })
        validate_target_leakage(df, feature_columns=["Air temperature [K]"])

    def test_validate_target_leakage_error(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Air temperature [K]": [300.0],
            "TWF": [1],  # Post-failure label
        })
        with pytest.raises(ValidationError):
            validate_target_leakage(df, feature_columns=["Air temperature [K]", "TWF"])

    def test_validate_feature_columns_all_present(self):
        df = pd.DataFrame({
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "temperature_diff": [20.0],
            "power": [5000.0],
            "wear_rate": [0.1],
            "torque_normalized": [0.05],
            "temp_wear_interaction": [2000.0],
        })
        validate_feature_columns(df, ["Air temperature [K]", "temperature_diff"])

    def test_validate_feature_columns_missing(self):
        df = pd.DataFrame({
            "Air temperature [K]": [300.0],
        })
        with pytest.raises(ValidationError):
            validate_feature_columns(df, ["Air temperature [K]", "Missing column"])

    def test_validate_dataset_minimal(self):
        df = pd.DataFrame({
            "UDI": [1],
            "Product ID": ["M_001"],
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "Machine failure": [0],
            "TWF": [0],
            "HDF": [0],
            "PWF": [0],
            "OSF": [0],
            "RNF": [0],
        })
        validate_dataset(df, check_feature_columns=False)

    def test_validate_dataset_with_features(self):
        df = pd.DataFrame({
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
            "Type_L": [1],
            "Type_M": [0],
            "Type_H": [0],
            "temperature_diff": [20.0],
            "power": [5000.0],
            "wear_rate": [0.1],
            "torque_normalized": [0.05],
            "temp_wear_interaction": [2000.0],
        })
        validate_dataset(df, check_feature_columns=True, skip_schema=True)
