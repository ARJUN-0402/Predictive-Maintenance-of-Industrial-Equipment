import math

import pandas as pd
import pytest

from src.feature_engineering import engineer_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [320.0],
            "Rotational speed [rpm]": [1000.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
        }
    )


def test_power_formula_known_inputs() -> None:
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [320.0],
            "Rotational speed [rpm]": [1000.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
        }
    )
    result = engineer_features(df)
    expected_power = 50.0 * 1000.0 * (2 * math.pi / 60)
    assert (
        abs(result["power"].iloc[0] - expected_power) < 0.01
    ), f"Expected power={expected_power}, got {result['power'].iloc[0]}"


def test_temperature_diff_known_inputs() -> None:
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [320.0],
            "Rotational speed [rpm]": [1000.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
        }
    )
    result = engineer_features(df)
    assert result["temperature_diff"].iloc[0] == 20.0


def test_wear_rate_known_inputs() -> None:
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [320.0],
            "Rotational speed [rpm]": [1000.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
        }
    )
    result = engineer_features(df)
    expected_wear_rate = 100.0 / (1000.0 + 1e-6)
    assert abs(result["wear_rate"].iloc[0] - expected_wear_rate) < 1e-6


def test_torque_normalized_known_inputs() -> None:
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [320.0],
            "Rotational speed [rpm]": [1000.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
        }
    )
    result = engineer_features(df)
    expected_torque_norm = 50.0 / (1000.0 + 1e-6)
    assert abs(result["torque_normalized"].iloc[0] - expected_torque_norm) < 1e-6


def test_temp_wear_interaction_known_inputs() -> None:
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [320.0],
            "Rotational speed [rpm]": [1000.0],
            "Torque [Nm]": [50.0],
            "Tool wear [min]": [100.0],
        }
    )
    result = engineer_features(df)
    expected_interaction = 20.0 * 100.0
    assert abs(result["temp_wear_interaction"].iloc[0] - expected_interaction) < 0.01


def test_shape_increases_by_five(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    assert result.shape[1] == sample_df.shape[1] + 5


def test_new_columns_added(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    assert "temperature_diff" in result.columns
    assert "power" in result.columns
    assert "wear_rate" in result.columns
    assert "torque_normalized" in result.columns
    assert "temp_wear_interaction" in result.columns
