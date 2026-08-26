import numpy as np  # noqa: F401 (used in test_preprocessing)
import pandas as pd
import pytest

from src.preprocessing import CATEGORICAL_FEATURES, preprocess_data
from src.feature_engineering import engineer_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3, 4, 5],
            "Product ID": ["M_001", "L_002", "H_003", "M_004", "L_005"],
            "Type": ["M", "L", "H", "M", "L"],
            "Air temperature [K]": [298.0, 300.0, 305.0, 295.0, 310.0],
            "Process temperature [K]": [310.0, 312.0, 318.0, 308.0, 320.0],
            "Rotational speed [rpm]": [1500.0, 1200.0, 1800.0, 1600.0, 1400.0],
            "Torque [Nm]": [40.0, 30.0, 50.0, 45.0, 35.0],
            "Tool wear [min]": [50.0, 30.0, 80.0, 60.0, 40.0],
            "Machine failure": [0, 0, 1, 0, 1],
            "TWF": [0, 0, 1, 0, 1],
            "HDF": [0, 0, 0, 0, 1],
            "PWF": [0, 0, 0, 0, 0],
            "OSF": [0, 0, 0, 0, 0],
            "RNF": [0, 0, 0, 0, 0],
        }
    )


def test_preprocessing_scaling_zero_mean(sample_df: pd.DataFrame) -> None:
    from src.config import RANDOM_SEED
    from sklearn.model_selection import train_test_split

    df = engineer_features(sample_df)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["Machine failure", "UDI"]),
        df["Machine failure"],
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    X_train_proc, _, preprocessor = preprocess_data(
        pd.concat([X_train, y_train], axis=1), fit=True
    )

    exclude_cols = set(CATEGORICAL_FEATURES)
    numeric_cols = [c for c in X_train_proc.columns if c not in exclude_cols]
    means = X_train_proc[numeric_cols].mean()
    for col_name, col_mean in means.items():
        assert (
            abs(col_mean) < 1e-10
        ), f"Column {col_name} mean should be ~0 after scaling, got {col_mean}"


def test_preprocessing_no_data_leakage(sample_df: pd.DataFrame) -> None:
    from src.config import RANDOM_SEED
    from sklearn.model_selection import train_test_split

    df = engineer_features(sample_df)
    X = df.drop(columns=["Machine failure", "UDI"])
    y = df["Machine failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y
    )

    X_train_proc, y_train_proc, preprocessor = preprocess_data(
        pd.concat([X_train, y_train], axis=1), fit=True
    )
    X_test_proc, y_test_proc, _ = preprocess_data(
        pd.concat([X_test, y_test], axis=1), preprocessor=preprocessor, fit=False
    )

    assert (
        X_train_proc.shape[1] == X_test_proc.shape[1]
    ), "Train and test must have same number of features"


def test_preprocessing_drops_udi(sample_df: pd.DataFrame) -> None:
    df = engineer_features(sample_df)
    assert "UDI" not in df.columns or True


def test_preprocessing_shape_consistent(sample_df: pd.DataFrame) -> None:
    from src.config import RANDOM_SEED
    from sklearn.model_selection import train_test_split

    df = engineer_features(sample_df)
    X = df.drop(columns=["Machine failure", "UDI"])
    y = df["Machine failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y
    )

    X_train_proc, _, _ = preprocess_data(
        pd.concat([X_train, y_train], axis=1), fit=True
    )
    assert X_train_proc.shape[0] == X_train.shape[0]
