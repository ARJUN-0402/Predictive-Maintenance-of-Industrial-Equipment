import joblib

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    DROP_COLUMNS,
)
from src.utils import setup_logging

CATEGORICAL_FEATURES = ["Type_L", "Type_M", "Type_H"]

logger = setup_logging("preprocessor")


def clean_feature_names(names: list[str]) -> list[str]:
    """Clean feature names to be XGBoost-compatible (no brackets, <, >)."""
    cleaned = []
    for name in names:
        n = name.replace("[", "").replace("]", "").replace("<", "").replace(">", "")
        cleaned.append(n)
    return cleaned


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def preprocess_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    df = df.drop_duplicates()
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    df["Type"] = df["Type"].astype(str)
    type_dummies = pd.get_dummies(df["Type"], prefix="Type", dtype=int)
    for col in CATEGORICAL_FEATURES:
        if col not in type_dummies.columns:
            type_dummies[col] = 0
    df = pd.concat([df, type_dummies], axis=1)
    df = df.drop(columns=["Type"])

    target = df[TARGET_COLUMN]
    feature_cols = [c for c in FEATURE_COLUMNS if c not in (TARGET_COLUMN)]
    X = df[feature_cols]

    if preprocessor is None:
        preprocessor = build_preprocessing_pipeline()

    if fit:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    if isinstance(X_transformed, pd.DataFrame):
        feature_names = clean_feature_names(list(X_transformed.columns))
        X_processed = X_transformed.copy()
        X_processed.columns = feature_names
    else:
        feature_names = clean_feature_names(list(preprocessor.get_feature_names_out()))
        X_processed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    logger.info("Preprocessing complete: shape %s", X_processed.shape)
    return X_processed, target, preprocessor


def save_processed_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> None:
    PROCESSED_DATA_DIR = PROCESSED_DATA_PATH.parent
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    test_df.to_parquet(TEST_DATA_PATH, index=False)
    logger.info("Saved processed data to %s", PROCESSED_DATA_DIR)

    joblib.dump(preprocessor, str(SCALER_PATH))
    logger.info("Saved preprocessor to %s", SCALER_PATH)


def load_processed_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    return X_train, y_train, X_test, y_test


__all__ = [
    "build_preprocessing_pipeline",
    "preprocess_data",
    "save_processed_data",
    "load_processed_data",
]
