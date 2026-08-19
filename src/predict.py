import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MODEL_REGISTRY_PATH,
    PROCESSED_DATA_PATH,
    SCALER_PATH,
    XGBoost_MODEL_PATH,
)
from src.feature_engineering import engineer_features
from src.utils import setup_logging

logger = setup_logging("predict")


def load_model(
    model_path: Path | None = None,
    scaler_path: Path | None = None,
) -> tuple:
    if model_path is None:
        model_path = XGBoost_MODEL_PATH
    if scaler_path is None:
        scaler_path = SCALER_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    model = joblib.load(str(model_path))
    preprocessor = joblib.load(str(scaler_path))
    return model, preprocessor


def prepare_input_df(
    raw_data: pd.DataFrame | dict[str, float],
) -> pd.DataFrame:
    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])
    else:
        df = raw_data.copy()
    return df


def predict(
    input_data: pd.DataFrame | dict[str, float],
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    threshold: float = 0.5,
) -> dict:
    model, preprocessor = load_model(model_path, scaler_path)
    df = prepare_input_df(input_data)

    # Engineer features (must match training pipeline)
    df = engineer_features(df)

    if "Type" in df.columns:
        type_dummies = pd.get_dummies(df["Type"], prefix="Type", dtype=int)
        for col in ["Type_L", "Type_M", "Type_H"]:
            if col not in type_dummies.columns:
                type_dummies[col] = 0
        df = pd.concat([df, type_dummies], axis=1)
        df = df.drop(columns=["Type"])

    feature_cols = [c for c in FEATURE_COLUMNS if c not in ("Machine failure", "Type")]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]

    X_scaled = preprocessor.transform(X)

    # Clean feature names to match XGBoost model expectations
    # The preprocessor may produce columns with original names;
    # ensure they match the booster's expected feature names
    booster = model.get_booster()
    expected_features = booster.feature_names
    current_features = list(X_scaled.columns)

    if current_features != expected_features:
        # Rename columns to match the model's expected feature names
        rename_dict = dict(zip(current_features, expected_features))
        X_scaled = X_scaled.rename(columns=rename_dict)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    results = []
    for i in range(len(y_prob)):
        results.append(
            {
                "prediction": int(y_pred[i]),
                "probability": float(y_prob[i]),
                "risk_level": _classify_risk(y_prob[i], threshold),
            }
        )
    return results[0] if len(results) == 1 else results


def _classify_risk(probability: float, threshold: float = 0.5) -> str:
    if probability > 0.7:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    return "Low"


def batch_predict(
    csv_path: Path,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    results = []
    for idx, row in df.iterrows():
        try:
            result = predict(row.to_dict(), model_path, scaler_path, threshold)
            results.append(result)
        except Exception as exc:
            logger.error("Error predicting row %d: %s", idx, exc)
            results.append(
                {"prediction": -1, "probability": 0.0, "risk_level": "Error"}
            )
    return pd.DataFrame(results)


__all__ = ["load_model", "prepare_input_df", "predict", "batch_predict"]