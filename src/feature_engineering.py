import logging

import pandas as pd

from src.utils import setup_logging

logger = setup_logging("feature_engineering")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Feature engineering: shape before %s", df.shape)

    df = df.copy()
    df["temperature_diff"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    df["power"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * 3.141592653589793 / 60)
    )
    df["wear_rate"] = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1e-6)
    df["torque_normalized"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1e-6)
    df["temp_wear_interaction"] = df["temperature_diff"] * df["Tool wear [min]"]

    logger.info("Feature engineering: shape after %s", df.shape)
    return df


__all__ = ["engineer_features"]