from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
DATASET_FILENAME = "ai4i2020.csv"
DATASET_ZIP = "ai4i2020.zip"

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

TARGET_COLUMN = "Machine failure"
DROP_COLUMNS = ["UDI", "TWF", "HDF", "PWF", "OSF", "RNF"]

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
    "Type_H",
    "temperature_diff",
    "power",
    "wear_rate",
    "torque_normalized",
    "temp_wear_interaction",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temperature_diff",
    "power",
    "wear_rate",
    "torque_normalized",
    "temp_wear_interaction",
]

XGBoost_PARAMS = {
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "scale_pos_weight": [1, 5, 10],
}

XGBoost_N_JOBS = -1
XGBoost_EVAL_METRIC = "logloss"

TEST_SIZE = 0.2
STRATIFY = True

MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
XGBoost_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_data.parquet"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.parquet"
