import json
import logging  # noqa: F401 (used indirectly via setup_logging)
import sys
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    MODEL_REGISTRY_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    XGBoost_PARAMS,
    XGBoost_EVAL_METRIC,
    XGBoost_N_JOBS,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    DROP_COLUMNS,
    XGBoost_MODEL_PATH,
)
from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess_data, save_processed_data
from src.utils import setup_logging

logger = setup_logging("train")


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    logger.info("Starting XGBoost GridSearchCV...")
    xgb = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric=XGBoost_EVAL_METRIC,
        n_jobs=XGBoost_N_JOBS,
    )
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=XGBoost_PARAMS,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    logger.info("XGBoost best params: %s", grid_search.best_params_)
    logger.info("XGBoost best CV ROC-AUC: %.4f", grid_search.best_score_)
    return grid_search.best_estimator_, grid_search.best_params_


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series
) -> GradientBoostingClassifier:  # noqa: E501
    logger.info("Training Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LogisticRegression:  # noqa: E501
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_model_registry(
    model_name: str,
    best_params: dict,
    metrics: dict,
    dataset_info: dict | None = None,
    feature_config: dict | None = None,
    threshold: float | None = None,
) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry: dict = {"versions": {}}
    if MODEL_REGISTRY_PATH.exists():
        with open(MODEL_REGISTRY_PATH, "r") as f:
            loaded = json.load(f)
        if "versions" in loaded:
            registry = loaded
        else:
            registry["versions"] = loaded
            if "dataset_info" in loaded:
                registry["shared"] = {
                    "dataset_info": loaded["dataset_info"],
                    "feature_config": loaded.get("feature_config", {}),
                }
            elif "shared" in loaded:
                registry["shared"] = loaded["shared"]

    versions = registry.setdefault("versions", {})

    version = f"v{len(versions) + 1}"
    entry: dict[str, object] = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "best_params": best_params,
        "metrics": metrics,
    }
    if threshold is not None:
        entry["threshold"] = threshold

    if (
        "shared" not in registry
        and dataset_info is not None
        and feature_config is not None
    ):
        registry["shared"] = {
            "dataset_info": dataset_info,
            "feature_config": feature_config,
        }
    elif "shared" not in registry and dataset_info is not None:
        registry["shared"] = {"dataset_info": dataset_info, "feature_config": {}}
    elif "shared" not in registry and feature_config is not None:
        registry["shared"] = {"dataset_info": {}, "feature_config": feature_config}

    versions[version] = entry

    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Model registry updated with %s", version)


def train_all_models() -> dict:
    logger.info("Loading dataset...")
    df = load_dataset()

    logger.info("Engineering features...")
    df = engineer_features(df)

    logger.info("Splitting data...")
    feature_cols = [
        c
        for c in df.columns
        if c not in ("Machine failure", "UDI", "TWF", "HDF", "PWF", "OSF", "RNF")
    ]  # noqa: E501
    X = df[feature_cols]
    y = df["Machine failure"]

    class_ratio = y.value_counts().to_dict()
    logger.info("Class distribution: %s", class_ratio)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    logger.info("Preprocessing data...")
    X_train_proc, y_train_proc, preprocessor = preprocess_data(train_df, fit=True)
    X_test_proc, y_test_proc, _ = preprocess_data(
        test_df, preprocessor=preprocessor, fit=False
    )

    save_processed_data(
        X_train_proc, y_train_proc, X_test_proc, y_test_proc, preprocessor
    )

    dataset_info = {
        "rows": len(df),
        "columns": list(df.columns),
        "class_balance": y.value_counts().to_dict(),
        "target_column": TARGET_COLUMN,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
    }

    feature_config = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": ["Type_L", "Type_M", "Type_H"],
        "drop_columns": DROP_COLUMNS,
    }

    models: dict = {}
    metrics_results: dict = {}

    logger.info("Training XGBoost...")
    xgb_model, xgb_params = train_xgboost(X_train_proc, y_train_proc)
    xgb_pred = xgb_model.predict(X_test_proc)
    xgb_prob = xgb_model.predict_proba(X_test_proc)[:, 1]
    xgb_metrics = compute_metrics(y_test_proc, xgb_pred, xgb_prob)
    models["xgboost"] = xgb_model
    metrics_results["xgboost"] = xgb_metrics
    save_model_registry(
        "xgboost",
        xgb_params,
        xgb_metrics,
        dataset_info=dataset_info,
        feature_config=feature_config,
    )
    logger.info("XGBoost metrics: %s", xgb_metrics)

    logger.info("Training Random Forest...")
    rf_model = train_random_forest(X_train_proc, y_train_proc)
    rf_pred = rf_model.predict(X_test_proc)
    rf_prob = rf_model.predict_proba(X_test_proc)[:, 1]
    rf_metrics = compute_metrics(y_test_proc, rf_pred, rf_prob)
    models["random_forest"] = rf_model
    metrics_results["random_forest"] = rf_metrics
    save_model_registry(
        "random_forest",
        {"n_estimators": 200, "class_weight": "balanced"},
        rf_metrics,
        dataset_info=dataset_info,
        feature_config=feature_config,
    )
    logger.info("Random Forest metrics: %s", rf_metrics)

    logger.info("Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train_proc, y_train_proc)
    gb_pred = gb_model.predict(X_test_proc)
    gb_prob = gb_model.predict_proba(X_test_proc)[:, 1]
    gb_metrics = compute_metrics(y_test_proc, gb_pred, gb_prob)
    models["gradient_boosting"] = gb_model
    metrics_results["gradient_boosting"] = gb_metrics
    save_model_registry(
        "gradient_boosting",
        {"n_estimators": 200},
        gb_metrics,
        dataset_info=dataset_info,
        feature_config=feature_config,
    )
    logger.info("Gradient Boosting metrics: %s", gb_metrics)

    logger.info("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_proc, y_train_proc)
    lr_pred = lr_model.predict(X_test_proc)
    lr_prob = lr_model.predict_proba(X_test_proc)[:, 1]
    lr_metrics = compute_metrics(y_test_proc, lr_pred, lr_prob)
    models["logistic_regression"] = lr_model
    metrics_results["logistic_regression"] = lr_metrics
    save_model_registry(
        "logistic_regression",
        {"class_weight": "balanced"},
        lr_metrics,
        dataset_info=dataset_info,
        feature_config=feature_config,
    )
    logger.info("Logistic Regression metrics: %s", lr_metrics)

    joblib.dump(models["xgboost"], str(XGBoost_MODEL_PATH))
    logger.info("Saved XGBoost model to %s", XGBoost_MODEL_PATH)

    return {
        "models": models,
        "metrics": metrics_results,
        "X_test": X_test_proc,
        "y_test": y_test_proc,
        "preprocessor": preprocessor,
        "dataset_info": dataset_info,
        "feature_config": feature_config,
    }


def run_full_experiment() -> dict[str, Any]:
    """Run the complete experiment: train → evaluate → threshold optimization.

    Convenience wrapper that chains the three pipeline stages without
    duplicating any logic. Each stage's functions are imported lazily
    to avoid circular imports at module load time.
    """
    logger.info("=== Stage 1/3: Training ===")
    results = train_all_models()

    logger.info("=== Stage 2/3: Evaluation ===")
    from src.evaluate import run_evaluation

    comparison_df = run_evaluation(
        results["models"], results["X_test"], results["y_test"]
    )
    results["comparison_df"] = comparison_df

    logger.info("=== Stage 3/3: Threshold Optimization ===")
    from src.threshold_optimization import (
        optimize_threshold,
        recommend_threshold_for_recall,
    )

    y_prob = results["models"]["xgboost"].predict_proba(results["X_test"])[:, 1]
    threshold_result = optimize_threshold(results["y_test"], y_prob)
    recall_result = recommend_threshold_for_recall(
        results["y_test"], y_prob, minimum_recall=0.80
    )
    results["threshold_result"] = threshold_result
    results["recall_result"] = recall_result

    return results


def main() -> int:
    """CLI entry point for training.

    Trains all models, saves artifacts, and prints a summary of test-set
    metrics to stdout. Returns 0 on success, 1 on failure.
    """
    try:
        results = train_all_models()
        print("\n=== Training Complete ===\n")
        for name, m in results["metrics"].items():
            print(
                f"  {name:20s}  acc={m['accuracy']:.4f}  "
                f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  "
                f"f1={m['f1']:.4f}  roc_auc={m['roc_auc']:.4f}"
            )
        print()
        return 0
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
