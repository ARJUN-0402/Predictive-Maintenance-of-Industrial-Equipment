import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.evaluate import compute_all_metrics, compare_models


def _make_mock_model(y_pred: np.ndarray, y_prob: np.ndarray) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = y_pred
    model.predict_proba.return_value = np.column_stack([1 - y_prob, y_prob])
    return model


@pytest.fixture
def mock_eval_data() -> dict:
    rng = np.random.default_rng(42)
    n = 200
    n_features = 13
    feature_names = [f"f{i}" for i in range(n_features)]

    X_test = pd.DataFrame(
        rng.standard_normal((n, n_features)), columns=feature_names
    )
    y_test = pd.Series(rng.integers(0, 2, size=n), name="Machine failure")

    y_prob = rng.uniform(0.0, 1.0, size=n)
    y_pred = (y_prob >= 0.5).astype(int)

    models = {
        name: _make_mock_model(y_pred, y_prob)
        for name in ("xgboost", "random_forest", "gradient_boosting", "logistic_regression")
    }

    metrics = {
        name: {
            "accuracy": 0.85,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "roc_auc": 0.9,
        }
        for name in models
    }

    return {
        "models": models,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
    }


class TestEvaluation:
    """Tests for the evaluation module."""

    def test_compute_all_metrics_basic(self):
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.4, 0.3, 0.8])
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_compute_all_metrics_perfect(self):
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2])
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] > 0.5

    def test_compute_all_metrics_imbalanced(self):
        y_true = pd.Series([0] * 9 + [1])  # 90% normal, 10% failure
        y_pred = np.array([0] * 10)  # Always predict normal
        y_prob = np.array([0.1] * 10)
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert metrics["recall"] == 0.0  # Missed the failure
        assert metrics["precision"] == 0.0  # No true positives

    def test_run_evaluation_returns_df(self, mock_eval_data: dict, tmp_path: Path):
        from src.evaluate import run_evaluation

        comparison_df = run_evaluation(
            mock_eval_data["models"],
            mock_eval_data["X_test"],
            mock_eval_data["y_test"],
            output_dir=tmp_path,
        )
        assert isinstance(comparison_df, pd.DataFrame)
        assert "roc_auc" in comparison_df.columns

    def test_compare_models_has_roc_auc(self, mock_eval_data: dict):
        comparison_df = compare_models(mock_eval_data["metrics"])
        assert "roc_auc" in comparison_df.columns
        top_model = comparison_df.index[0]
        assert top_model in comparison_df.index


class TestThresholdOptimization:
    """Tests for threshold optimization functionality."""

    def test_optimize_threshold_returns_dict(self):
        import warnings
        warnings.filterwarnings("ignore")
        from src.threshold_optimization import optimize_threshold

        rng = np.random.default_rng(42)
        y_true = pd.Series(rng.integers(0, 2, size=200))
        y_prob = rng.uniform(0.05, 0.95, size=200)

        result = optimize_threshold(y_true, y_prob)
        assert "best_threshold" in result
        assert "best_f1" in result
        assert "best_recall" in result
        assert "best_precision" in result

    def test_optimize_threshold_synthetic_analysis(self):
        from src.threshold_optimization import optimize_threshold

        rng = np.random.default_rng(0)
        y_true = pd.Series(rng.integers(0, 2, size=200))
        y_prob = rng.uniform(0.05, 0.95, size=200)

        result = optimize_threshold(y_true, y_prob, threshold_range=(0.05, 0.95), step=0.05)
        assert "best_threshold" in result
        assert "best_f1" in result
        assert "best_precision" in result
        assert "best_recall" in result
        assert "threshold_data" in result

        data = result["threshold_data"]
        for col in ("threshold", "precision", "recall", "f1", "false_negatives"):
            assert col in data.columns

        assert abs(float(data["f1"].max()) - result["best_f1"]) < 1e-9

        assert len(data) > 1
        assert data["f1"].between(0.0, 1.0).all()
        assert data["precision"].between(0.0, 1.0).all()
        assert data["recall"].between(0.0, 1.0).all()

        import warnings
        warnings.filterwarnings("ignore")
        from src.threshold_optimization import recommend_threshold_for_recall

        result = recommend_threshold_for_recall(y_true, y_prob, minimum_recall=0.80)
        assert "threshold" in result
        assert "recall" in result
        assert "precision" in result
        assert "f1" in result
